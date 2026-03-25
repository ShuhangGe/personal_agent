"""Subagent manager for expert-based task execution."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.expert_library import ExpertLibrary
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import build_assistant_message


_SAVE_EXPERT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_expert_profile",
            "description": "Save a new expert profile based on the completed task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expert_name": {
                        "type": "string",
                        "description": "Short kebab-case name for this expert (e.g. 'scrape-product-prices')",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-sentence description of what this expert does",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags for matching future tasks",
                    },
                    "approach": {
                        "type": "string",
                        "description": "Brief description of the approach and steps that worked",
                    },
                    "memory_notes": {
                        "type": "string",
                        "description": "Key facts learned during this task worth remembering for next time",
                    },
                },
                "required": ["expert_name", "description", "tags", "approach"],
            },
        },
    }
]


class SubagentManager:
    """Manages expert subagent execution with persistent profiles and memory."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        web_search_config: "WebSearchConfig | None" = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.expert_library = ExpertLibrary(workspace)
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        expert_name: str | None = None,
        context: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
    ) -> str:
        """Spawn an expert subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:40] + ("..." if len(task) > 40 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, expert_name, context)
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        expert_info = f" (expert: {expert_name})" if expert_name else " (generic expert)"
        logger.info("Spawned subagent [{}]{}: {}", task_id, expert_info, display_label)
        return f"Expert{expert_info} started for: {display_label} (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        expert_name: str | None = None,
        context: str | None = None,
    ) -> None:
        """Execute the expert subagent task with worklog updates."""
        logger.info("Expert [{}] starting task: {}", task_id, label)
        tools_used: list[str] = []
        effective_expert_name = expert_name

        try:
            tools = self._build_expert_tools()
            system_prompt = self._build_expert_prompt(expert_name, context)

            user_message = task
            if context:
                user_message = f"Context from conversation:\n{context}\n\nTask:\n{task}"

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            max_iterations = 25
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                )

                if response.has_tool_calls:
                    tool_call_dicts = [
                        tc.to_openai_tool_call() for tc in response.tool_calls
                    ]
                    messages.append(build_assistant_message(
                        response.content or "",
                        tool_calls=tool_call_dicts,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))

                    for tool_call in response.tool_calls:
                        tools_used.append(tool_call.name)
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug("Expert [{}] executing: {}({})", task_id, tool_call.name, args_str[:200])
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            # Save detailed result to file and get short summary
            result_path, worklog_path = self._save_task_artifacts(
                expert_name=effective_expert_name,
                task_id=task_id,
                task=task,
                label=label,
                final_result=final_result,
                tools_used=tools_used,
                status="success",
            )

            # Post-completion: create/update expert profile
            await self._post_completion(
                expert_name=effective_expert_name,
                task=task,
                final_result=final_result,
                tools_used=list(set(tools_used)),
                status="success",
            )

            short_summary = self._extract_short_summary(final_result)
            logger.info("Expert [{}] completed successfully", task_id)
            await self._announce_result(
                task_id, label, task, short_summary, result_path, worklog_path, origin, "ok"
            )

        except Exception as e:
            error_msg = f"Error: {e}"
            logger.error("Expert [{}] failed: {}", task_id, e)

            result_path, worklog_path = self._save_task_artifacts(
                expert_name=effective_expert_name,
                task_id=task_id,
                task=task,
                label=label,
                final_result=error_msg,
                tools_used=tools_used,
                status="error",
            )

            await self._announce_result(
                task_id, label, task, error_msg, result_path, worklog_path, origin, "error"
            )

    def _build_expert_tools(self) -> ToolRegistry:
        """Build the full tool set for an expert subagent."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        return tools

    def _build_expert_prompt(self, expert_name: str | None, context: str | None) -> str:
        """Build the expert subagent system prompt with profile, memory, skills, and worklog instructions."""
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)

        parts = [f"""# Expert Subagent

{time_ctx}

You are an expert subagent spawned by the orchestrator to complete a specific task.
You have full access to tools and skills. Stay focused on the assigned task.

## Workspace
{self.workspace}
"""]

        # Load expert profile + memory if this is a known expert
        if expert_name and self.expert_library.expert_exists(expert_name):
            profile = self.expert_library.load_expert_profile(expert_name)
            if profile:
                parts.append(f"## Your Expert Profile\n\n{profile}")

            memory = self.expert_library.load_expert_memory(expert_name)
            if memory:
                parts.append(f"## Your Memory (from previous runs)\n\n{memory}")

        # Load full skills catalog
        skills_loader = SkillsLoader(self.workspace)

        always_skills = skills_loader.get_always_skills()
        if always_skills:
            always_content = skills_loader.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"## Active Skills\n\n{always_content}")

        skills_summary = skills_loader.build_skills_summary()
        if skills_summary:
            parts.append(f"""## Skills

Read SKILL.md with read_file to use a skill.

{skills_summary}""")

        # Worklog instructions
        worklog_path = self.expert_library.get_worklog_path(expert_name or "_generic")
        parts.append(f"""## Work Process Rules (MUST FOLLOW)

You MUST maintain a live work log throughout your task execution.
Worklog file: {worklog_path}

**Before starting work:**
1. Write your plan to the worklog using write_file. Format:

# Task: [brief task description]
Started: [current time]
Status: in_progress

## Plan
1. [ ] First step
2. [ ] Second step
3. [ ] ...

## Progress

**During work — after each significant step:**
2. Update the worklog using edit_file. Mark completed steps with ✅, current step with 🔄:

### [time] Step description ✅
Key findings or actions taken

### [time] Current step 🔄
What you're doing now...

**When finished:**
3. Update worklog status to "completed"
4. Your final response must be a SHORT summary (2-3 sentences max) of what was accomplished.
   Do NOT include the full detailed results — those are saved separately.
   Just tell the orchestrator: what you did, whether it succeeded, and any key output paths.
""")

        return "\n\n".join(parts)

    def _save_task_artifacts(
        self,
        expert_name: str | None,
        task_id: str,
        task: str,
        label: str,
        final_result: str,
        tools_used: list[str],
        status: str,
    ) -> tuple[Path, Path]:
        """Save the detailed result file and finalize worklog. Returns (result_path, worklog_path)."""
        name = expert_name or f"_task-{task_id}"
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M")

        result_content = f"""# Result: {label}
Completed: {now_str}
Status: {status}
Task ID: {task_id}

## Task
{task}

## Detailed Result
{final_result}

## Tools Used
{', '.join(set(tools_used)) if tools_used else 'none'}
"""
        result_path = self.expert_library.save_result(name, result_content)

        # Append to worklog
        worklog_path = self.expert_library.get_worklog_path(name)
        try:
            existing = worklog_path.read_text(encoding="utf-8") if worklog_path.exists() else ""
            footer = f"\n\n---\nCompleted: {now_str} | Status: {status} | Result: {result_path}\n"
            self.expert_library.write_worklog(name, existing + footer)
        except OSError:
            pass

        return result_path, worklog_path

    def _extract_short_summary(self, full_result: str) -> str:
        """Extract a short summary from the expert's final response (already instructed to be short)."""
        lines = full_result.strip().split("\n")
        summary_lines = []
        char_count = 0
        for line in lines:
            if char_count + len(line) > 300:
                break
            summary_lines.append(line)
            char_count += len(line)
        return "\n".join(summary_lines) if summary_lines else full_result[:300]

    async def _post_completion(
        self,
        expert_name: str | None,
        task: str,
        final_result: str,
        tools_used: list[str],
        status: str,
    ) -> None:
        """After task completion: create new expert or update existing one's memory."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        if expert_name and self.expert_library.expert_exists(expert_name):
            # Existing expert: update usage stats and history
            self.expert_library.record_usage(expert_name)
            self.expert_library.append_expert_history(
                expert_name,
                f"[{now_str}] Task: {task[:100]} | Tools: {', '.join(tools_used)} | Status: {status}",
            )
            # Update expert memory via LLM
            await self._update_expert_memory(expert_name, task, final_result)
        else:
            # New task with no expert: generate a new expert profile via LLM
            await self._create_expert_from_task(task, final_result, tools_used)

    async def _create_expert_from_task(
        self,
        task: str,
        result: str,
        tools_used: list[str],
    ) -> None:
        """Use LLM to generate a new expert profile from the completed task."""
        prompt = f"""A task was just completed successfully. Create an expert profile for this type of work.

Task: {task}

Result summary: {result[:500]}

Tools used: {', '.join(tools_used)}

Call the save_expert_profile tool with:
- expert_name: short kebab-case name describing this expert's specialty (e.g. 'scrape-product-prices')
- description: one sentence about what this expert does
- tags: comma-separated keywords for matching similar future tasks
- approach: brief description of the approach that worked
- memory_notes: key facts learned that should be remembered for next time"""

        messages = [
            {"role": "system", "content": "You create expert profiles from completed tasks. Call the save_expert_profile tool."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=_SAVE_EXPERT_TOOL,
                model=self.model,
                tool_choice={"type": "function", "function": {"name": "save_expert_profile"}},
            )

            if not response.has_tool_calls:
                logger.warning("Expert profile creation: LLM did not call save_expert_profile")
                return

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, list):
                args = args[0] if args else {}

            name = args.get("expert_name", "").strip()
            if not name:
                logger.warning("Expert profile creation: empty expert_name")
                return

            tags = [t.strip() for t in args.get("tags", "").split(",") if t.strip()]

            self.expert_library.create_expert(
                name=name,
                description=args.get("description", ""),
                tags=tags,
                task=task,
                approach=args.get("approach", ""),
                tools_used=tools_used,
            )

            memory_notes = args.get("memory_notes", "")
            if memory_notes:
                self.expert_library.save_expert_memory(name, memory_notes)

            logger.info("Created new expert from task: {}", name)

        except Exception:
            logger.exception("Failed to create expert profile from task")

    async def _update_expert_memory(
        self,
        expert_name: str,
        task: str,
        result: str,
    ) -> None:
        """Use LLM to update an existing expert's memory with new learnings."""
        current_memory = self.expert_library.load_expert_memory(expert_name)

        prompt = f"""Update this expert's memory with any new facts learned from the latest task.

## Current Memory
{current_memory or "(empty)"}

## Latest Task
{task}

## Latest Result
{result[:500]}

Return the updated memory as markdown. Include all existing facts plus new ones.
If nothing new was learned, return the memory unchanged."""

        messages = [
            {"role": "system", "content": "You update an expert's persistent memory. Return only the updated memory content, nothing else."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.model,
            )
            if response.content and response.content.strip():
                self.expert_library.save_expert_memory(expert_name, response.content.strip())
                logger.debug("Updated memory for expert: {}", expert_name)
        except Exception:
            logger.exception("Failed to update expert memory for {}", expert_name)

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        short_summary: str,
        result_path: Path,
        worklog_path: Path,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the expert's result to the orchestrator via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Expert '{label}' {status_text}]

Summary: {short_summary}

Full result: {result_path}
Work log: {worklog_path}

Relay the summary to the user naturally. Keep it brief (1-2 sentences). Do not mention expert IDs or technical internals."""

        msg = InboundMessage(
            channel="system",
            sender_id="expert",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Expert [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running expert subagents."""
        return len(self._running_tasks)
