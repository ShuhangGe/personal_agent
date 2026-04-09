"""Subagent manager for expert-based task execution."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import time
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
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import build_assistant_message, ensure_dir


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

_TOOL_RESULT_MAX_CHARS = 16_000
_MAX_EVAL_ROUNDS = 5


class SubagentManager:
    """Manages expert subagent execution with isolated workspaces and persistent sessions."""

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

    # ── Core execution ────────────────────────────────────────────────────

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        expert_name: str | None = None,
        context: str | None = None,
    ) -> None:
        """Execute the expert subagent with evaluator review loop."""
        logger.info("Expert [{}] starting task: {}", task_id, label)
        tools_used: list[str] = []
        effective_expert_name = expert_name
        start_time = time.monotonic()

        # Resolve the expert's isolated workspace and session storage.
        temp_name = f"_task-{task_id}"
        expert_dir_name = expert_name if (expert_name and self.expert_library.expert_exists(expert_name)) else temp_name

        # Migrate old flat layout if needed
        if expert_name and self.expert_library.expert_exists(expert_name):
            self.expert_library._migrate_flat_to_nested(expert_name)

        expert_workspace = self.expert_library.get_expert_workspace(expert_dir_name)
        evaluator_workspace = self.expert_library.get_evaluator_workspace(expert_dir_name)

        try:
            tools = self._build_expert_tools(expert_workspace)
            system_prompt = self._build_expert_prompt(expert_name, context, expert_workspace, expert_dir_name)

            # Load persistent session history for known experts
            expert_session_mgr = SessionManager(
                self.expert_library.get_expert_dir(expert_dir_name) / "expert"
            )
            session = expert_session_mgr.get_or_create("expert")
            history = session.get_history(max_messages=0)

            user_message = task
            if context:
                user_message = f"Context from conversation:\n{context}\n\nTask:\n{task}"

            final_result: str | None = None
            verdict = "NOT GOOD"
            evaluator_feedback = ""
            eval_round = 0

            # Build the initial message list ONCE — rounds accumulate context
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_message},
            ]

            for eval_round_num in range(1, _MAX_EVAL_ROUNDS + 1):
                eval_round = eval_round_num

                if eval_round_num > 1:
                    # Continue the conversation — expert sees all previous tool calls
                    messages.append(build_assistant_message(
                        final_result or "",
                        reasoning_content="",
                    ))
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The evaluator reviewed your previous output and found issues.\n\n"
                            f"## Evaluator Feedback\n{evaluator_feedback}\n\n"
                            f"Please address the evaluator's feedback and produce an improved output.\n"
                            f"You can see all your previous work in the conversation history — build on it, don't start over."
                        ),
                    })

                # Run expert LLM loop
                max_iterations = 25
                iteration = 0

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

                # Run evaluator
                verdict, evaluator_feedback = await self._run_evaluator(
                    expert_dir_name=expert_dir_name,
                    task=task,
                    expert_output=final_result,
                    round_num=eval_round_num,
                    expert_workspace=expert_workspace,
                    evaluator_workspace=evaluator_workspace,
                )

                logger.info(
                    "Expert [{}] eval round {}/{}: verdict={}",
                    task_id, eval_round_num, _MAX_EVAL_ROUNDS, verdict,
                )

                if verdict == "GOOD":
                    break

            elapsed = time.monotonic() - start_time

            # Save session
            self._save_session_turn(session, messages, 1 + len(history))
            expert_session_mgr.save(session)

            # Derive status from eval verdict
            task_status = "success" if verdict == "GOOD" else "partial"

            # Save result with evaluation metrics
            result_path, worklog_path = self._save_task_artifacts(
                expert_name=expert_dir_name,
                task_id=task_id,
                task=task,
                label=label,
                final_result=final_result,
                tools_used=tools_used,
                status=task_status,
                eval_verdict=verdict,
                eval_rounds=eval_round,
                elapsed_seconds=elapsed,
            )

            # Post-completion: create/update expert profile and memory
            created_name = await self._post_completion(
                expert_name=effective_expert_name,
                temp_name=temp_name,
                task=task,
                final_result=final_result,
                tools_used=list(set(tools_used)),
                status=task_status,
                eval_verdict=verdict,
                eval_rounds=eval_round,
                evaluator_feedback=evaluator_feedback,
            )

            # If a new expert was created from a temp task, update paths
            if created_name and created_name != expert_dir_name:
                new_results = self.expert_library.list_results(created_name)
                if new_results:
                    result_path = new_results[0]
                worklog_path = self.expert_library.get_worklog_path(created_name)

            logger.info(
                "Expert [{}] done: verdict={}, rounds={}, {:.0f}s",
                task_id, verdict, eval_round, elapsed,
            )

            await self._announce_result(
                task_id, label, task, result_path, worklog_path, origin,
                verdict=verdict, eval_rounds=eval_round, elapsed_seconds=elapsed,
                evaluator_feedback=evaluator_feedback,
            )

        except Exception as e:
            error_msg = f"Error: {e}"
            logger.error("Expert [{}] failed: {}", task_id, e)

            result_path, worklog_path = self._save_task_artifacts(
                expert_name=expert_dir_name,
                task_id=task_id,
                task=task,
                label=label,
                final_result=error_msg,
                tools_used=tools_used,
                status="error",
            )

            await self._announce_result(
                task_id, label, task, result_path, worklog_path, origin,
                verdict="ERROR", eval_rounds=0, elapsed_seconds=time.monotonic() - start_time,
            )

    # ── Tool and prompt building ──────────────────────────────────────────

    def _build_expert_tools(self, expert_workspace: Path) -> ToolRegistry:
        """Build the full tool set, sandboxed to the expert's own workspace."""
        tools = ToolRegistry()
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            tools.register(cls(workspace=expert_workspace, allowed_dir=expert_workspace))
        tools.register(ExecTool(
            working_dir=str(expert_workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=True,
            path_append=self.exec_config.path_append,
        ))
        tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        return tools

    def _build_expert_prompt(
        self,
        expert_name: str | None,
        context: str | None,
        expert_workspace: Path,
        expert_dir_name: str | None = None,
    ) -> str:
        """Build the expert subagent system prompt with profile, memory, skills, and worklog instructions."""
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)

        parts = [f"""# Expert Subagent

{time_ctx}

You are an expert subagent spawned by the orchestrator to complete a specific task.
You have full access to tools and skills. Stay focused on the assigned task.

## Your Workspace
{expert_workspace}

This is YOUR isolated workspace. All files you create or edit live here.
Use relative paths when possible — they resolve against your workspace.
"""]

        # Load expert profile + memory if this is a known expert
        if expert_name and self.expert_library.expert_exists(expert_name):
            profile = self.expert_library.load_expert_profile(expert_name)
            if profile:
                parts.append(f"## Your Expert Profile\n\n{profile}")

            # Load soul (identity/personality)
            soul = self.expert_library.load_expert_soul(expert_name)
            if soul:
                parts.append(f"## Your Identity (Soul)\n\n{soul}")

            # Load memory (what works/doesn't work)
            memory = self.expert_library.load_expert_memory(expert_name)
            if memory:
                parts.append(f"## Your Memory (What Works)\n\n{memory}")

            # Load experience (lessons learned)
            experience = self.expert_library.load_expert_experience(expert_name)
            if experience:
                parts.append(f"## Your Experience (Lessons Learned)\n\n{experience}")

            # Load evaluator guardrails (read-only constraints from the evaluator)
            guardrails = self.expert_library.load_evaluator_guardrails(expert_name)
            if guardrails:
                parts.append(f"""## Guardrails (from Evaluator — YOU MUST FOLLOW)

The following guardrails are maintained by your evaluator based on past reviews.
You MUST follow these rules. You CANNOT modify this file.
Violating these guardrails will result in a NOT GOOD verdict.

{guardrails}""")

        # Load full skills catalog from the main workspace
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
        dir_name = expert_dir_name or expert_name or "_generic"
        worklog_path = self.expert_library.get_worklog_path(dir_name)
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

    # ── Evaluator methods ────────────────────────────────────────────────

    async def _run_evaluator(
        self,
        expert_dir_name: str,
        task: str,
        expert_output: str,
        round_num: int,
        expert_workspace: Path,
        evaluator_workspace: Path,
    ) -> tuple[str, str]:
        """Run the evaluator to review expert output.

        Returns (verdict, feedback) where verdict is "GOOD" or "NOT GOOD".
        """
        eval_tools = self._build_evaluator_tools(expert_workspace, evaluator_workspace)
        eval_prompt = self._build_evaluator_prompt(
            expert_dir_name, expert_workspace, evaluator_workspace,
        )

        # Load evaluator session
        eval_session_mgr = SessionManager(
            self.expert_library.get_expert_dir(expert_dir_name) / "evaluator"
        )
        eval_session = eval_session_mgr.get_or_create("evaluator")
        eval_history = eval_session.get_history(max_messages=0)

        user_msg = (
            f"## Task (Round {round_num})\n{task}\n\n"
            f"## Expert Output to Review\n{expert_output}\n\n"
            f"Review the expert's output above. Check if it fully addresses the task.\n"
            f"Provide your verdict with specific feedback."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": eval_prompt},
            *eval_history,
            {"role": "user", "content": user_msg},
        ]

        max_iterations = 10
        iteration = 0
        eval_result: str | None = None

        while iteration < max_iterations:
            iteration += 1

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=eval_tools.get_definitions(),
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
                    result = await eval_tools.execute(tool_call.name, tool_call.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
            else:
                eval_result = response.content
                break

        if eval_result is None:
            eval_result = "No evaluation response generated."

        # Save evaluator session turn
        self._save_session_turn(eval_session, messages, 1 + len(eval_history))
        eval_session_mgr.save(eval_session)

        verdict, feedback = self._parse_evaluator_verdict(eval_result)
        return verdict, feedback

    def _build_evaluator_tools(
        self,
        expert_workspace: Path,
        evaluator_workspace: Path,
    ) -> ToolRegistry:
        """Build tools for the evaluator — can read expert workspace, write to own workspace."""
        tools = ToolRegistry()
        # ReadFile and ListDir can read from both evaluator workspace and expert workspace (read-only)
        for cls in (ReadFileTool, ListDirTool):
            tools.register(cls(
                workspace=evaluator_workspace,
                allowed_dir=evaluator_workspace,
                read_only_dirs=[expert_workspace],
            ))
        # Write and Edit restricted to evaluator workspace only
        for cls in (WriteFileTool, EditFileTool):
            tools.register(cls(workspace=evaluator_workspace, allowed_dir=evaluator_workspace))
        tools.register(ExecTool(
            working_dir=str(evaluator_workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=True,
            path_append=self.exec_config.path_append,
        ))
        tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        return tools

    def _build_evaluator_prompt(
        self,
        expert_dir_name: str,
        expert_workspace: Path,
        evaluator_workspace: Path,
    ) -> str:
        """Build the evaluator's system prompt."""
        from nanobot.agent.context import ContextBuilder

        time_ctx = ContextBuilder._build_runtime_context(None, None)

        parts = [f"""# Evaluator Agent

{time_ctx}

You are an evaluator agent. Your job is to critically review the output produced by the expert agent.
You must assess whether the output correctly and completely fulfills the assigned task.

## Expert Workspace (READ-ONLY)
{expert_workspace}

You can read files from the expert's workspace to verify their work, but you CANNOT modify anything there.

## Your Workspace (READ-WRITE)
{evaluator_workspace}

This is your own workspace for writing notes or analysis files.
"""]

        # Load evaluator soul, memory, experience if expert exists
        expert_name = None
        # Extract the actual expert name (strip _task- prefix for temp dirs)
        if not expert_dir_name.startswith("_task-"):
            expert_name = expert_dir_name

        if expert_name and self.expert_library.expert_exists(expert_name):
            soul = self.expert_library.load_evaluator_soul(expert_name)
            if soul:
                parts.append(f"## Your Identity (Soul)\n\n{soul}")

            memory = self.expert_library.load_evaluator_memory(expert_name)
            if memory:
                parts.append(f"## Your Memory\n\n{memory}")

            experience = self.expert_library.load_evaluator_experience(expert_name)
            if experience:
                parts.append(f"## Your Experience\n\n{experience}")

            guardrails = self.expert_library.load_evaluator_guardrails(expert_name)
            if guardrails:
                parts.append(f"## Current Guardrails (you maintain this)\n\n{guardrails}")

        parts.append("""## Review Instructions

1. Read the task description carefully
2. Review the expert's output thoroughly
3. Use read_file and list_dir to verify files in the expert's workspace
4. IMPORTANT: You can ONLY access the expert's workspace and your own workspace.
   Do NOT attempt to access any files outside these directories — they are blocked by sandbox.
   Evaluate based solely on the files the expert created, not the original source.
5. Check if the expert violated any existing guardrails (failed approaches,
   anti-patterns, quality standards). If so, this is an automatic NOT GOOD.
6. Assess correctness, completeness, edge cases, and overall quality
7. Provide your review with specific feedback — especially note:
   - Any approach that failed and should be added to guardrails
   - Any quality standard that should be enforced going forward
8. End with your verdict using the required format:

---VERDICT---
Status: GOOD
---END VERDICT---

OR:

---VERDICT---
Status: NOT GOOD
Issues: [comma-separated list of specific issues]
---END VERDICT---
""")

        return "\n\n".join(parts)

    @staticmethod
    def _parse_evaluator_verdict(response: str) -> tuple[str, str]:
        """Parse verdict and feedback from evaluator response.

        Returns (verdict, feedback) where verdict is "GOOD" or "NOT GOOD".
        """
        # Extract verdict block
        verdict_match = re.search(
            r"---VERDICT---\s*Status:\s*(GOOD|NOT GOOD)\s*(?:Issues:\s*(.+?))?\s*---END VERDICT---",
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if verdict_match:
            verdict = verdict_match.group(1).upper()
            issues = verdict_match.group(2)
        else:
            # Fallback: look for any GOOD/NOT GOOD in the response
            if re.search(r"\bGOOD\b", response, re.IGNORECASE):
                verdict = "GOOD"
            else:
                verdict = "NOT GOOD"
            issues = None

        # Feedback is the full response minus the verdict block
        feedback = response
        if verdict_match:
            feedback = response[:verdict_match.start()] + response[verdict_match.end():]
        feedback = feedback.strip()

        # If verdict is NOT GOOD and issues were provided, include them
        if verdict == "NOT GOOD" and issues:
            feedback = f"Issues: {issues.strip()}\n\n{feedback}"

        return verdict, feedback

    def _save_session_turn(
        self,
        session: Session,
        messages: list[dict],
        skip: int,
    ) -> None:
        """Save new-turn messages into the expert's persistent session."""
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
            if role == "tool" and isinstance(content, str) and len(content) > _TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:_TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    # ── Artifact saving ──────────────────────────────────────────────────

    def _save_task_artifacts(
        self,
        expert_name: str,
        task_id: str,
        task: str,
        label: str,
        final_result: str,
        tools_used: list[str],
        status: str,
        eval_verdict: str = "",
        eval_rounds: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> tuple[Path, Path]:
        """Save the detailed result file and finalize worklog. Returns (result_path, worklog_path)."""
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M")

        eval_metrics = ""
        if eval_verdict:
            eval_metrics = f"""
## Evaluation
Verdict: {eval_verdict}
Rounds: {eval_rounds}
Time: {elapsed_seconds:.1f}s
"""

        result_content = f"""# Result: {label}
Completed: {now_str}
Status: {status}
Task ID: {task_id}
{eval_metrics}
## Task
{task}

## Detailed Result
{final_result}

## Tools Used
{', '.join(set(tools_used)) if tools_used else 'none'}
"""
        result_path = self.expert_library.save_result(expert_name, result_content)

        worklog_path = self.expert_library.get_worklog_path(expert_name)
        try:
            existing = worklog_path.read_text(encoding="utf-8") if worklog_path.exists() else ""
            footer = f"\n\n---\nCompleted: {now_str} | Status: {status} | Result: {result_path}\n"
            self.expert_library.write_worklog(expert_name, existing + footer)
        except OSError:
            pass

        return result_path, worklog_path

    def _extract_short_summary(self, full_result: str) -> str:
        """Extract a short summary from the expert's final response."""
        lines = full_result.strip().split("\n")
        summary_lines = []
        char_count = 0
        for line in lines:
            if char_count + len(line) > 300:
                break
            summary_lines.append(line)
            char_count += len(line)
        return "\n".join(summary_lines) if summary_lines else full_result[:300]

    # ── Post-completion ──────────────────────────────────────────────────

    async def _post_completion(
        self,
        expert_name: str | None,
        temp_name: str,
        task: str,
        final_result: str,
        tools_used: list[str],
        status: str,
        eval_verdict: str = "",
        eval_rounds: int = 0,
        evaluator_feedback: str = "",
    ) -> str | None:
        """After task completion: create new expert or update existing one's memory.

        Also updates evaluator memory and guardrails.
        Returns the created expert name if a new expert was born, else None.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        if expert_name and self.expert_library.expert_exists(expert_name):
            self.expert_library.record_usage(expert_name)
            self.expert_library.append_expert_history(
                expert_name,
                f"[{now_str}] Task: {task[:100]} | Tools: {', '.join(tools_used)} | Status: {status} | Eval: {eval_verdict} ({eval_rounds} rounds)",
            )
            await self._update_expert_memory(expert_name, task, final_result)
            await self._update_expert_experience(expert_name, task, final_result, status)
            await self._update_evaluator_memory(expert_name, task, final_result, eval_verdict, eval_rounds)
            await self._update_evaluator_guardrails(
                expert_name, task, final_result, eval_verdict, eval_rounds, evaluator_feedback,
            )
            return None
        else:
            created_name = await self._create_expert_from_task(task, final_result, tools_used, temp_name)
            if created_name:
                await self._update_evaluator_memory(created_name, task, final_result, eval_verdict, eval_rounds)
                await self._update_evaluator_guardrails(
                    created_name, task, final_result, eval_verdict, eval_rounds, evaluator_feedback,
                )
            return created_name

    async def _create_expert_from_task(
        self,
        task: str,
        result: str,
        tools_used: list[str],
        temp_name: str,
    ) -> str | None:
        """Use LLM to generate a new expert profile, then migrate temp data to it."""
        prompt = f"""A task was just completed. Create an expert profile for this type of work.

Task: {task}

Result summary: {result[:500]}

Tools used: {', '.join(tools_used)}

Call the save_expert_profile tool with:
- expert_name: 2-3 word English name in kebab-case describing the expert's general purpose.
  Keep it SHORT (max 30 chars). Examples: 'novel-analyzer', 'web-scraper', 'code-reviewer'.
  NEVER use Chinese or other non-ASCII characters.
- description: one sentence about what this expert does
- tags: comma-separated keywords for matching similar future tasks
- approach: brief description of the approach that worked
- memory_notes: key facts learned that should be remembered for next time

If you cannot call the tool, respond ONLY with a JSON object containing these fields."""

        messages = [
            {"role": "system", "content": "You create expert profiles from completed tasks. Call the save_expert_profile tool. The expert_name MUST be short English kebab-case only. If you cannot call tools, respond with a JSON object instead."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=_SAVE_EXPERT_TOOL,
                model=self.model,
                tool_choice={"type": "function", "function": {"name": "save_expert_profile"}},
            )

            args: dict[str, Any] | None = None

            if response.has_tool_calls:
                args = response.tool_calls[0].arguments
                if isinstance(args, str):
                    args = json.loads(args)
                if isinstance(args, list):
                    args = args[0] if args else {}
            elif response.content:
                # Some models don't support tool_choice — try parsing text as JSON
                args = self._extract_json_from_text(response.content)
                if args:
                    logger.info("Expert profile creation: extracted profile from text response (model may not support tool_choice)")

            if not args or not isinstance(args, dict):
                logger.warning("Expert profile creation: no valid profile data, using fallback")
                return self._create_fallback_expert(task, tools_used, temp_name)

            name = args.get("expert_name", "").strip()
            if not name:
                logger.warning("Expert profile creation: empty expert_name, using fallback")
                return self._create_fallback_expert(task, tools_used, temp_name)

            name = self._sanitize_expert_name(name)
            if not name:
                return self._create_fallback_expert(task, tools_used, temp_name)

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

            # Migrate temp workspace, sessions, results to the new expert directory
            self._migrate_temp_to_expert(temp_name, name)

            logger.info("Created new expert from task: {}", name)
            return name

        except Exception:
            logger.exception("Failed to create expert profile from task, using fallback")
            return self._create_fallback_expert(task, tools_used, temp_name)

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any] | None:
        """Try to extract a JSON object from LLM text when tool calling fails."""
        # Try the whole text first
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and "expert_name" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        # Try to find a JSON block in the text
        match = re.search(r"\{[^{}]*\"expert_name\"[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    @staticmethod
    def _sanitize_expert_name(name: str) -> str | None:
        """Sanitize an expert name to short kebab-case ASCII. Returns None if unusable."""
        name = re.sub(r"[^a-z0-9\-]", "-", name.lower())
        name = re.sub(r"-{2,}", "-", name).strip("-")
        if not re.search(r"[a-z]", name) or len(name) < 2:
            return None
        return name[:30]

    def _create_fallback_expert(
        self,
        task: str,
        tools_used: list[str],
        temp_name: str,
    ) -> str | None:
        """Create a basic expert profile when LLM profile generation fails.

        Derives a meaningful name from the task's dominant tools and keywords
        instead of a random hash.
        """
        name = self._derive_name_from_task(task, tools_used)

        self.expert_library.create_expert(
            name=name,
            description=f"Auto-created expert for: {task[:100]}",
            tags=[],
            task=task,
            approach="Automatic profile — approach not recorded.",
            tools_used=tools_used,
        )

        self._migrate_temp_to_expert(temp_name, name)
        logger.info("Created fallback expert profile: {}", name)
        return name

    @staticmethod
    def _derive_name_from_task(task: str, tools_used: list[str]) -> str:
        """Derive a short kebab-case name from the task description and tools.

        Uses keyword extraction as a best-effort fallback when LLM naming fails.
        """
        tool_hints = {
            "web_search": "researcher",
            "web_fetch": "web-reader",
            "exec": "script-runner",
            "read_file": "file-reader",
            "write_file": "file-writer",
        }
        # Pick the most descriptive tool
        for tool, hint in tool_hints.items():
            if tool in tools_used and len(tools_used) <= 2:
                return hint

        # Extract English words from the task
        words = re.findall(r"[a-zA-Z]{3,}", task)
        if words:
            slug = "-".join(w.lower() for w in words[:3])
            slug = slug[:25]
            return slug

        # Last resort
        short_id = uuid.uuid4().hex[:6]
        return f"task-{short_id}"

    def _migrate_temp_to_expert(self, temp_name: str, expert_name: str) -> None:
        """Move workspace, sessions, and results from a temp directory to the new expert."""
        temp_dir = self.expert_library.get_expert_dir(temp_name)
        expert_dir = self.expert_library.get_expert_dir(expert_name)

        if not temp_dir.exists():
            return

        # Migrate expert and evaluator subdirs (temp was created with nested layout)
        for subdir in (
            "expert/workspace", "expert/sessions", "expert/memory",
            "evaluator/workspace", "evaluator/sessions", "evaluator/memory",
        ):
            src = temp_dir / subdir
            dst = expert_dir / subdir
            if not src.exists():
                continue
            if dst.exists():
                for item in src.iterdir():
                    target = dst / item.name
                    if not target.exists():
                        if item.is_dir():
                            shutil.copytree(str(item), str(target))
                        else:
                            shutil.copy2(str(item), str(target))
            else:
                shutil.copytree(str(src), str(dst))

        # Migrate results (top-level)
        for subdir in ("results",):
            src = temp_dir / subdir
            dst = expert_dir / subdir
            if not src.exists():
                continue
            if dst.exists():
                for item in src.iterdir():
                    target = dst / item.name
                    if not target.exists():
                        shutil.copy2(str(item), str(target))
            else:
                shutil.copytree(str(src), str(dst))

        # Copy worklog if it exists
        temp_worklog = temp_dir / "WORKLOG.md"
        if temp_worklog.exists():
            expert_worklog = expert_dir / "WORKLOG.md"
            if not expert_worklog.exists():
                shutil.copy2(str(temp_worklog), str(expert_worklog))

        # Clean up temp directory
        try:
            shutil.rmtree(str(temp_dir))
        except OSError:
            logger.warning("Could not clean up temp expert dir: {}", temp_dir)

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

    async def _update_expert_experience(
        self,
        expert_name: str,
        task: str,
        result: str,
        status: str,
    ) -> None:
        """Use LLM to extract and save lessons learned from this task execution."""
        now_str = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""Extract lessons learned from this task execution.

## Task
{task}

## Result
{result[:800]}

## Status
{status}

Provide a concise lesson learned (2-3 sentences max) that would help in future similar tasks.
Focus on:
- What worked well
- What could be improved
- Key insights or patterns discovered

Format your response as a brief paragraph that can be appended to an experience log."""

        messages = [
            {"role": "system", "content": "You extract concise lessons learned from task executions. Be brief and actionable."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.model,
            )
            if response.content and response.content.strip():
                experience_entry = f"""## {now_str}

### Task: {task[:80]}...

**What Happened:** {response.content.strip()}

---
"""
                self.expert_library.append_expert_experience(expert_name, experience_entry)
                logger.debug("Updated experience for expert: {}", expert_name)
        except Exception:
            logger.exception("Failed to update expert experience for {}", expert_name)

    async def _update_evaluator_memory(
        self,
        expert_name: str,
        task: str,
        result: str,
        verdict: str,
        eval_rounds: int,
    ) -> None:
        """Update evaluator memory with evaluation learnings."""
        now_str = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""Update the evaluator's memory with insights from this evaluation.

## Current Evaluator Memory
{self.expert_library.load_evaluator_memory(expert_name) or "(empty)"}

## Task Evaluated
{task}

## Expert Output
{result[:500]}

## Evaluation Result
Verdict: {verdict}
Rounds: {eval_rounds}

What patterns or insights should the evaluator remember for future reviews?
Return the updated memory as markdown."""

        messages = [
            {"role": "system", "content": "You update an evaluator's persistent memory. Return only the updated memory content."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.model,
            )
            if response.content and response.content.strip():
                self.expert_library.save_evaluator_memory(expert_name, response.content.strip())
                # Also append experience
                experience_entry = f"""## {now_str}

### Task: {task[:80]}...
**Verdict:** {verdict} after {eval_rounds} round(s)

---
"""
                self.expert_library.append_evaluator_experience(expert_name, experience_entry)
                logger.debug("Updated evaluator memory for expert: {}", expert_name)
        except Exception:
            logger.exception("Failed to update evaluator memory for {}", expert_name)

    async def _update_evaluator_guardrails(
        self,
        expert_name: str,
        task: str,
        result: str,
        verdict: str,
        eval_rounds: int,
        evaluator_feedback: str,
    ) -> None:
        """Update GUARDRAILS.md with lessons from this evaluation.

        Called for both GOOD and NOT GOOD verdicts so the expert accumulates
        knowledge about what to avoid and what quality standards to maintain.
        """
        current = self.expert_library.load_evaluator_guardrails(expert_name)
        now_str = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""You maintain a GUARDRAILS file that tells the expert what NOT to do and
what quality standards to meet. Update it based on this evaluation.

## Current Guardrails
{current or "(empty — first evaluation)"}

## Task
{task}

## Expert Output (excerpt)
{result[:600]}

## Evaluator Feedback
{evaluator_feedback[:800] if evaluator_feedback else "(no detailed feedback)"}

## Verdict
{verdict} after {eval_rounds} round(s) — Date: {now_str}

Rules for updating:
- Keep ALL existing guardrail entries (never remove learned lessons).
- If verdict is NOT GOOD: add the specific failed approach and why it failed
  under "Failed Approaches". Add any new anti-patterns discovered.
- If verdict is GOOD: add any quality standards or good practices that should
  be maintained. If the expert almost failed, note what to watch out for.
- Be concise and specific — each entry should be actionable.
- Use the existing section structure (Failed Approaches, Anti-Patterns,
  Quality Standards). Add new sections only if truly needed.

Return the complete updated GUARDRAILS.md content."""

        messages = [
            {"role": "system", "content": "You maintain an expert's guardrails file. Return only the updated guardrails content as markdown. Preserve all existing entries."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.model,
            )
            if response.content and response.content.strip():
                self.expert_library.save_evaluator_guardrails(
                    expert_name, response.content.strip(),
                )
                logger.debug("Updated guardrails for expert: {}", expert_name)
        except Exception:
            logger.exception("Failed to update guardrails for {}", expert_name)

    # ── Announcement ─────────────────────────────────────────────────────

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result_path: Path,
        worklog_path: Path,
        origin: dict[str, str],
        verdict: str = "GOOD",
        eval_rounds: int = 1,
        elapsed_seconds: float = 0.0,
        evaluator_feedback: str = "",
    ) -> None:
        """Announce the expert's result to the orchestrator via the message bus."""
        if verdict == "ERROR":
            status_text = "failed"
            announce_content = f"""[Expert '{label}' {status_text}]

Full result: {result_path}

Relay the error to the user naturally. Keep it brief."""
        elif verdict == "GOOD":
            announce_content = f"""Expert '{label}' done. GOOD, {eval_rounds} round{"s" if eval_rounds != 1 else ""}, {elapsed_seconds:.0f}s. Result saved: {result_path}

Relay this to the user naturally. Keep it brief (1-2 sentences). Do not mention expert IDs or technical internals."""
        else:
            # NOT GOOD after max rounds
            issues_summary = ""
            if evaluator_feedback:
                # Take first 200 chars of feedback as issues summary
                issues_summary = evaluator_feedback[:200]
            announce_content = f"""Expert '{label}' done. NOT GOOD after {eval_rounds} rounds, {elapsed_seconds:.0f}s. Issues: {issues_summary}

Full result: {result_path}
Relay this to the user naturally. Include the key issues found. Keep it brief."""
        msg = InboundMessage(
            channel="system",
            sender_id="expert",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Expert [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    # ── Cancellation ─────────────────────────────────────────────────────

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
