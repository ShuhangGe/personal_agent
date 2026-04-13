"""Subagent manager for expert-based task execution."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from loguru import logger

from nanobot.agent.agent_library import AgentLibrary
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


_TOOL_RESULT_MAX_CHARS = 16_000
_MAX_EVAL_ROUNDS = 5
_MEMORY_MAX_CHARS = 3000
_GUARDRAILS_MAX_CHARS = 2500


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
        self.agent_library = AgentLibrary(workspace)
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}
        self._interrupt_events: dict[str, asyncio.Event] = {}
        self._pending_interrupts: dict[str, str] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        agent_name: str | None = None,
        context: str | None = None,
        suggested_name: str | None = None,
        suggested_description: str | None = None,
        output_dir: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        on_progress: Any = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:40] + ("..." if len(task) > 40 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(
                task_id, task, display_label, origin, agent_name, context,
                suggested_name=suggested_name,
                suggested_description=suggested_description,
                output_dir=output_dir,
                on_progress=on_progress,
            )
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

        agent_info = f" (subagent: {agent_name})" if agent_name else " (new subagent)"
        name_info = f" as '{suggested_name}'" if suggested_name and not agent_name else ""
        logger.info("Spawned subagent [{}]{}{}: {}", task_id, agent_info, name_info, display_label)
        return f"Subagent{agent_info}{name_info} started for: {display_label} (id: {task_id}). I'll notify you when it completes."

    # ── Core execution ────────────────────────────────────────────────────

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        agent_name: str | None = None,
        context: str | None = None,
        suggested_name: str | None = None,
        suggested_description: str | None = None,
        output_dir: str | None = None,
        on_progress: Any = None,
    ) -> None:
        """Execute the subagent with evaluator review loop."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)
        tools_used: list[str] = []
        effective_agent_name = agent_name
        start_time = time.monotonic()

        # Set up interrupt mechanism
        interrupt_event = asyncio.Event()
        self._interrupt_events[task_id] = interrupt_event

        # Resolve the agent directory name at spawn time — no temp dirs.
        if agent_name and self.agent_library.agent_exists(agent_name):
            # Reusing an existing agent
            agent_dir_name = agent_name
            self.agent_library._migrate_flat_to_nested(agent_name)
        elif suggested_name:
            sanitized = self._sanitize_agent_name(suggested_name)
            agent_dir_name = sanitized if sanitized else self._derive_name_from_task(task, [])
        else:
            agent_dir_name = self._derive_name_from_task(task, [])

        expert_workspace = self.agent_library.get_expert_workspace(agent_dir_name)
        evaluator_workspace = self.agent_library.get_evaluator_workspace(agent_dir_name)

        # Resolve output_dir for shared batch output
        resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

        try:
            tools = self._build_expert_tools(expert_workspace, resolved_output_dir)
            system_prompt = self._build_expert_prompt(
                agent_name, context, expert_workspace, agent_dir_name,
                output_dir=resolved_output_dir,
            )

            # Load persistent session history for known experts
            expert_session_mgr = SessionManager(
                self.agent_library.get_agent_dir(agent_dir_name) / "expert"
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
                    agent_dir_name=agent_dir_name,
                    task=task,
                    expert_output=final_result,
                    round_num=eval_round_num,
                    expert_workspace=expert_workspace,
                    evaluator_workspace=evaluator_workspace,
                )

                logger.info(
                    "Subagent [{}] eval round {}/{}: verdict={}",
                    task_id, eval_round_num, _MAX_EVAL_ROUNDS, verdict,
                )

                # Per-round progress notification
                if on_progress:
                    summary = evaluator_feedback[:200] if evaluator_feedback else "working..."
                    try:
                        await on_progress(
                            f"Round {eval_round_num}/{_MAX_EVAL_ROUNDS}: {verdict}. {summary}"
                        )
                    except Exception:
                        logger.debug("Progress callback failed for [{}]", task_id)

                # Interrupt check (only when verdict is not GOOD)
                if verdict != "GOOD":
                    interrupt_evt = self._interrupt_events.get(task_id)
                    if interrupt_evt:
                        try:
                            await asyncio.wait_for(interrupt_evt.wait(), timeout=3.0)
                            user_feedback = self._pending_interrupts.pop(task_id, "")
                            if user_feedback:
                                messages.append({"role": "user", "content": f"User interrupt: {user_feedback}"})
                        except asyncio.TimeoutError:
                            pass  # No interrupt, continue normally

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
                agent_name=agent_dir_name,
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
            await self._post_completion(
                agent_dir_name=agent_dir_name,
                is_reuse=bool(effective_agent_name and self.agent_library.agent_exists(effective_agent_name)),
                task=task,
                final_result=final_result,
                tools_used=list(set(tools_used)),
                status=task_status,
                eval_verdict=verdict,
                eval_rounds=eval_round,
                evaluator_feedback=evaluator_feedback,
                suggested_description=suggested_description,
            )

            logger.info(
                "Subagent [{}] done: verdict={}, rounds={}, {:.0f}s",
                task_id, verdict, eval_round, elapsed,
            )

            await self._announce_result(
                task_id, label, task, result_path, worklog_path, origin,
                verdict=verdict, eval_rounds=eval_round, elapsed_seconds=elapsed,
                evaluator_feedback=evaluator_feedback,
                agent_dir_name=agent_dir_name,
            )

        except Exception as e:
            error_msg = f"Error: {e}"
            logger.error("Subagent [{}] failed: {}", task_id, e)

            # Ensure the agent profile exists even on error
            self._ensure_agent_profile(
                agent_dir_name, task, suggested_description,
            )

            result_path, worklog_path = self._save_task_artifacts(
                agent_name=agent_dir_name,
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
        finally:
            self._interrupt_events.pop(task_id, None)
            self._pending_interrupts.pop(task_id, None)

    # ── Tool and prompt building ──────────────────────────────────────────

    def _build_expert_tools(
        self, expert_workspace: Path, output_dir: Path | None = None,
    ) -> ToolRegistry:
        """Build the full tool set, sandboxed to the expert's own workspace.

        Read-only access is granted to the user's home directory so the expert
        can read files the user references (e.g. ~/Desktop/some_file.txt).
        Write/edit remain restricted to the expert workspace, plus output_dir
        if provided (for batch tasks that write to a shared directory).
        """
        home_dir = Path.home()
        extra_write = [output_dir] if output_dir else []
        tools = ToolRegistry()
        # ReadFile and ListDir: can read from home dir (read-only)
        for cls in (ReadFileTool, ListDirTool):
            tools.register(cls(
                workspace=expert_workspace,
                allowed_dir=expert_workspace,
                read_only_dirs=[home_dir],
            ))
        # Write and Edit: workspace + output_dir if provided
        for cls in (WriteFileTool, EditFileTool):
            tools.register(cls(
                workspace=expert_workspace,
                allowed_dir=expert_workspace,
                extra_write_dirs=extra_write,
            ))
        tools.register(ExecTool(
            working_dir=str(expert_workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=True,
            path_append=self.exec_config.path_append,
            extra_allowed_dirs=[str(d) for d in extra_write],
        ))
        tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        return tools

    def _build_expert_prompt(
        self,
        agent_name: str | None,
        context: str | None,
        expert_workspace: Path,
        agent_dir_name: str | None = None,
        output_dir: Path | None = None,
    ) -> str:
        """Build the expert subagent system prompt with profile, memory, skills, and worklog instructions."""
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)

        home_dir = Path.home()
        parts = [f"""# Expert Subagent

{time_ctx}

You are an expert subagent spawned by the orchestrator to complete a specific task.
You have full access to tools and skills. Stay focused on the assigned task.

## Your Workspace
{expert_workspace}

This is YOUR isolated workspace. All files you create or edit MUST live here.
Use relative paths when possible — they resolve against your workspace.

## File Access
- **read_file** and **list_dir**: Can read anywhere under `{home_dir}` (read-only).
  Use these to access files the user references, e.g. `~/Downloads/report.pdf`.
- **write_file** and **edit_file**: Can ONLY write inside your workspace above.
- **exec**: Can ONLY access paths inside your workspace. Cannot read or write outside it.
  If you need to read a file from outside, use `read_file` instead.
"""]

        if output_dir:
            parts.append(f"""## Shared Output Directory
{output_dir}

You are part of a batch task. Write your final output files to this shared directory.
**write_file**, **edit_file**, and **exec** can all access this directory.
Do NOT write intermediate files here — only final results.
""")

        # Load expert profile + memory if this is a known expert
        if agent_name and self.agent_library.agent_exists(agent_name):
            profile = self.agent_library.load_agent_profile(agent_name)
            if profile:
                parts.append(f"## Your Expert Profile\n\n{profile}")

            # Load soul (identity/personality)
            soul = self.agent_library.load_expert_soul(agent_name)
            if soul:
                parts.append(f"## Your Identity (Soul)\n\n{soul}")

            # Load memory (what works/doesn't work)
            memory = self.agent_library.load_expert_memory(agent_name)
            if memory:
                parts.append(f"## Your Memory (What Works)\n\n{memory}")

            # Load experience (lessons learned)
            experience = self.agent_library.load_expert_experience(agent_name)
            if experience:
                parts.append(f"## Your Experience (Lessons Learned)\n\n{experience}")

            # Load evaluator guardrails (read-only constraints from the evaluator)
            guardrails = self.agent_library.load_evaluator_guardrails(agent_name)
            if guardrails:
                parts.append(f"""## Guardrails (from Evaluator — YOU MUST FOLLOW)

The following guardrails are maintained by your evaluator based on past reviews.
You MUST follow these rules. You CANNOT modify this file.
Violating these guardrails will result in a NOT GOOD verdict.

{guardrails}""")

            # Load user preferences (learned by evaluator from past feedback)
            preferences = self.agent_library.load_evaluator_preferences(agent_name)
            if preferences:
                parts.append(f"## User Preferences (from Evaluator — FOLLOW THESE)\n\n{preferences}")

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
        dir_name = agent_dir_name or agent_name or "_generic"
        worklog_path = self.agent_library.get_worklog_path(dir_name)
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

        # Nudge: encourage mid-task knowledge persistence for known agents
        if agent_dir_name:
            memory_path = self.agent_library._memory_dir(agent_dir_name) / "MEMORY.md"
            parts.append(f"""**Knowledge Persistence:**
5. Your memory file is at: {memory_path}
   After completing any step where you learned something valuable (a working approach,
   a pitfall to avoid, a useful API or pattern), update your MEMORY.md using edit_file.
   Don't wait until the end — save knowledge as you discover it.""")

        return "\n\n".join(parts)

    # ── Evaluator methods ────────────────────────────────────────────────

    async def _run_evaluator(
        self,
        agent_dir_name: str,
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
            agent_dir_name, expert_workspace, evaluator_workspace,
        )

        # Load evaluator session
        eval_session_mgr = SessionManager(
            self.agent_library.get_agent_dir(agent_dir_name) / "evaluator"
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
        agent_dir_name: str,
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
        resolved_name = agent_dir_name if self.agent_library.agent_exists(agent_dir_name) else None

        if resolved_name and self.agent_library.agent_exists(resolved_name):
            soul = self.agent_library.load_evaluator_soul(resolved_name)
            if soul:
                parts.append(f"## Your Identity (Soul)\n\n{soul}")

            memory = self.agent_library.load_evaluator_memory(resolved_name)
            if memory:
                parts.append(f"## Your Memory\n\n{memory}")

            experience = self.agent_library.load_evaluator_experience(resolved_name)
            if experience:
                parts.append(f"## Your Experience\n\n{experience}")

            guardrails = self.agent_library.load_evaluator_guardrails(resolved_name)
            if guardrails:
                parts.append(f"## Current Guardrails (you maintain this)\n\n{guardrails}")

            preferences = self.agent_library.load_evaluator_preferences(resolved_name)
            if preferences:
                parts.append(f"## User Preferences (you maintain this)\n\n{preferences}")

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
        agent_name: str,
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
        result_path = self.agent_library.save_result(agent_name, result_content)

        worklog_path = self.agent_library.get_worklog_path(agent_name)
        try:
            existing = worklog_path.read_text(encoding="utf-8") if worklog_path.exists() else ""
            footer = f"\n\n---\nCompleted: {now_str} | Status: {status} | Result: {result_path}\n"
            self.agent_library.write_worklog(agent_name, existing + footer)
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
        agent_dir_name: str,
        is_reuse: bool,
        task: str,
        final_result: str,
        tools_used: list[str],
        status: str,
        eval_verdict: str = "",
        eval_rounds: int = 0,
        evaluator_feedback: str = "",
        suggested_description: str | None = None,
    ) -> None:
        """After task completion: create new agent profile or update existing one's memory.

        The directory already exists with a meaningful name (resolved at spawn time).
        This method writes AGENT.md if missing, then updates memory.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        if is_reuse:
            self.agent_library.record_usage(agent_dir_name)
            self.agent_library.append_expert_history(
                agent_dir_name,
                f"[{now_str}] Task: {task[:100]} | Tools: {', '.join(tools_used)} | Status: {status} | Eval: {eval_verdict} ({eval_rounds} rounds)",
            )
            await self._update_expert_memory(agent_dir_name, task, final_result)
            await self._update_expert_experience(agent_dir_name, task, final_result, status)
        else:
            # New agent — ensure profile exists (may already be written by error handler)
            self._ensure_agent_profile(agent_dir_name, task, suggested_description)

        # Always update evaluator state
        await self._update_evaluator_memory(agent_dir_name, task, final_result, eval_verdict, eval_rounds)
        await self._update_evaluator_guardrails(
            agent_dir_name, task, final_result, eval_verdict, eval_rounds, evaluator_feedback,
        )

    def _ensure_agent_profile(
        self,
        agent_dir_name: str,
        task: str,
        suggested_description: str | None = None,
    ) -> None:
        """Write AGENT.md and initialize agent files if they don't exist yet."""
        if self.agent_library.agent_exists(agent_dir_name):
            return

        description = suggested_description or f"Handles tasks related to: {agent_dir_name}"
        self.agent_library.create_agent(
            name=agent_dir_name,
            description=description,
            tags=[],
            task=task,
            approach="",
        )
        logger.info("Created subagent profile: {}", agent_dir_name)

    @staticmethod
    def _sanitize_agent_name(name: str) -> str | None:
        """Sanitize an agent name to short kebab-case ASCII. Returns None if unusable."""
        name = re.sub(r"[^a-z0-9\-]", "-", name.lower())
        name = re.sub(r"-{2,}", "-", name).strip("-")
        if not re.search(r"[a-z]", name) or len(name) < 2:
            return None
        return name[:30]

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

        # Extract English words from the task (NOT file paths)
        # Only match whole words, not path fragments like 'Users', 'Desktop', etc.
        skip_words = {"users", "desktop", "home", "nanobot", "workspace", "shuhangge"}
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", task)
                 if w.lower() not in skip_words]
        if words:
            slug = "-".join(words[:3])
            slug = slug[:25]
            return slug

        # Last resort: agent-{short_hash} for non-English tasks
        short_id = uuid.uuid4().hex[:6]
        return f"agent-{short_id}"

    async def _update_expert_memory(
        self,
        agent_name: str,
        task: str,
        result: str,
    ) -> None:
        """Use LLM to update an existing expert's memory with new learnings."""
        current_memory = self.agent_library.load_expert_memory(agent_name)

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
                content = response.content.strip()
                content = await self._consolidate_memory(content, _MEMORY_MAX_CHARS, "expert memory")
                self.agent_library.save_expert_memory(agent_name, content)
                logger.debug("Updated memory for expert: {}", agent_name)
        except Exception:
            logger.exception("Failed to update expert memory for {}", agent_name)

    async def _update_expert_experience(
        self,
        agent_name: str,
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
                self.agent_library.append_expert_experience(agent_name, experience_entry)
                logger.debug("Updated experience for expert: {}", agent_name)
        except Exception:
            logger.exception("Failed to update expert experience for {}", agent_name)

    async def _update_evaluator_memory(
        self,
        agent_name: str,
        task: str,
        result: str,
        verdict: str,
        eval_rounds: int,
    ) -> None:
        """Update evaluator memory with evaluation learnings."""
        now_str = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""Update the evaluator's memory with insights from this evaluation.

## Current Evaluator Memory
{self.agent_library.load_evaluator_memory(agent_name) or "(empty)"}

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
                content = response.content.strip()
                content = await self._consolidate_memory(content, _MEMORY_MAX_CHARS, "evaluator memory")
                self.agent_library.save_evaluator_memory(agent_name, content)
                # Also append experience
                experience_entry = f"""## {now_str}

### Task: {task[:80]}...
**Verdict:** {verdict} after {eval_rounds} round(s)

---
"""
                self.agent_library.append_evaluator_experience(agent_name, experience_entry)
                logger.debug("Updated evaluator memory for expert: {}", agent_name)
        except Exception:
            logger.exception("Failed to update evaluator memory for {}", agent_name)

    async def _update_evaluator_guardrails(
        self,
        agent_name: str,
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
        current = self.agent_library.load_evaluator_guardrails(agent_name)
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
                content = response.content.strip()
                content = await self._consolidate_memory(content, _GUARDRAILS_MAX_CHARS, "guardrails")
                self.agent_library.save_evaluator_guardrails(
                    agent_name, content,
                )
                logger.debug("Updated guardrails for expert: {}", agent_name)
        except Exception:
            logger.exception("Failed to update guardrails for {}", agent_name)

    async def update_evaluator_preferences(
        self,
        agent_name: str,
        feedback: str,
        verdict: str,
    ) -> None:
        """Update PREFERENCES.md based on user feedback.

        Called when the user rejects a result that the evaluator approved.
        The evaluator reflects on what it missed and records it.
        """
        current = self.agent_library.load_evaluator_preferences(agent_name)

        prompt = f"""You maintain a PREFERENCES file that captures this user's likes, dislikes,
and style preferences. Update it based on their feedback.

## Current Preferences
{current or "(empty — first feedback)"}

## User Feedback
{feedback}

## Verdict on Previous Work
{verdict}

Rules for updating:
- Keep ALL existing preference entries.
- Under "Likes": add what the user seems to prefer based on this feedback.
- Under "Dislikes (Avoid)": add what the user explicitly rejected or complained about.
- Under "Style Preferences": add any style/format preferences implied by the feedback.
- Be specific and actionable — the expert will read these to guide its work.
- Do NOT repeat entries that are already present.

Return the complete updated PREFERENCES.md content."""

        messages = [
            {"role": "system", "content": "You maintain a user preferences file. Return only the updated preferences content as markdown. Preserve all existing entries."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.model,
            )
            if response.content and response.content.strip():
                self.agent_library.save_evaluator_preferences(
                    agent_name, response.content.strip(),
                )
                logger.info("Updated preferences for {} based on user feedback", agent_name)
        except Exception:
            logger.exception("Failed to update preferences for {}", agent_name)

    # ── Memory consolidation ────────────────────────────────────────────

    async def _consolidate_memory(
        self,
        content: str,
        max_chars: int,
        label: str,
    ) -> str:
        """If content exceeds max_chars, use LLM to compress it."""
        if len(content) <= max_chars:
            return content

        prompt = f"""Compress the following {label} to under {max_chars} characters.
Keep ALL critical facts, rules, and patterns. Remove redundancy and verbosity.
Return ONLY the compressed content, nothing else.

## Content to Compress ({len(content)} chars)
{content}"""

        messages = [
            {"role": "system", "content": f"You compress {label} to fit within a character limit while preserving all essential information."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.model,
            )
            if response.content and len(response.content.strip()) <= max_chars * 1.1:
                return response.content.strip()
        except Exception:
            logger.debug("Memory consolidation failed, keeping original")
        return content[:max_chars]

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
        agent_dir_name: str = "",
    ) -> None:
        """Announce the subagent's result to the orchestrator via the message bus."""
        if verdict == "ERROR":
            status_text = "failed"
            announce_content = f"""[Subagent '{label}' {status_text}]

Full result: {result_path}

Relay the error to the user naturally. Keep it brief."""
        elif verdict == "GOOD":
            announce_content = f"""Subagent '{label}' done. GOOD, {eval_rounds} round{"s" if eval_rounds != 1 else ""}, {elapsed_seconds:.0f}s. Result saved: {result_path}

Relay this to the user naturally. Keep it brief (1-2 sentences). Do not mention subagent IDs or technical internals."""
        else:
            # NOT GOOD after max rounds
            issues_summary = ""
            if evaluator_feedback:
                # Take first 200 chars of feedback as issues summary
                issues_summary = evaluator_feedback[:200]
            announce_content = f"""Subagent '{label}' done. NOT GOOD after {eval_rounds} rounds, {elapsed_seconds:.0f}s. Issues: {issues_summary}

Full result: {result_path}
Relay this to the user naturally. Include the key issues found. Keep it brief."""
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            metadata={"_task_id": task_id, "_agent_name": agent_dir_name},
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    # ── Cancellation ─────────────────────────────────────────────────────

    def interrupt_task(self, task_id: str, feedback: str = "") -> bool:
        """Interrupt a running subagent with optional user feedback.

        Returns True if the task was found and interrupted.
        """
        evt = self._interrupt_events.get(task_id)
        if evt and not evt.is_set():
            if feedback:
                self._pending_interrupts[task_id] = feedback
            evt.set()
            return True
        return False

    def get_session_tasks(self, session_key: str) -> set[str]:
        """Return the set of task IDs for a given session."""
        return set(self._session_tasks.get(session_key, set()))

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
