"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanobot.agent.agent_library import AgentLibrary
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import build_assistant_message, detect_image_mime

if TYPE_CHECKING:
    from nanobot.agent.enhanced_memory import EnhancedMemoryConsolidator


class ContextBuilder:
    """Builds the context (system prompt + messages) for the orchestrator agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path, enhanced_memory: "EnhancedMemoryConsolidator | None" = None):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        self.agent_library = AgentLibrary(workspace)
        self.enhanced_memory = enhanced_memory

    def build_system_prompt(self, skill_names: list[str] | None = None, query: str | None = None) -> str:
        """Build the orchestrator system prompt: identity, memory, agent library (no skills).

        Args:
            skill_names: Optional list of skills to activate
            query: Optional search query for semantic memory retrieval
        """
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # Use semantic search if available and query is provided
        memory_context = self._get_memory_context_with_search(query)
        if memory_context:
            parts.append(f"# Memory\n\n{memory_context}")

        agent_summary = self.agent_library.build_agent_summary()
        if agent_summary:
            parts.append(f"""# Agent Library

You have a library of saved subagents. When a task matches an existing subagent with matching expertise, use its name in the spawn call.
When no subagent matches, omit agent_name to spawn a new subagent (a new subagent profile will be saved automatically after the task completes).

{agent_summary}""")
        else:
            parts.append("""# Agent Library

No saved subagents yet. When you spawn a task, a new subagent will be used.
After it completes, a new subagent profile will be saved automatically for future reuse.""")

        return "\n\n---\n\n".join(parts)

    def _get_memory_context_with_search(self, query: str | None = None) -> str:
        """Get memory context, optionally using semantic search.

        Args:
            query: Optional search query. If provided and semantic search is available,
                   will retrieve relevant memories.

        Returns:
            Memory context string
        """
        # Always include long-term memory
        long_term = self.memory.read_long_term()

        # If we have enhanced memory and a query, we could do semantic search
        # For now, we'll just use the basic memory context
        # Semantic search can be added later as an enhancement
        if long_term:
            return f"## Long-term Memory\n{long_term}"
        return ""

    def _get_identity(self) -> str:
        """Get the core identity section for the orchestrator."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# nanobot Orchestrator 🐈

You are nanobot, an orchestrator AI assistant. Your job is to **plan, delegate, and monitor** — NOT to execute tasks directly.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md
- Agent library: {workspace_path}/agents/

## How You Work

1. **Understand** the user's request
2. **Delegate** by spawning subagents via the `spawn` tool
3. **Monitor** — subagents maintain live work logs you can reference
4. **Respond** — relay subagent results to the user naturally

## Delegation Rules

- For any task requiring tools (file operations, web search, code execution, etc.), **spawn a subagent**.
- **ONE task = ONE subagent.** Give the subagent a single, comprehensive task description with ALL the work it needs to do. The subagent has its own tools and will figure out the steps itself. Do NOT break a task into micro-steps and spawn separate subagents for each step — that creates unnecessary subagents and wastes resources.
- If a subagent with matching expertise exists in your library, specify its `agent_name`.
- If no subagent matches, spawn without `agent_name` — a new subagent will handle it and a new profile will be saved.
- Provide detailed task descriptions and relevant context to subagents — they cannot see the conversation.
- Only spawn multiple subagents when subtasks are truly **independent and unrelated** (e.g., "research topic A" and "fix bug in module B"). Sequential steps of the same task must go to ONE subagent.

## Fast Path (respond directly)

For these, respond directly WITHOUT spawning:
- Greetings, acknowledgments, thanks
- Simple clarifying questions
- Explaining what you've done or plan to do
- Relaying subagent results to the user

## Guidelines
- State intent before delegating, but NEVER predict results before receiving them.
- Ask for clarification when the request is ambiguous.
- When a subagent reports back, relay the result naturally to the user.
- If the user wants details, point them to the subagent's result file or work log."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        user_content = self._build_user_content(current_message, media)

        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
