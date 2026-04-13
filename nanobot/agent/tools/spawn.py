"""Spawn tool for delegating tasks to subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._on_progress = None

    def set_progress_callback(self, callback) -> None:
        """Set the progress callback for subagent notifications."""
        self._on_progress = callback

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task. "
            "Use agent_name to reuse a saved subagent, or omit it for a new subagent. "
            "The subagent will work independently, maintain a live work log, "
            "save detailed results to a file, and report back a short summary when done."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Detailed task description with all necessary context for the subagent. "
                        "Be specific — the subagent cannot see the conversation history."
                    ),
                },
                "agent_name": {
                    "type": "string",
                    "description": (
                        "Name of a saved subagent to use (from the agent library). "
                        "Omit to spawn a new subagent for new task types."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Short display label for tracking (shown to user)",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant conversation context or background info the subagent needs",
                },
                "suggested_name": {
                    "type": "string",
                    "description": (
                        "Short kebab-case name for a new subagent (2-3 English words, max 30 chars). "
                        "Only provide when NOT reusing an existing agent via agent_name. "
                        "Examples: 'novel-analyzer', 'web-scraper', 'code-reviewer'. "
                        "Use English only, no special characters."
                    ),
                },
                "suggested_description": {
                    "type": "string",
                    "description": (
                        "One-sentence description of this agent's GENERAL capability (not this specific task). "
                        "Only provide when NOT reusing an existing agent. "
                        "Example: 'Analyzes novels and long-form fiction' not 'Analyzes the novel XYZ'"
                    ),
                },
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Shared output directory for batch tasks. "
                        "The subagent can write to this directory in addition to its own workspace. "
                        "Use when spawning multiple subagents that should produce results in one place."
                    ),
                },
                "group_id": {
                    "type": "string",
                    "description": (
                        "ID to group related subagents into one batch task. "
                        "All subagents with the same group_id are tracked together. "
                        "You receive ONE combined notification when ALL subagents in the group finish, "
                        "instead of individual notifications for each. "
                        "Use any short unique identifier, e.g. 'novel-analysis-batch'."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        agent_name: str | None = None,
        label: str | None = None,
        context: str | None = None,
        suggested_name: str | None = None,
        suggested_description: str | None = None,
        output_dir: str | None = None,
        group_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            agent_name=agent_name,
            context=context,
            suggested_name=suggested_name,
            suggested_description=suggested_description,
            output_dir=output_dir,
            group_id=group_id,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
            on_progress=self._on_progress,
        )
