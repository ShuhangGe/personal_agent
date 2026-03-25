"""Spawn tool for delegating tasks to expert subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn an expert subagent for task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for expert announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn an expert subagent to handle a task. "
            "Use expert_name to reuse a saved expert, or omit it for a new generic expert. "
            "The expert will work independently, maintain a live work log, "
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
                        "Detailed task description with all necessary context for the expert. "
                        "Be specific — the expert cannot see the conversation history."
                    ),
                },
                "expert_name": {
                    "type": "string",
                    "description": (
                        "Name of a saved expert to use (from the expert library). "
                        "Omit to spawn a generic expert for new task types."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Short display label for tracking (shown to user)",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant conversation context or background info the expert needs",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        expert_name: str | None = None,
        label: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn an expert subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            expert_name=expert_name,
            context=context,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )
