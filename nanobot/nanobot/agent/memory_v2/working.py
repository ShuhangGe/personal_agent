"""Working memory store - short-term in-memory storage."""

from typing import Any

from nanobot.agent.memory_v2.models import (
    Message,
    MemoryResult,
    MemoryStats,
    MemoryTier,
    WorkingMemory,
)


class WorkingMemoryStore:
    """
    In-memory storage for active conversation context.

    Provides fast access to recent messages and temporary state.
    Automatically manages context window size.
    """

    def __init__(self, max_messages: int = 100) -> None:
        """
        Initialize working memory store.

        Args:
            max_messages: Maximum messages per session
        """
        self.max_messages = max_messages
        self._sessions: dict[str, WorkingMemory] = {}

    def get_or_create(self, session_key: str) -> WorkingMemory:
        """Get existing session or create new one."""
        if session_key not in self._sessions:
            self._sessions[session_key] = WorkingMemory(
                session_key=session_key, context_window=self.max_messages
            )
        return self._sessions[session_key]

    def add(
        self, content: str, metadata: dict[str, Any] | None = None, role: str = "user"
    ) -> str:
        """
        Add message to working memory.

        Args:
            content: Message content
            metadata: Optional metadata (must include session_key)
            role: Message role (user, assistant, etc.)

        Returns:
            ID of added message
        """
        metadata = metadata or {}
        session_key = metadata.get("session_key", "default")
        memory = self.get_or_create(session_key)

        message = Message(
            content=content, role=role, timestamp=memory.last_accessed, metadata=metadata
        )

        memory.add_message(message)
        return message.id

    def search(self, query: str, limit: int = 10) -> list[MemoryResult]:
        """
        Simple keyword search in working memory.

        Args:
            query: Search query
            limit: Max results to return

        Returns:
            List of matching memory results
        """
        results = []
        query_lower = query.lower()

        for session in self._sessions.values():
            for msg in session.messages:
                if query_lower in msg.content.lower():
                    results.append(
                        MemoryResult(
                            id=msg.id,
                            content=msg.content,
                            tier=MemoryTier.WORKING,
                            score=1.0,  # Perfect match for working memory
                            metadata=msg.metadata,
                            created_at=msg.timestamp,
                        )
                    )
                    if len(results) >= limit:
                        return results

        return results

    def get(self, session_key: str) -> WorkingMemory | None:
        """Get working memory for session."""
        return self._sessions.get(session_key)

    def get_context(
        self, session_key: str, max_messages: int = 10
    ) -> list[Message]:
        """Get recent messages for a session."""
        memory = self.get(session_key)
        if memory is None:
            return []
        return memory.get_context()[-max_messages:]

    def clear(self, session_key: str) -> None:
        """Clear working memory for session."""
        self._sessions.pop(session_key, None)

    def get_stats(self) -> MemoryStats:
        """Get working memory statistics."""
        total_messages = sum(len(s.messages) for s in self._sessions.values())
        return MemoryStats(
            tier=MemoryTier.WORKING, total_memories=len(self._sessions), total_messages=total_messages
        )

    def cleanup_expired(self, timeout_minutes: int = 30) -> None:
        """Remove sessions that haven't been accessed recently."""
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(minutes=timeout_minutes)
        expired_keys = [
            key
            for key, session in self._sessions.items()
            if session.last_accessed < cutoff
        ]
        for key in expired_keys:
            self._sessions.pop(key, None)
