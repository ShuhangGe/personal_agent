"""Memory manager - unified interface for the three-tier memory system."""

from pathlib import Path
from typing import Any

from nanobot.agent.memory_v2.config import MemoryConfig
from nanobot.agent.memory_v2.embeddings import EmbeddingProvider
from nanobot.agent.memory_v2.entities import EntityExtractor
from nanobot.agent.memory_v2.longterm import LongTermMemoryStore
from nanobot.agent.memory_v2.models import (
    LongTermMemory,
    MemoryResult,
    MemoryStats,
    MemoryTier,
    SearchFilters,
)
from nanobot.agent.memory_v2.session import SessionMemoryStore
from nanobot.agent.memory_v2.working import WorkingMemoryStore


class MemoryManager:
    """
    Unified interface for all memory operations.

    Provides a unified interface for searching, storing, and managing
    memories across all three tiers. Handles consolidation triggers
    and coordinates between different memory stores.
    """

    def __init__(
        self,
        workspace: Path,
        config: MemoryConfig | None = None,
    ) -> None:
        """
        Initialize memory manager.

        Args:
            workspace: Root directory for memory storage
            config: Memory configuration (uses defaults if None)
        """
        self.workspace = Path(workspace)
        self.config = config or MemoryConfig()

        # Initialize tier stores
        self.working = WorkingMemoryStore(
            max_messages=self.config.tiers.working.max_messages
        )

        # Initialize embedding provider
        self.embedder = EmbeddingProvider(
            provider=self.config.embeddings.provider,
            model=self.config.embeddings.model,
            dimension=self.config.embeddings.dimension,
            api_key=self.config.embeddings.api_key,
        )

        # Initialize session and longterm stores (not initialized yet)
        self.session = SessionMemoryStore(
            workspace=workspace,
            retention_days=self.config.tiers.session.retention_days,
            vec_dimension=self.config.embeddings.dimension,
        )

        self.longterm = LongTermMemoryStore(workspace=workspace)

        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(
            enabled=self.config.entities.enabled,
            min_confidence=self.config.entities.min_confidence,
            types=self.config.entities.types,
        )

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connections and indexes."""
        if self._initialized:
            return

        await self.session.initialize(embedder=self.embedder)
        await self.longterm.initialize(embedder=self.embedder)
        self._initialized = True

    async def shutdown(self) -> None:
        """Cleanup resources."""
        await self.session.shutdown()
        await self.longterm.shutdown()

    # Search methods

    async def search(
        self,
        query: str,
        *,
        tiers: list[MemoryTier] | None = None,
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> list[MemoryResult]:
        """
        Search across memory tiers.

        Args:
            query: Search query
            tiers: Tiers to search (default: all)
            filters: Optional filters
            limit: Max results to return

        Returns:
            Ranked list of memory results
        """
        await self.initialize()

        if tiers is None:
            tiers = [MemoryTier.WORKING, MemoryTier.SESSION, MemoryTier.LONGTERM]

        results = []

        # Search each tier
        if MemoryTier.WORKING in tiers:
            working_results = self.working.search(query, limit)
            results.extend(working_results)

        if MemoryTier.SESSION in tiers:
            session_results = await self.session.search(query, limit, filters)
            results.extend(session_results)

        if MemoryTier.LONGTERM in tiers:
            longterm_results = await self.longterm.search(query, limit, filters)
            # Convert to MemoryResult format
            for lt in longterm_results:
                results.append(
                    MemoryResult(
                        id=lt.id,
                        content=lt.fact,
                        tier=MemoryTier.LONGTERM,
                        score=lt.importance,
                        metadata={"category": lt.category},
                        created_at=lt.created_at,
                    )
                )

        # Re-rank by relevance
        return self._rerank(results, query)[:limit]

    async def search_recent(
        self,
        query: str,
        *,
        hours: int = 24,
        limit: int = 10,
    ) -> list[MemoryResult]:
        """Search recent session memories."""
        await self.initialize()

        from datetime import datetime, timedelta

        filters = SearchFilters(
            time_range=SearchFilters.__dataclass_fields__.get("time_range", type(None))
        )
        filters.time_range = type(
            "TimeRange",
            (),
            {"start": datetime.now() - timedelta(hours=hours)},
        )()

        return await self.session.search(query, limit, filters)

    async def search_facts(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        limit: int = 10,
    ) -> list[LongTermMemory]:
        """Search long-term facts."""
        await self.initialize()

        filters = SearchFilters(categories=categories)
        return await self.longterm.search(query, limit, filters)

    # Storage methods

    async def remember(
        self,
        content: str,
        *,
        tier: MemoryTier = MemoryTier.SESSION,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> str:
        """
        Store content in specified memory tier.

        Args:
            content: Content to store
            tier: Memory tier to store in
            metadata: Optional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            ID of stored memory
        """
        await self.initialize()

        if tier == MemoryTier.WORKING:
            return self.working.add(content, metadata)
        elif tier == MemoryTier.SESSION:
            return await self.session.add(content, metadata, importance)
        else:
            return await self.longterm.add(content, metadata, importance)

    async def forget(
        self,
        memory_id: str,
        tier: MemoryTier,
    ) -> bool:
        """Remove memory from specified tier."""
        await self.initialize()

        if tier == MemoryTier.WORKING:
            # Working memory doesn't have persistent IDs
            return False
        elif tier == MemoryTier.SESSION:
            return await self.session.remove(memory_id)
        else:
            return await self.longterm.remove(memory_id)

    # Working memory helpers

    def get_working_memory(self, session_key: str):
        """Get working memory for a session."""
        return self.working.get_or_create(session_key)

    def add_to_working_memory(
        self, content: str, session_key: str, role: str = "user", metadata: dict[str, Any] | None = None
    ) -> str:
        """Add message to working memory."""
        metadata = metadata or {}
        metadata["session_key"] = session_key
        return self.working.add(content, metadata, role)

    def clear_working_memory(self, session_key: str) -> None:
        """Clear working memory for a session."""
        self.working.clear(session_key)

    # Statistics

    async def get_stats(
        self,
        tier: MemoryTier | None = None,
    ) -> MemoryStats | dict[MemoryTier, MemoryStats]:
        """Get memory usage statistics."""
        await self.initialize()

        if tier == MemoryTier.WORKING:
            return self.working.get_stats()
        elif tier == MemoryTier.SESSION:
            return await self.session.get_stats()
        elif tier == MemoryTier.LONGTERM:
            return await self.longterm.get_stats()
        else:
            # Return stats for all tiers
            return {
                MemoryTier.WORKING: self.working.get_stats(),
                MemoryTier.SESSION: await self.session.get_stats(),
                MemoryTier.LONGTERM: await self.longterm.get_stats(),
            }

    # Utility methods

    def _rerank(self, results: list[MemoryResult], query: str) -> list[MemoryResult]:
        """Re-rank results by relevance."""
        # Simple re-ranking based on tier preference
        # Prefer working > session > longterm for same score
        tier_priority = {
            MemoryTier.WORKING: 3,
            MemoryTier.SESSION: 2,
            MemoryTier.LONGTERM: 1,
        }

        def sort_key(result: MemoryResult):
            return (result.score, tier_priority.get(result.tier, 0))

        return sorted(results, key=sort_key, reverse=True)

    async def get_memory_context(self, current_query: str | None = None) -> str:
        """
        Get relevant memory context for the current query.

        Args:
            current_query: Current query to find relevant memories for

        Returns:
            Formatted memory context string
        """
        await self.initialize()

        if not current_query:
            # Load all long-term memory
            if self.longterm.memory_file.exists():
                return self.longterm.memory_file.read_text(encoding="utf-8")
            return ""

        # Search for relevant memories
        relevant = await self.search_facts(current_query, limit=5)

        if not relevant:
            return ""

        # Format results
        lines = ["## Relevant Memories"]
        for mem in relevant:
            lines.append(f"- {mem.fact}")

        return "\n".join(lines)
