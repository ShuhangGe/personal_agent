"""Session memory store - SQLite-backed medium-term storage with semantic search."""

import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from nanobot.agent.memory_v2.embeddings import EmbeddingProvider
from nanobot.agent.memory_v2.models import (
    MemoryResult,
    MemoryStats,
    MemoryTier,
    SearchFilters,
    SessionMemory,
)


class SessionMemoryStore:
    """
    SQLite-backed storage for session memories with semantic search.

    Provides medium-term storage with vector similarity search,
    entity extraction, and automatic expiration.
    """

    def __init__(
        self, workspace: Path, retention_days: int = 30, vec_dimension: int = 384
    ) -> None:
        """
        Initialize session memory store.

        Args:
            workspace: Workspace directory
            retention_days: Days to retain memories
            vec_dimension: Dimension of vector embeddings
        """
        self.workspace = workspace
        self.retention_days = retention_days
        self.vec_dimension = vec_dimension

        # Create memory directory
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.memory_dir / "session_memories.db"

        self._db: aiosqlite.Connection | None = None
        self._embedder: EmbeddingProvider | None = None

    async def initialize(self, embedder: EmbeddingProvider | None = None) -> None:
        """
        Initialize database connection and create tables.

        Args:
            embedder: Embedding provider for semantic search
        """
        self._db = await aiosqlite.connect(self.db_path)
        self._embedder = embedder

        # Enable WAL mode for better concurrency
        await self._db.execute("PRAGMA journal_mode=WAL")

        await self._create_tables()
        await self._create_indexes()

        # Clean up expired memories
        await self.cleanup_expired()

    async def _create_tables(self) -> None:
        """Create database tables."""
        # Main memories table
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS session_memories (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                importance_score REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                topics TEXT,
                sentiment REAL DEFAULT 0.0
            )
        """
        )

        # Full-text search table
        await self._db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS session_memories_fts
            USING fts5(summary, content=session_memories, content_rowid=rowid)
        """
        )

        # Note: sqlite-vec would be created here if available
        # For now, we'll store embeddings in the main table

    async def _create_indexes(self) -> None:
        """Create database indexes."""
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_key
            ON session_memories(session_key)
        """
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON session_memories(created_at)
        """
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_importance
            ON session_memories(importance_score)
        """
        )

        await self._db.commit()

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> str:
        """
        Add memory to session store.

        Args:
            content: Content to store
            metadata: Optional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            ID of stored memory
        """
        import uuid

        memory_id = str(uuid.uuid4())
        session_key = metadata.get("session_key", "default") if metadata else "default"

        # Generate embedding if provider available
        embedding = None
        if self._embedder:
            embedding = await self._embedder.embed(content)
            # Convert numpy array to bytes for storage
            if embedding is not None and hasattr(embedding, 'tobytes'):
                embedding = embedding.tobytes()

        now = datetime.now().isoformat()

        # Store in database
        await self._db.execute(
            """
            INSERT INTO session_memories
            (id, session_key, summary, embedding, created_at, updated_at, importance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                memory_id,
                session_key,
                content,
                embedding,
                now,
                now,
                importance,
            ),
        )

        # Add to FTS
        await self._db.execute(
            """
            INSERT INTO session_memories_fts (rowid, summary)
            VALUES (?, ?)
        """,
            (memory_id, content),
        )

        await self._db.commit()
        return memory_id

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[MemoryResult]:
        """
        Search session memories.

        Args:
            query: Search query
            limit: Max results to return
            filters: Optional filters

        Returns:
            List of memory results
        """
        results = []

        # Keyword search using FTS5
        keyword_results = await self._search_keyword(query, limit * 2)
        results.extend(keyword_results)

        # Semantic search if embedder available
        if self._embedder:
            semantic_results = await self._search_semantic(query, limit * 2)
            # Merge with reciprocal rank fusion
            results = self._merge_results(results, semantic_results)

        # Apply filters
        if filters:
            results = [r for r in results if self._matches_filters(r, filters)]

        # Sort by score and limit
        results = sorted(results, key=lambda x: -x.score)[:limit]

        return results

    async def _search_keyword(self, query: str, limit: int) -> list[MemoryResult]:
        """Search using FTS5."""
        try:
            cursor = await self._db.execute(
                """
                SELECT sm.id, sm.summary, sm.created_at, sm.importance_score
                FROM session_memories sm
                JOIN session_memories_fts sm_fts ON sm.id = sm_fts.rowid
                WHERE session_memories_fts MATCH ?
                ORDER BY bm25(sm_fts)
                LIMIT ?
            """,
                (query, limit),
            )

            rows = await cursor.fetchall()
            return [
                MemoryResult(
                    id=row[0],
                    content=row[1],
                    tier=MemoryTier.SESSION,
                    score=1.0,  # Will be re-ranked
                    metadata={"created_at": row[2], "importance": row[3]},
                    created_at=datetime.fromisoformat(row[2]),
                )
                for row in rows
            ]
        except Exception:
            # FTS5 query might fail, try simple LIKE query
            cursor = await self._db.execute(
                """
                SELECT id, summary, created_at, importance_score
                FROM session_memories
                WHERE summary LIKE ?
                ORDER BY importance_score DESC
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )

            rows = await cursor.fetchall()
            return [
                MemoryResult(
                    id=row[0],
                    content=row[1],
                    tier=MemoryTier.SESSION,
                    score=row[3],
                    metadata={"created_at": row[2]},
                    created_at=datetime.fromisoformat(row[2]),
                )
                for row in rows
            ]

    async def _search_semantic(self, query: str, limit: int) -> list[MemoryResult]:
        """Search using vector similarity."""
        if not self._embedder:
            return []

        try:
            query_embedding = await self._embedder.embed(query)

            # Fetch all memories with embeddings
            cursor = await self._db.execute(
                """
                SELECT id, summary, created_at, importance_score, embedding
                FROM session_memories
                WHERE embedding IS NOT NULL
                LIMIT 100
            """
            )

            rows = await cursor.fetchall()
            results = []

            for row in rows:
                embedding_bytes = row[4]
                embedding = None
                if embedding_bytes is not None:
                    # Convert bytes back to numpy array
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    # Reshape if needed (assuming 1D array for now)
                    if embedding.ndim == 0:
                        embedding = embedding.reshape(-1)
                if embedding is not None:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    results.append(
                        MemoryResult(
                            id=row[0],
                            content=row[1],
                            tier=MemoryTier.SESSION,
                            score=similarity,
                            metadata={"created_at": row[2], "importance": row[3]},
                            created_at=datetime.fromisoformat(row[2]),
                        )
                    )

            # Sort by similarity and return top results
            results.sort(key=lambda x: -x.score)
            return results[:limit]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
        except Exception:
            return 0.0

    def _merge_results(
        self, results1: list[MemoryResult], results2: list[MemoryResult]
    ) -> list[MemoryResult]:
        """Merge result lists with reciprocal rank fusion."""
        scores = {}

        # Score from first list
        for i, r in enumerate(results1):
            scores[r.id] = scores.get(r.id, 0) + 1.0 / (i + 1)

        # Score from second list
        for i, r in enumerate(results2):
            scores[r.id] = scores.get(r.id, 0) + 1.0 / (i + 1)

        # Create merged results
        merged = {}
        for r in results1 + results2:
            if r.id not in merged:
                merged[r.id] = r
            merged[r.id].score = scores.get(r.id, 0.0)

        # Sort by combined score
        return sorted(merged.values(), key=lambda x: -x.score)

    def _matches_filters(self, result: MemoryResult, filters: SearchFilters) -> bool:
        """Check if result matches filters."""
        if filters.time_range:
            if result.created_at:
                if filters.time_range.start and result.created_at < filters.time_range.start:
                    return False
                if filters.time_range.end and result.created_at > filters.time_range.end:
                    return False

        if filters.min_importance:
            importance = result.metadata.get("importance", 0.0)
            if importance < filters.min_importance:
                return False

        return True

    async def remove(self, memory_id: str) -> bool:
        """Remove memory from session store."""
        await self._db.execute("DELETE FROM session_memories WHERE id = ?", (memory_id,))
        await self._db.commit()
        return True

    async def cleanup_expired(self) -> None:
        """Remove memories older than retention period."""
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()

        await self._db.execute(
            "DELETE FROM session_memories WHERE created_at < ?", (cutoff,)
        )
        await self._db.commit()

    async def get_stats(self) -> MemoryStats:
        """Get session memory statistics."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM session_memories")
        count = (await cursor.fetchone())[0]

        # Calculate storage size
        storage_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return MemoryStats(
            tier=MemoryTier.SESSION,
            total_memories=count,
            retention_days=self.retention_days,
            storage_size_bytes=storage_size,
        )

    async def shutdown(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
