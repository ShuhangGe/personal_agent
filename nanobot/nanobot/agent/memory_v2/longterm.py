"""Long-term memory store - Hybrid Markdown + SQLite storage."""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from nanobot.agent.memory_v2.embeddings import EmbeddingProvider
from nanobot.agent.memory_v2.models import (
    LongTermMemory,
    MemoryResult,
    MemoryStats,
    MemoryTier,
    SearchFilters,
)


class LongTermMemoryStore:
    """
    Hybrid storage for long-term knowledge.

    Combines human-readable Markdown files with searchable
    SQLite storage and vector embeddings.
    """

    def __init__(self, workspace: Path) -> None:
        """
        Initialize long-term memory store.

        Args:
            workspace: Workspace directory
        """
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.db_path = self.memory_dir / "longterm_memories.db"

        self._db: aiosqlite.Connection | None = None
        self._embedder: EmbeddingProvider | None = None

    async def initialize(self, embedder: EmbeddingProvider | None = None) -> None:
        """Initialize database and load existing memories."""
        self._db = await aiosqlite.connect(self.db_path)
        self._embedder = embedder

        # Enable WAL mode
        await self._db.execute("PRAGMA journal_mode=WAL")

        await self._create_tables()

        # Index existing MEMORY.md if not indexed
        await self._index_existing_memories()

    async def _create_tables(self) -> None:
        """Create database tables."""
        # Main facts table
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS longterm_memories (
                id TEXT PRIMARY KEY,
                fact TEXT NOT NULL,
                category TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                expires_at TEXT
            )
        """
        )

        # Full-text search table
        await self._db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS longterm_memories_fts
            USING fts5(fact, category, content=longterm_memories, content_rowid=rowid)
        """
        )

        # Create indexes
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_category
            ON longterm_memories(category)
        """
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_importance
            ON longterm_memories(importance)
        """
        )

        await self._db.commit()

    async def _index_existing_memories(self) -> None:
        """Index existing facts from MEMORY.md if not already indexed."""
        # Check if we have any indexed memories
        cursor = await self._db.execute("SELECT COUNT(*) FROM longterm_memories")
        count = (await cursor.fetchone())[0]

        if count > 0:
            return  # Already indexed

        # Parse MEMORY.md and index facts
        if not self.memory_file.exists():
            return

        content = self.memory_file.read_text(encoding="utf-8")
        await self._parse_and_index_memory_md(content)

    async def _parse_and_index_memory_md(self, content: str) -> None:
        """Parse MEMORY.md content and index facts."""
        import uuid

        lines = content.split("\n")
        current_category = "general"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for category headers
            if line.startswith("## ") or line.startswith("# "):
                current_category = line.lstrip("#").strip().lower()
                continue

            # Check for bullet points (facts)
            if line.startswith("- ") or line.startswith("* "):
                fact = line[2:].strip()

                # Index in database
                memory_id = str(uuid.uuid4())
                now = datetime.now().isoformat()

                await self._db.execute(
                    """
                    INSERT INTO longterm_memories
                    (id, fact, category, created_at, updated_at, importance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (memory_id, fact, current_category, now, now, 1.0),
                )

                await self._db.execute(
                    """
                    INSERT INTO longterm_memories_fts (rowid, fact, category)
                    VALUES (?, ?, ?)
                """,
                    (memory_id, fact, current_category),
                )

        await self._db.commit()

    async def add(
        self,
        fact: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 1.0,
    ) -> str:
        """
        Add fact to long-term memory.

        Args:
            fact: Fact to store
            metadata: Optional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            ID of stored fact
        """
        import uuid

        memory_id = str(uuid.uuid4())
        category = metadata.get("category", "general") if metadata else "general"

        # Generate embedding if provider available
        embedding = None
        if self._embedder:
            embedding = await self._embedder.embed(fact)
            # Convert numpy array to bytes for storage
            if embedding is not None and hasattr(embedding, 'tobytes'):
                embedding = embedding.tobytes()

        now = datetime.now().isoformat()

        # Store in database
        await self._db.execute(
            """
            INSERT INTO longterm_memories
            (id, fact, category, embedding, created_at, updated_at, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                memory_id,
                fact,
                category,
                embedding,
                now,
                now,
                importance,
            ),
        )

        # Add to FTS
        await self._db.execute(
            """
            INSERT INTO longterm_memories_fts (rowid, fact, category)
            VALUES (?, ?, ?)
        """,
            (memory_id, fact, category),
        )

        await self._db.commit()

        # Append to MEMORY.md
        await self._append_to_memory_md(fact, category)

        return memory_id

    async def _append_to_memory_md(self, fact: str, category: str) -> None:
        """Append fact to MEMORY.md file."""
        # Read existing content
        existing_content = ""
        if self.memory_file.exists():
            existing_content = self.memory_file.read_text(encoding="utf-8")

        # Check if category exists
        category_header = f"## {category.title()}"
        if category_header in existing_content:
            # Append to existing category
            lines = existing_content.split("\n")
            new_lines = []
            inserted = False

            for i, line in enumerate(lines):
                new_lines.append(line)
                if not inserted and line.strip() == category_header:
                    # Find the end of this category
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("## "):
                            # Insert before next category
                            new_lines.insert(j, f"- {fact}")
                            inserted = True
                            break
                    if not inserted:
                        # End of file, append here
                        new_lines.append(f"- {fact}")
                        inserted = True

            content = "\n".join(new_lines)
        else:
            # Add new category
            if existing_content and not existing_content.endswith("\n"):
                existing_content += "\n"
            content = f"{existing_content}\n{category_header}\n- {fact}\n"

        # Write back
        self.memory_file.write_text(content, encoding="utf-8")

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[LongTermMemory]:
        """
        Search long-term memories.

        Args:
            query: Search query
            limit: Max results to return
            filters: Optional filters

        Returns:
            List of LongTermMemory objects
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
                SELECT lm.id, lm.fact, lm.category, lm.created_at, lm.importance
                FROM longterm_memories lm
                JOIN longterm_memories_fts lm_fts ON lm.id = lm_fts.rowid
                WHERE longterm_memories_fts MATCH ?
                ORDER BY bm25(lm_fts)
                LIMIT ?
            """,
                (query, limit),
            )

            rows = await cursor.fetchall()
            return [
                MemoryResult(
                    id=row[0],
                    content=row[1],
                    tier=MemoryTier.LONGTERM,
                    score=1.0,
                    metadata={"category": row[2], "created_at": row[3], "importance": row[4]},
                    created_at=datetime.fromisoformat(row[3]),
                )
                for row in rows
            ]
        except Exception:
            # FTS5 query might fail, try simple LIKE query
            cursor = await self._db.execute(
                """
                SELECT id, fact, category, created_at, importance
                FROM longterm_memories
                WHERE fact LIKE ?
                ORDER BY importance DESC
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )

            rows = await cursor.fetchall()
            return [
                MemoryResult(
                    id=row[0],
                    content=row[1],
                    tier=MemoryTier.LONGTERM,
                    score=row[4],
                    metadata={"category": row[2], "created_at": row[3]},
                    created_at=datetime.fromisoformat(row[3]),
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
                SELECT id, fact, category, created_at, importance, embedding
                FROM longterm_memories
                WHERE embedding IS NOT NULL
                LIMIT 100
            """
            )

            rows = await cursor.fetchall()
            results = []

            for row in rows:
                embedding_bytes = row[5]
                embedding = None
                if embedding_bytes is not None:
                    # Convert bytes back to numpy array
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    # Reshape if needed
                    if embedding.ndim == 0:
                        embedding = embedding.reshape(-1)

                if embedding is not None:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    results.append(
                        MemoryResult(
                            id=row[0],
                            content=row[1],
                            tier=MemoryTier.LONGTERM,
                            score=similarity,
                            metadata={"category": row[2], "created_at": row[3], "importance": row[4]},
                            created_at=datetime.fromisoformat(row[3]),
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
        if filters.time_range and result.created_at:
            if filters.time_range.start and result.created_at < filters.time_range.start:
                return False
            if filters.time_range.end and result.created_at > filters.time_range.end:
                return False

        if filters.categories:
            category = result.metadata.get("category", "")
            if category not in filters.categories:
                return False

        if filters.min_importance:
            importance = result.metadata.get("importance", 0.0)
            if importance < filters.min_importance:
                return False

        return True

    async def remove(self, memory_id: str) -> bool:
        """Remove fact from long-term memory."""
        await self._db.execute("DELETE FROM longterm_memories WHERE id = ?", (memory_id,))
        await self._db.commit()

        # Rebuild MEMORY.md (simplified)
        await self._rebuild_memory_md()

        return True

    async def _rebuild_memory_md(self) -> None:
        """Rebuild MEMORY.md from database."""
        cursor = await self._db.execute(
            "SELECT fact, category FROM longterm_memories ORDER BY category, created_at"
        )
        rows = await cursor.fetchall()

        # Group by category
        categories = {}
        for fact, category in rows:
            if category not in categories:
                categories[category] = []
            categories[category].append(fact)

        # Build content
        lines = ["# Long-term Memory\n"]
        for category, facts in sorted(categories.items()):
            lines.append(f"\n## {category.title()}\n")
            for fact in facts:
                lines.append(f"- {fact}")

        # Write to file
        self.memory_file.write_text("\n".join(lines), encoding="utf-8")

    async def append_history(self, entry: str) -> None:
        """Append entry to HISTORY.md."""
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    async def get_stats(self) -> MemoryStats:
        """Get long-term memory statistics."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM longterm_memories")
        count = (await cursor.fetchone())[0]

        # Calculate storage size
        storage_size = sum(
            f.stat().st_size for f in [self.db_path, self.memory_file, self.history_file]
            if f.exists()
        )

        return MemoryStats(
            tier=MemoryTier.LONGTERM, total_memories=count, storage_size_bytes=storage_size
        )

    async def shutdown(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
