"""Memory search engine with hybrid search capabilities."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path
import math

import numpy as np
from loguru import logger

try:
    from whoosh.index import create_in, open_dir, exists_in
    from whoosh.fields import Schema, TEXT, ID, DATETIME, FLOAT
    from whoosh.qparser import QueryParser, MultifieldParser
    from whoosh.query import And, Or, Term
    import whoosh.scoring as scoring
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False
    logger.warning("Whoosh not available, keyword search will be limited")

from nanobot.agent.memory.types import (
    MemorySearchConfig,
    MemorySearchResult,
    SearchResult,
    SearchMode,
    MemoryEntry,
)
from nanobot.agent.memory.vector_store import VectorStore
from nanobot.agent.memory.embeddings import EmbeddingProvider


class MemorySearchEngine:
    """Advanced memory search with hybrid capabilities."""

    def __init__(
        self,
        workspace: Path,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ):
        self.workspace = workspace
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self._index_dir = workspace / "memory" / "search_index"
        self._lock = asyncio.Lock()

    def _get_index_path(self) -> Path:
        """Get path for Whoosh index."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        return self._index_dir

    def _get_schema(self) -> Any:
        """Get Whoosh schema for full-text search."""
        return Schema(
            memory_id=ID(stored=True, unique=True),
            content=TEXT(stored=True),
            source=ID(stored=True),
            timestamp=DATETIME(stored=True),
            importance=FLOAT(stored=True),
            tags=TEXT(stored=True),
        )

    def _get_index(self) -> Any:
        """Get or create Whoosh index."""
        if not WHOOSH_AVAILABLE:
            raise RuntimeError("Whoosh is not available")

        index_path = self._get_index_path()

        try:
            if exists_in(str(index_path)):
                index = open_dir(str(index_path))
            else:
                index = create_in(str(index_path), schema=self._get_schema())
            return index
        except Exception as e:
            logger.error(f"Failed to get Whoosh index: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize the search engine."""
        try:
            if WHOOSH_AVAILABLE:
                self._get_index()
                logger.info("Search engine initialized with Whoosh")
            else:
                logger.info("Search engine initialized (Whoosh unavailable)")

            await self.vector_store.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise

    async def search(
        self,
        query: str,
        config: Optional[MemorySearchConfig] = None,
    ) -> MemorySearchResult:
        """Search memory using configured strategy.

        Args:
            query: Search query
            config: Search configuration (uses defaults if None)

        Returns:
            Complete search result with metadata
        """
        if config is None:
            config = MemorySearchConfig()

        start_time = time.time()
        used_vector = False
        used_keyword = False

        try:
            results: list[SearchResult] = []

            # Execute search based on mode
            if config.mode == SearchMode.VECTOR:
                results = await self._vector_search(query, config)
                used_vector = True

            elif config.mode == SearchMode.KEYWORD:
                results = await self._keyword_search(query, config)
                used_keyword = True

            elif config.mode == SearchMode.HYBRID:
                results = await self._hybrid_search(query, config)
                used_vector = True
                used_keyword = True

            elif config.mode == SearchMode.TEMPORAL:
                results = await self._temporal_search(query, config)
                used_vector = True  # Temporal uses vector as base

            # Apply filters
            results = self._apply_filters(results, config)

            # Apply relevance threshold
            results = [
                r for r in results
                if r.score >= config.similarity_threshold
            ]

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)

            # Update ranks
            for i, result in enumerate(results):
                result.rank = i

            # Limit results
            results = results[:config.max_results]

            # Apply token limit if configured
            filtered_by_tokens = False
            estimated_tokens = 0
            if config.estimate_tokens:
                results, estimated_tokens = self._apply_token_limit(
                    results, config.max_tokens
                )
                filtered_by_tokens = len(results) < config.min(config.max_results, len(results))

            search_time = (time.time() - start_time) * 1000  # Convert to ms

            return MemorySearchResult(
                query=query,
                results=results,
                mode=config.mode,
                total_results=len(results),
                search_time_ms=search_time,
                used_vector_search=used_vector,
                used_keyword_search=used_keyword,
                filtered_by_tokens=filtered_by_tokens,
                estimated_tokens=estimated_tokens,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty result on error
            return MemorySearchResult(
                query=query,
                results=[],
                mode=config.mode,
                total_results=0,
                search_time_ms=(time.time() - start_time) * 1000,
                used_vector_search=False,
                used_keyword_search=False,
            )

    async def _vector_search(
        self,
        query: str,
        config: MemorySearchConfig,
    ) -> list[SearchResult]:
        """Pure vector semantic search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.embed_with_cache(query)

            # Build metadata filters
            filter_metadata = {}
            if config.allowed_sources:
                # Can't filter by list in ChromaDB, skip for now
                pass
            if config.min_importance > 0:
                filter_metadata["importance"] = {"$gte": config.min_importance}

            # Search vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                n_results=config.max_results * 2,  # Get more for ranking
                filter_metadata=filter_metadata if filter_metadata else None,
            )

            # Adjust scores based on importance and temporal decay
            return self._adjust_scores(results, config)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _keyword_search(
        self,
        query: str,
        config: MemorySearchConfig,
    ) -> list[SearchResult]:
        """Pure keyword-based full-text search."""
        if not WHOOSH_AVAILABLE:
            logger.warning("Whoosh not available, keyword search disabled")
            return []

        try:
            index = self._get_index()

            with index.searcher() as searcher:
                # Parse query
                parser = MultifieldParser(
                    ["content", "tags"],
                    schema=index.schema
                )
                query_obj = parser.parse(query)

                # Search
                hits = searcher.search(
                    query_obj,
                    limit=config.max_results * 2,
                    weighted=True,
                )

                # Convert to SearchResult
                results = []
                for i, hit in enumerate(hits):
                    # Create mock MemoryEntry from hit
                    memory = MemoryEntry(
                        id=hit["memory_id"],
                        content=hit.get("content", ""),
                        source=hit.get("source", "conversation"),
                        importance=hit.get("importance", 0.5),
                        tags=hit.get("tags", "").split(","),
                        timestamp=hit.get("timestamp", datetime.now()),
                    )

                    # Use score as relevance (Whoosh provides BM25 scoring)
                    score = min(1.0, hit.score / 10.0)  # Normalize roughly

                    results.append(
                        SearchResult(
                            memory=memory,
                            score=score,
                            rank=i,
                        )
                    )

                return self._adjust_scores(results, config)

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    async def _hybrid_search(
        self,
        query: str,
        config: MemorySearchConfig,
    ) -> list[SearchResult]:
        """Hybrid search combining vector and keyword results."""
        # Get results from both methods
        vector_results = await self._vector_search(query, config)
        keyword_results = await self._keyword_search(query, config)

        # Merge results using score fusion
        merged = {}

        # Add vector results with weight
        for result in vector_results:
            memory_id = result.memory.id
            if memory_id not in merged:
                merged[memory_id] = result
                merged[memory_id].score = result.score * config.vector_weight
            else:
                # Combine scores
                merged[memory_id].score += result.score * config.vector_weight

        # Add keyword results with weight
        for result in keyword_results:
            memory_id = result.memory.id
            if memory_id not in merged:
                merged[memory_id] = result
                merged[memory_id].score = result.score * config.keyword_weight
            else:
                # Combine scores
                merged[memory_id].score += result.score * config.keyword_weight

        # Normalize scores and apply diversity
        results = list(merged.values())
        results = self._apply_diversity(results, config.diversity_threshold)
        results = self._adjust_scores(results, config)

        return results

    async def _temporal_search(
        self,
        query: str,
        config: MemorySearchConfig,
    ) -> list[SearchResult]:
        """Time-weighted search prioritizing recent memories."""
        # Start with vector search
        results = await self._vector_search(query, config)

        # Apply strong temporal weighting
        now = datetime.now()
        for result in results:
            if result.memory.timestamp:
                days_old = (now - result.memory.timestamp).days
                # Exponential decay
                temporal_factor = math.exp(-config.temporal_decay * days_old)
                result.score = result.score * (0.5 + 0.5 * temporal_factor)

        return results

    def _adjust_scores(
        self,
        results: list[SearchResult],
        config: MemorySearchConfig,
    ) -> list[SearchResult]:
        """Adjust scores based on importance and other factors."""
        for result in results:
            # Apply importance weight
            importance_boost = result.memory.importance * config.importance_weight
            result.score = result.score * (1.0 - config.importance_weight) + importance_boost

        return results

    def _apply_diversity(
        self,
        results: list[SearchResult],
        threshold: float,
    ) -> list[SearchResult]:
        """Apply maximal marginal relevance (MMR) for diversity."""
        if not results or threshold <= 0:
            return results

        selected = []
        remaining = results.copy()

        while remaining:
            # Select highest scored result
            best = max(remaining, key=lambda x: x.score)
            selected.append(best)
            remaining.remove(best)

            # Remove similar results
            if remaining:
                for other in remaining[:]:
                    # Simple similarity check based on content overlap
                    if self._are_similar(best.memory.content, other.memory.content, threshold):
                        remaining.remove(other)

        return selected

    def _are_similar(self, text1: str, text2: str, threshold: float) -> bool:
        """Check if two texts are similar (simple word overlap)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= threshold

    def _apply_filters(
        self,
        results: list[SearchResult],
        config: MemorySearchConfig,
    ) -> list[SearchResult]:
        """Apply metadata filters to results."""
        filtered = []

        for result in results:
            # Filter by source
            if config.allowed_sources and result.memory.source not in config.allowed_sources:
                continue

            # Filter by importance
            if result.memory.importance < config.min_importance:
                continue

            # Filter by tags
            if config.allowed_tags:
                if not any(tag in result.memory.tags for tag in config.allowed_tags):
                    continue

            # Filter out excluded tags
            if config.excluded_tags:
                if any(tag in result.memory.tags for tag in config.excluded_tags):
                    continue

            filtered.append(result)

        return filtered

    def _apply_token_limit(
        self,
        results: list[SearchResult],
        max_tokens: int,
    ) -> tuple[list[SearchResult], int]:
        """Limit results based on estimated token count."""
        total_tokens = 0
        limited_results = []

        for result in results:
            # Estimate tokens for this result
            tokens = len(result.memory.content) // 4  # Rough estimate

            if total_tokens + tokens <= max_tokens:
                limited_results.append(result)
                total_tokens += tokens
            else:
                break

        return limited_results, total_tokens

    async def index_memory(self, memory: MemoryEntry) -> bool:
        """Index a memory for keyword search.

        Args:
            memory: Memory entry to index

        Returns:
            True if successful, False otherwise
        """
        if not WHOOSH_AVAILABLE:
            return False

        try:
            index = self._get_index()

            writer = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: index.writer()
            )

            writer = await writer

            writer.add_document(
                memory_id=memory.id,
                content=memory.content,
                source=memory.source.value,
                timestamp=memory.timestamp,
                importance=memory.importance,
                tags=",".join(memory.tags),
            )

            await asyncio.get_event_loop().run_in_executor(
                None,
                writer.commit
            )

            logger.debug(f"Indexed memory {memory.id} for keyword search")
            return True

        except Exception as e:
            logger.error(f"Failed to index memory {memory.id}: {e}")
            return False

    async def close(self) -> None:
        """Close the search engine and release resources."""
        await self.vector_store.close()
        logger.info("Search engine closed")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4  # Rough estimate
