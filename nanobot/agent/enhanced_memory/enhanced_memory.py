"""Enhanced memory system integrating vector search with existing nanobot memory."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from nanobot.agent.enhanced_memory.types import (
    MemoryEntry,
    MemorySearchConfig,
    MemorySearchResult,
    MemorySource,
    SearchMode,
)
from nanobot.agent.enhanced_memory.embeddings import create_embedding_provider
from nanobot.agent.enhanced_memory.vector_store import VectorStore
from nanobot.agent.enhanced_memory.search import MemorySearchEngine
from nanobot.agent.enhanced_memory.chunkers import get_chunker
from nanobot.agent.memory import MemoryStore, MemoryConsolidator


class EnhancedMemorySystem:
    """Enhanced memory system with vector search capabilities."""

    def __init__(
        self,
        workspace: Path,
        embedding_provider_type: str = "openai",
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        enable_vector_search: bool = True,
        enable_keyword_search: bool = True,
    ):
        self.workspace = workspace
        self.embedding_provider_type = embedding_provider_type
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.enable_vector_search = enable_vector_search
        self.enable_keyword_search = enable_keyword_search

        # Initialize components
        self.embedding_provider = None
        self.vector_store = None
        self.search_engine = None
        self.memory_store = MemoryStore(workspace)

    async def initialize(self) -> None:
        """Initialize the enhanced memory system."""
        try:
            # Initialize embedding provider
            if self.enable_vector_search:
                self.embedding_provider = await create_embedding_provider(
                    provider_type=self.embedding_provider_type,
                    api_key=self.api_key,
                    model=self.embedding_model,
                )

                if not await self.embedding_provider.is_available():
                    logger.warning(f"Embedding provider {self.embedding_provider_type} not available")
                    self.enable_vector_search = False
                else:
                    logger.info(f"Initialized embedding provider: {self.embedding_provider_type}")

            # Initialize vector store
            if self.enable_vector_search and self.embedding_provider:
                embedding_dim = self.embedding_provider.dimension
                self.vector_store = VectorStore(
                    workspace=self.workspace,
                    embedding_dimension=embedding_dim,
                )
                await self.vector_store.initialize()

            # Initialize search engine
            if self.enable_vector_search or self.enable_keyword_search:
                self.search_engine = MemorySearchEngine(
                    workspace=self.workspace,
                    vector_store=self.vector_store,
                    embedding_provider=self.embedding_provider,
                )
                await self.search_engine.initialize()

            logger.info("Enhanced memory system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced memory system: {e}")
            raise

    async def search_memory(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        max_results: int = 10,
        **kwargs: Any,
    ) -> MemorySearchResult:
        """Search memory using enhanced capabilities.

        Args:
            query: Search query
            mode: Search mode (vector, keyword, hybrid, temporal)
            max_results: Maximum number of results
            **kwargs: Additional search configuration

        Returns:
            Search results with metadata
        """
        if not self.search_engine:
            logger.warning("Search engine not initialized, returning empty results")
            return MemorySearchResult(
                query=query,
                results=[],
                mode=mode,
                total_results=0,
                search_time_ms=0.0,
            )

        config = MemorySearchConfig(
            mode=mode,
            max_results=max_results,
            **kwargs,
        )

        return await self.search_engine.search(query, config)

    async def add_memory(
        self,
        content: str,
        source: MemorySource = MemorySource.CONVERSATION,
        session_id: str = "",
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[MemoryEntry]:
        """Add a memory entry to the enhanced system.

        Args:
            content: Memory content
            source: Source type
            session_id: Session identifier
            importance: Importance score (0-1)
            tags: Optional tags
            metadata: Additional metadata

        Returns:
            Created memory entry or None if failed
        """
        try:
            memory_id = str(uuid.uuid4())

            # Generate embedding if vector search is enabled
            embedding = None
            if self.enable_vector_search and self.embedding_provider:
                embedding = await self.embedding_provider.embed_with_cache(content)

            # Create memory entry
            memory = MemoryEntry(
                id=memory_id,
                content=content,
                embedding=embedding,
                timestamp=datetime.now(),
                source=source,
                session_id=session_id,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {},
            )

            # Store in vector database
            if self.vector_store and embedding is not None:
                await self.vector_store.add_memory(memory)

            # Index for keyword search
            if self.search_engine:
                await self.search_engine.index_memory(memory)

            logger.debug(f"Added memory {memory_id} to enhanced system")
            return memory

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return None

    async def consolidate_messages_with_embeddings(
        self,
        messages: list[dict[str, Any]],
        provider: Any,  # LLMProvider
        model: str,
        chunk_large_messages: bool = True,
        chunk_size: int = 512,
    ) -> bool:
        """Consolidate messages with embedding generation.

        This extends the existing MemoryStore consolidation with vector indexing.

        Args:
            messages: Messages to consolidate
            provider: LLM provider for summarization
            model: Model to use
            chunk_large_messages: Whether to chunk large messages
            chunk_size: Size for chunking

        Returns:
            True if successful, False otherwise
        """
        try:
            # First, use existing consolidation logic
            success = await self.memory_store.consolidate(messages, provider, model)

            if not success:
                logger.warning("Standard consolidation failed, skipping vector indexing")
                return False

            # Then, index messages in vector database
            for message in messages:
                content = message.get("content", "")
                if not content:
                    continue

                # Determine if chunking is needed
                if chunk_large_messages and len(content) > chunk_size * 4:
                    await self._chunk_and_index_message(message, chunk_size)
                else:
                    await self._index_single_message(message)

            logger.info(f"Consolidated and indexed {len(messages)} messages")
            return True

        except Exception as e:
            logger.error(f"Enhanced consolidation failed: {e}")
            return False

    async def _chunk_and_index_message(
        self,
        message: dict[str, Any],
        chunk_size: int,
    ) -> None:
        """Chunk and index a large message."""
        content = message.get("content", "")
        if not content:
            return

        # Use semantic chunker
        chunker = get_chunker("semantic", chunk_size=chunk_size, overlap=50)
        chunks = chunker.chunk(content)

        parent_id = str(uuid.uuid4())

        for chunk in chunks:
            await self.add_memory(
                content=chunk.content,
                source=MemorySource.CONVERSATION,
                session_id=message.get("session_id", ""),
                importance=0.5,
                tags=["conversation", "chunked"],
                metadata={
                    "role": message.get("role", ""),
                    "chunk_index": chunk.index,
                    "total_chunks": chunk.total,
                    "parent_id": parent_id,
                },
            )

    async def _index_single_message(self, message: dict[str, Any]) -> None:
        """Index a single message without chunking."""
        content = message.get("content", "")
        if not content:
            return

        await self.add_memory(
            content=content,
            source=MemorySource.CONVERSATION,
            session_id=message.get("session_id", ""),
            importance=0.5,
            tags=["conversation"],
            metadata={
                "role": message.get("role", ""),
            },
        )

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 2000,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> str:
        """Get relevant memory context for prompt injection.

        Args:
            query: Query to search for
            max_tokens: Maximum tokens to include
            mode: Search mode

        Returns:
            Formatted context string
        """
        try:
            result = await self.search_memory(
                query=query,
                mode=mode,
                max_tokens=max_tokens,
                estimate_tokens=True,
            )

            if result.results:
                return result.get_context_string(max_tokens)
            else:
                return ""

        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return ""

    async def close(self) -> None:
        """Close the enhanced memory system."""
        if self.search_engine:
            await self.search_engine.close()
        if self.vector_store:
            await self.vector_store.close()
        logger.info("Enhanced memory system closed")

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the memory system.

        Returns:
            Dictionary with system statistics
        """
        stats = {
            "enabled": {
                "vector_search": self.enable_vector_search,
                "keyword_search": self.enable_keyword_search,
            },
        }

        if self.vector_store:
            stats["vector_store"] = await self.vector_store.get_stats()

        if self.embedding_provider:
            stats["embedding_provider"] = {
                "type": self.embedding_provider_type,
                "model": self.embedding_provider.model,
                "dimension": self.embedding_provider.dimension,
            }

        return stats


class EnhancedMemoryConsolidator(MemoryConsolidator):
    """Enhanced memory consolidator with vector indexing."""

    def __init__(
        self,
        workspace: Path,
        provider: Any,  # LLMProvider
        model: str,
        sessions: Any,  # SessionManager
        context_window_tokens: int,
        build_messages: Any,
        get_tool_definitions: Any,
        enable_vector_indexing: bool = True,
        embedding_provider_type: str = "openai",
        api_key: Optional[str] = None,
    ):
        super().__init__(
            workspace=workspace,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
        )

        self.enable_vector_indexing = enable_vector_indexing
        self.enhanced_system = EnhancedMemorySystem(
            workspace=workspace,
            embedding_provider_type=embedding_provider_type,
            api_key=api_key,
        )

    async def initialize(self) -> None:
        """Initialize the enhanced consolidator."""
        await self.enhanced_system.initialize()

    async def consolidate_messages(self, messages: list[dict[str, Any]]) -> bool:
        """Consolidate messages with vector indexing."""
        # First do standard consolidation
        success = await super().consolidate_messages(messages)

        # Then index in vector database
        if success and self.enable_vector_indexing:
            try:
                await self.enhanced_system.consolidate_messages_with_embeddings(
                    messages=messages,
                    provider=self.provider,
                    model=self.model,
                )
            except Exception as e:
                logger.error(f"Vector indexing failed: {e}")
                # Don't fail consolidation if vector indexing fails

        return success

    async def search_memory(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        max_results: int = 10,
    ) -> MemorySearchResult:
        """Search memory with enhanced capabilities."""
        return await self.enhanced_system.search_memory(
            query=query,
            mode=mode,
            max_results=max_results,
        )

    async def close(self) -> None:
        """Close the enhanced consolidator."""
        await self.enhanced_system.close()
