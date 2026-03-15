"""Vector database for storing and searching memory embeddings."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available, vector search will be limited")

from nanobot.agent.memory.types import MemoryEntry, MemorySource, SearchResult


class VectorStore:
    """Vector database for memory storage and retrieval."""

    def __init__(
        self,
        workspace: Path,
        collection_name: str = "nanobot_memory",
        embedding_dimension: int = 1536,
    ):
        self.workspace = workspace
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._lock = asyncio.Lock()

    def _get_db_path(self) -> Path:
        """Get path for ChromaDB storage."""
        db_dir = self.workspace / "memory" / "vector_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir

    def _get_client(self) -> Any:
        """Get or create ChromaDB client."""
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("ChromaDB is not available")

        if self._client is None:
            db_path = self._get_db_path()
            self._client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            logger.info(f"ChromaDB client initialized at {db_path}")

        return self._client

    def _get_collection(self) -> Any:
        """Get or create ChromaDB collection."""
        client = self._get_client()

        try:
            # Try to get existing collection
            self._collection = client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection
            self._collection = client.create_collection(
                name=self.collection_name,
                metadata={"dimension": self.embedding_dimension}
            )
            logger.info(f"Created new collection: {self.collection_name}")

        return self._collection

    async def initialize(self) -> None:
        """Initialize the vector store."""
        async with self._lock:
            try:
                self._get_collection()
                logger.info("Vector store initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise

    async def close(self) -> None:
        """Close the vector store and release resources."""
        async with self._lock:
            if self._client is not None:
                try:
                    self._client = None
                    self._collection = None
                    logger.info("Vector store closed")
                except Exception as e:
                    logger.error(f"Error closing vector store: {e}")

    async def add_memory(self, memory: MemoryEntry) -> bool:
        """Add a memory entry to the vector store.

        Args:
            memory: Memory entry to add

        Returns:
            True if successful, False otherwise
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, skipping vector storage")
            return False

        if memory.embedding is None:
            logger.warning(f"Memory {memory.id} has no embedding, skipping")
            return False

        async with self._lock:
            try:
                collection = self._get_collection()

                # Prepare metadata for ChromaDB
                metadata = {
                    "timestamp": memory.timestamp.isoformat(),
                    "source": memory.source.value,
                    "session_id": memory.session_id,
                    "importance": memory.importance,
                    "chunk_index": memory.chunk_index,
                    "total_chunks": memory.total_chunks,
                    "parent_id": memory.parent_id or "",
                    "tags": ",".join(memory.tags),
                }

                # Add to collection
                collection.add(
                    ids=[memory.id],
                    embeddings=[memory.embedding.tolist()],
                    documents=[memory.content],
                    metadatas=[metadata],
                )

                logger.debug(f"Added memory {memory.id} to vector store")
                return True

            except Exception as e:
                logger.error(f"Failed to add memory {memory.id}: {e}")
                return False

    async def add_memories_batch(self, memories: list[MemoryEntry]) -> int:
        """Add multiple memories in batch.

        Args:
            memories: List of memories to add

        Returns:
            Number of successfully added memories
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, skipping batch add")
            return 0

        # Filter memories with embeddings
        valid_memories = [m for m in memories if m.embedding is not None]
        if not valid_memories:
            logger.warning("No memories with embeddings to add")
            return 0

        async with self._lock:
            try:
                collection = self._get_collection()

                ids = []
                embeddings = []
                documents = []
                metadatas = []

                for memory in valid_memories:
                    ids.append(memory.id)
                    embeddings.append(memory.embedding.tolist())
                    documents.append(memory.content)
                    metadatas.append({
                        "timestamp": memory.timestamp.isoformat(),
                        "source": memory.source.value,
                        "session_id": memory.session_id,
                        "importance": memory.importance,
                        "chunk_index": memory.chunk_index,
                        "total_chunks": memory.total_chunks,
                        "parent_id": memory.parent_id or "",
                        "tags": ",".join(memory.tags),
                    })

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

                logger.info(f"Added {len(valid_memories)} memories to vector store")
                return len(valid_memories)

            except Exception as e:
                logger.error(f"Failed to batch add memories: {e}")
                return 0

    async def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar memories by vector embedding.

        Args:
            query_embedding: Query vector embedding
            n_results: Maximum number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results with scores
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, returning empty results")
            return []

        async with self._lock:
            try:
                collection = self._get_collection()

                # Prepare where clause for filtering
                where_clause = None
                if filter_metadata:
                    where_clause = {}
                    for key, value in filter_metadata.items():
                        if key in filter_metadata:
                            where_clause[key] = value

                # Search
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    where=where_clause,
                )

                # Convert to SearchResult objects
                search_results = []
                if results["ids"] and results["ids"][0]:
                    for i, memory_id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else None
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        content = results["documents"][0][i] if results["documents"] else ""

                        # Convert distance to similarity score (ChromaDB uses L2 distance)
                        score = 1.0 / (1.0 + distance) if distance is not None else 0.0

                        # Create MemoryEntry
                        memory = MemoryEntry(
                            id=memory_id,
                            content=content,
                            metadata=metadata,
                            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                            source=MemorySource(metadata.get("source", "conversation")),
                            session_id=metadata.get("session_id", ""),
                            importance=metadata.get("importance", 0.5),
                            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                            chunk_index=metadata.get("chunk_index", 0),
                            total_chunks=metadata.get("total_chunks", 1),
                            parent_id=metadata.get("parent_id") or None,
                        )

                        search_results.append(
                            SearchResult(
                                memory=memory,
                                score=score,
                                distance=distance,
                                rank=i,
                            )
                        )

                logger.debug(f"Vector search returned {len(search_results)} results")
                return search_results

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the vector store.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if successful, False otherwise
        """
        if not CHROMADB_AVAILABLE:
            return False

        async with self._lock:
            try:
                collection = self._get_collection()
                collection.delete(ids=[memory_id])
                logger.debug(f"Deleted memory {memory_id} from vector store")
                return True
            except Exception as e:
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False

    async def update_memory(self, memory: MemoryEntry) -> bool:
        """Update an existing memory in the vector store.

        Args:
            memory: Memory entry with updated data

        Returns:
            True if successful, False otherwise
        """
        if not CHROMADB_AVAILABLE:
            return False

        # ChromaDB handles updates by deleting and re-adding
        await self.delete_memory(memory.id)
        return await self.add_memory(memory)

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        if not CHROMADB_AVAILABLE:
            return {"available": False}

        async with self._lock:
            try:
                collection = self._get_collection()
                count = collection.count()

                return {
                    "available": True,
                    "collection_name": self.collection_name,
                    "total_memories": count,
                    "embedding_dimension": self.embedding_dimension,
                    "db_path": str(self._get_db_path()),
                }
            except Exception as e:
                logger.error(f"Failed to get vector store stats: {e}")
                return {"available": False, "error": str(e)}

    async def clear(self) -> bool:
        """Clear all memories from the vector store.

        Returns:
            True if successful, False otherwise
        """
        if not CHROMADB_AVAILABLE:
            return False

        async with self._lock:
            try:
                client = self._get_client()
                client.delete_collection(name=self.collection_name)
                self._collection = None
                logger.info(f"Cleared vector store collection: {self.collection_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear vector store: {e}")
                return False
