"""Core types and data structures for the enhanced memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum

import numpy as np


class MemorySource(str, Enum):
    """Types of memory sources."""

    CONVERSATION = "conversation"
    DOCUMENT = "document"
    CODE = "code"
    TOOL_RESULT = "tool_result"
    USER_INPUT = "user_input"
    SYSTEM = "system"


class SearchMode(str, Enum):
    """Search modes for memory retrieval."""

    VECTOR = "vector"  # Pure semantic search
    KEYWORD = "keyword"  # Full-text search
    HYBRID = "hybrid"  # Combined vector + keyword
    TEMPORAL = "temporal"  # Time-weighted recent memories


@dataclass
class MemoryEntry:
    """A single memory entry with vector embedding."""

    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Core metadata fields
    timestamp: datetime = field(default_factory=datetime.now)
    source: MemorySource = MemorySource.CONVERSATION
    session_id: str = ""
    importance: float = 0.5  # 0-1 score
    tags: list[str] = field(default_factory=list)

    # Access tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    # Chunking info
    chunk_index: int = 0
    total_chunks: int = 1
    parent_id: Optional[str] = None  # For chunks of same document

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "session_id": self.session_id,
            "importance": self.importance,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dictionary storage."""
        embedding_data = data.get("embedding")
        embedding = np.array(embedding_data) if embedding_data is not None else None

        return cls(
            id=data["id"],
            content=data["content"],
            embedding=embedding,
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=MemorySource(data.get("source", "conversation")),
            session_id=data.get("session_id", ""),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 1),
            parent_id=data.get("parent_id"),
        )

    def record_access(self) -> None:
        """Record an access to this memory."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class SearchResult:
    """A single search result with relevance score."""

    memory: MemoryEntry
    score: float  # 0-1 relevance score
    distance: Optional[float] = None  # Vector distance (if applicable)
    rank: int = 0

    def __post_init__(self):
        """Ensure score is normalized."""
        self.score = max(0.0, min(1.0, self.score))


@dataclass
class MemorySearchConfig:
    """Configuration for memory search."""

    mode: SearchMode = SearchMode.HYBRID
    max_results: int = 10
    similarity_threshold: float = 0.7
    diversity_threshold: float = 0.3  # For diverse sampling
    temporal_decay: float = 0.1  # Weight for recent memories
    importance_weight: float = 0.2  # Weight for importance score

    # Vector search settings
    vector_weight: float = 0.7  # Weight in hybrid search
    keyword_weight: float = 0.3  # Weight in hybrid search

    # Result filtering
    min_importance: float = 0.0
    allowed_sources: Optional[list[MemorySource]] = None
    allowed_tags: Optional[list[str]] = None
    excluded_tags: Optional[list[str]] = None

    # Context awareness
    max_tokens: int = 2000  # Max tokens to return
    estimate_tokens: bool = True


@dataclass
class MemorySearchResult:
    """Complete search result with metadata."""

    query: str
    results: list[SearchResult]
    mode: SearchMode
    total_results: int
    search_time_ms: float
    used_vector_search: bool = False
    used_keyword_search: bool = False
    filtered_by_tokens: bool = False
    estimated_tokens: int = 0

    def get_top_results(self, n: int = 5) -> list[SearchResult]:
        """Get top N results."""
        return self.results[:n]

    def get_contents(self) -> list[str]:
        """Get just the content of results."""
        return [result.memory.content for result in self.results]

    def get_context_string(self, max_tokens: Optional[int] = None) -> str:
        """Get formatted context string for prompt injection."""
        if not self.results:
            return ""

        lines = ["## Relevant Memory"]
        for i, result in enumerate(self.results[:5], 1):
            metadata_str = f" [{result.memory.source.value}]"
            lines.append(f"{i}. {result.memory.content}{metadata_str}")

        return "\n".join(lines)
