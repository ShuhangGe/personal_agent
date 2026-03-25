"""Data models for the three-tier memory system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np


class MemoryTier(str, Enum):
    """Memory storage tiers."""

    WORKING = "working"
    SESSION = "session"
    LONGTERM = "longterm"


@dataclass
class Message:
    """A single message in working memory."""

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    role: str = "user"  # user, assistant, system, tool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """Extracted entity from conversation."""

    name: str
    type: str  # person, organization, location, concept, etc.
    mentions: int = 1
    confidence: float = 1.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Relationship between entities."""

    source: str  # Entity name
    target: str  # Entity name
    type: str  # works_for, knows, located_in, etc.
    confidence: float = 1.0
    first_seen: datetime = field(default_factory=datetime.now)


@dataclass
class WorkingMemory:
    """Short-term conversation context."""

    session_key: str
    messages: list[Message] = field(default_factory=list)
    context_window: int = 100  # Max messages to keep

    # Recent entities mentioned
    entities: dict[str, Entity] = field(default_factory=dict)

    # Current conversation state
    state: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

    def add_message(self, message: Message) -> None:
        """Add message and maintain context window."""
        self.messages.append(message)
        if len(self.messages) > self.context_window:
            self.messages.pop(0)
        self.last_accessed = datetime.now()

    def get_context(self, max_tokens: int = 4000) -> list[Message]:
        """Get recent messages (simplified - doesn't count tokens)."""
        return self.messages[-self.context_window :]

    def add_entity(self, entity: Entity) -> None:
        """Add or update entity in working memory."""
        if entity.name in self.entities:
            existing = self.entities[entity.name]
            existing.mentions += 1
            existing.last_seen = datetime.now()
        else:
            self.entities[entity.name] = entity


@dataclass
class SessionMemory:
    """Medium-term conversation memory with semantic indexing."""

    # Metadata
    id: str = field(default_factory=lambda: str(uuid4()))
    session_key: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Content
    summary: str = ""
    embedding: np.ndarray | None = None  # Vector representation

    # Extracted entities and relationships
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

    # Importance scoring
    importance_score: float = 0.0  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: datetime | None = None

    # Topic classification
    topics: list[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1.0 to 1.0


@dataclass
class LongTermMemory:
    """Persistent knowledge with semantic indexing."""

    # Metadata
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Content
    fact: str = ""  # Human-readable fact
    category: str = ""  # preference, relationship, project, etc.
    embedding: np.ndarray | None = None

    # Importance and access
    importance: float = 1.0  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: datetime | None = None

    # Validation
    confidence: float = 1.0  # How certain we are
    source: str = ""  # Where this came from

    # Expiration (optional)
    expires_at: datetime | None = None


@dataclass
class TimeRange:
    """Time range filter for searches."""

    start: datetime | None = None
    end: datetime | None = None


@dataclass
class SearchFilters:
    """Filters for memory search."""

    time_range: TimeRange | None = None
    categories: list[str] | None = None
    min_importance: float | None = None
    session_keys: list[str] | None = None


@dataclass
class MemoryResult:
    """Search result from any memory tier."""

    id: str
    content: str
    tier: MemoryTier
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    tier: MemoryTier | None = None
    total_memories: int = 0
    total_messages: int = 0
    retention_days: int | None = None
    storage_size_bytes: int = 0
