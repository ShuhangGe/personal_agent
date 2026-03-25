"""Configuration for the memory system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WorkingTierConfig:
    """Configuration for working memory tier."""

    max_messages: int = 100
    max_tokens: int = 4000
    timeout_minutes: int = 30


@dataclass
class SessionTierConfig:
    """Configuration for session memory tier."""

    retention_days: int = 30
    max_memories: int = 10000
    consolidation_interval: int = 10  # messages
    importance_threshold: float = 0.6


@dataclass
class LongTermTierConfig:
    """Configuration for long-term memory tier."""

    storage_dir: str = "memory"
    memory_file: str = "MEMORY.md"
    history_file: str = "HISTORY.md"
    auto_consolidate: bool = True


@dataclass
class TierConfig:
    """Configuration for all memory tiers."""

    working: WorkingTierConfig = None
    session: SessionTierConfig = None
    longterm: LongTermTierConfig = None

    def __post_init__(self):
        if self.working is None:
            self.working = WorkingTierConfig()
        if self.session is None:
            self.session = SessionTierConfig()
        if self.longterm is None:
            self.longterm = LongTermTierConfig()


@dataclass
class SearchConfig:
    """Configuration for memory search."""

    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    default_limit: int = 10


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: str = "local"  # openai, local, custom
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # 1536 for OpenAI, 384 for local
    batch_size: int = 10
    api_key: str | None = None
    endpoint: str | None = None  # For custom provider


@dataclass
class EntityConfig:
    """Configuration for entity extraction."""

    enabled: bool = True
    min_confidence: float = 0.7
    types: list[str] = None

    def __post_init__(self):
        if self.types is None:
            self.types = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]


@dataclass
class MemoryConfig:
    """Complete memory system configuration."""

    tiers: TierConfig = None
    search: SearchConfig = None
    embeddings: EmbeddingConfig = None
    entities: EntityConfig = None

    def __post_init__(self):
        if self.tiers is None:
            self.tiers = TierConfig()
        if self.search is None:
            self.search = SearchConfig()
        if self.embeddings is None:
            self.embeddings = EmbeddingConfig()
        if self.entities is None:
            self.entities = EntityConfig()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MemoryConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        tiers_data = config_dict.get("tiers", {})
        tiers = TierConfig(
            working=WorkingTierConfig(**tiers_data.get("working", {})),
            session=SessionTierConfig(**tiers_data.get("session", {})),
            longterm=LongTermTierConfig(**tiers_data.get("longterm", {})),
        )

        search = SearchConfig(**config_dict.get("search", {}))
        embeddings = EmbeddingConfig(**config_dict.get("embeddings", {}))
        entities = EntityConfig(**config_dict.get("entities", {}))

        return cls(tiers=tiers, search=search, embeddings=embeddings, entities=entities)

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "MemoryConfig":
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            return cls()  # Return default config

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        return cls.from_dict(config_dict.get("memory", {}))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "tiers": {
                "working": self.tiers.working.__dict__,
                "session": self.tiers.session.__dict__,
                "longterm": self.tiers.longterm.__dict__,
            },
            "search": self.search.__dict__,
            "embeddings": self.embeddings.__dict__,
            "entities": self.entities.__dict__,
        }
