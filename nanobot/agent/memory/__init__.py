"""Enhanced memory system with vector database and semantic search."""

from nanobot.agent.memory.types import (
    MemoryEntry,
    MemorySearchConfig,
    MemorySearchResult,
    SearchResult,
    MemorySource,
    SearchMode,
)
from nanobot.agent.memory.embeddings import (
    EmbeddingProvider,
    create_embedding_provider,
    OpenAIEmbeddingProvider,
    LocalEmbeddingProvider,
    OllamaEmbeddingProvider,
)
from nanobot.agent.memory.vector_store import VectorStore
from nanobot.agent.memory.search import MemorySearchEngine
from nanobot.agent.memory.chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    CodeChunker,
    get_chunker,
    estimate_tokens,
)
from nanobot.agent.memory.enhanced_memory import (
    EnhancedMemorySystem,
    EnhancedMemoryConsolidator,
)

__all__ = [
    # Types
    "MemoryEntry",
    "MemorySearchConfig",
    "MemorySearchResult",
    "SearchResult",
    "MemorySource",
    "SearchMode",
    # Embeddings
    "EmbeddingProvider",
    "create_embedding_provider",
    "OpenAIEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OllamaEmbeddingProvider",
    # Vector Store
    "VectorStore",
    # Search
    "MemorySearchEngine",
    # Chunkers
    "FixedSizeChunker",
    "SemanticChunker",
    "CodeChunker",
    "get_chunker",
    "estimate_tokens",
    # Enhanced System
    "EnhancedMemorySystem",
    "EnhancedMemoryConsolidator",
]
