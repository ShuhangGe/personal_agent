"""Embedding providers for vector generation."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Optional
from pathlib import Path

import numpy as np
from loguru import logger


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    def __init__(
        self,
        model: str,
        dimension: int,
        cache_size: int = 1000,
    ):
        self.model = model
        self.dimension = dimension
        self.cache_size = cache_size
        self._cache: dict[str, np.ndarray] = {}

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"{self.model}:{text}"

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        return self._cache.get(self._get_cache_key(text))

    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Simple FIFO eviction (could be improved to LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[self._get_cache_key(text)] = embedding

    async def embed_with_cache(self, text: str) -> np.ndarray:
        """Generate embedding with caching."""
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        embedding = await self.embed(text)
        self._save_to_cache(text, embedding)
        return embedding

    async def embed_batch_with_cache(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts with caching."""
        embeddings = []
        texts_to_embed = []

        # Check cache for each text
        for text in texts:
            cached = self._get_from_cache(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                texts_to_embed.append(text)

        # Embed texts not in cache
        if texts_to_embed:
            new_embeddings = await self.embed_batch(texts_to_embed)
            new_idx = 0
            for i, embedding in enumerate(embeddings):
                if embedding is None:
                    embedding = new_embeddings[new_idx]
                    embeddings[i] = embedding
                    self._save_to_cache(texts_to_embed[new_idx], embedding)
                    new_idx += 1

        return embeddings  # type: ignore

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        cache_size: int = 1000,
    ):
        super().__init__(model, dimension, cache_size)
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("OpenAI package not installed")
                raise
        return self._client

    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            await self.embed("test")
            return True
        except Exception:
            return False

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
            )
            embeddings = [
                np.array(item.embedding, dtype=np.float32)
                for item in response.data
            ]
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        cache_size: int = 1000,
        device: str = "cpu",
    ):
        super().__init__(model, dimension, cache_size)
        self.device = device
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _load_model(self) -> Any:
        """Load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model, device=self.device)
                logger.info(f"Loaded local embedding model: {self.model}")
            except ImportError:
                logger.error("sentence-transformers package not installed")
                raise
        return self._model

    async def is_available(self) -> bool:
        """Check if local model is available."""
        try:
            self._load_model()
            return True
        except Exception:
            return False

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding using local model."""
        async with self._lock:
            model = self._load_model()
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: model.encode(text, convert_to_numpy=True)
            )
            return embedding.astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        async with self._lock:
            model = self._load_model()
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(texts, convert_to_numpy=True)
            )
            return [emb.astype(np.float32) for emb in embeddings]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider."""

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        dimension: int = 1024,
        cache_size: int = 1000,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(model, dimension, cache_size)
        self.base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create Ollama client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
            except ImportError:
                logger.error("httpx package not installed")
                raise
        return self._client

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama API."""
        client = self._get_client()
        try:
            response = await client.post(
                "/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["embedding"], dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't support batch embeddings, so do them sequentially
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings


async def create_embedding_provider(
    provider_type: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Create an embedding provider based on type.

    Args:
        provider_type: Type of provider ("openai", "local", "ollama")
        api_key: API key for cloud providers
        model: Model name to use
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedding provider

    Raises:
        ValueError: If provider type is unknown
    """
    if provider_type == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires api_key")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            dimension=1536,
            **kwargs,
        )

    elif provider_type == "local":
        return LocalEmbeddingProvider(
            model=model or "all-MiniLM-L6-v2",
            dimension=384,
            **kwargs,
        )

    elif provider_type == "ollama":
        return OllamaEmbeddingProvider(
            model=model or "mxbai-embed-large",
            dimension=1024,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")


@lru_cache(maxsize=128)
def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)."""
    # Simple estimation: ~4 characters per token
    return len(text) // 4
