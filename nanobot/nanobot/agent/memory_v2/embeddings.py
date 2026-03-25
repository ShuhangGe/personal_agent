"""Embedding provider for semantic search."""

import os
from functools import lru_cache
from typing import Any

import numpy as np


class EmbeddingProvider:
    """
    Provider for generating text embeddings.

    Supports multiple backends:
    - Ollama (local, fast, GGUF models)
    - OpenAI API (requires API key)
    - sentence-transformers (local, free)
    - Custom endpoint
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen3-embedding:0.6b",
        dimension: int = 1024,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """
        Initialize embedding provider.

        Args:
            provider: Provider type (ollama, openai, local, custom)
            model: Model name
            dimension: Embedding dimension
            api_key: API key for OpenAI
            endpoint: Custom endpoint URL
        """
        self.provider = provider
        self.model = model
        self.dimension = dimension
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.endpoint = endpoint or "http://127.0.0.1:11434"

        self._model = None
        self._client = None

        # Initialize model
        if provider == "ollama":
            self._init_ollama_client()
        elif provider == "local":
            self._init_local_model()
        elif provider == "openai":
            self._init_openai_client()

    def _init_local_model(self) -> None:
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model)
            # Update dimension based on model
            self.dimension = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

    def _init_ollama_client(self) -> None:
        """Initialize Ollama client (uses HTTP API)."""
        try:
            import requests

            self._client = requests.Session()
            # Test connection
            response = self._client.get(f"{self.endpoint}/api/tags", timeout=5)
            response.raise_for_status()
        except ImportError:
            raise ImportError(
                "requests is required for Ollama embeddings. " "Install with: pip install requests"
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.endpoint}. "
                f"Make sure Ollama is running: ollama serve. Error: {e}"
            )

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            if not self.api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")

            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. " "Install with: pip install openai"
            )

    async def embed(self, text: str | list[str]) -> np.ndarray | list[np.ndarray]:
        """
        Generate embedding(s) for text(s).

        Args:
            text: Text or list of texts to embed

        Returns:
            Embedding vector or list of embedding vectors
        """
        if self.provider == "ollama":
            return self._embed_ollama(text)
        elif self.provider == "local":
            return self._embed_local(text)
        elif self.provider == "openai":
            return await self._embed_openai(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _embed_ollama(self, text: str | list[str]) -> np.ndarray | list[np.ndarray]:
        """Generate embeddings using Ollama API."""
        texts = [text] if isinstance(text, str) else text

        embeddings = []
        for t in texts:
            response = self._client.post(
                f"{self.endpoint}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["embedding"], dtype=np.float32)
            embeddings.append(embedding)

        return embeddings[0] if isinstance(text, str) else embeddings

    def _embed_local(self, text: str | list[str]) -> np.ndarray | list[np.ndarray]:
        """Generate embeddings using local model."""
        if self._model is None:
            self._init_local_model()

        if isinstance(text, str):
            return self._model.encode(text, convert_to_numpy=True)
        else:
            return self._model.encode(text, convert_to_numpy=True)

    async def _embed_openai(self, text: str | list[str]) -> np.ndarray | list[np.ndarray]:
        """Generate embeddings using OpenAI API."""
        if self._client is None:
            self._init_openai_client()

        texts = [text] if isinstance(text, str) else text

        response = self._client.embeddings.create(input=texts, model=self.model)

        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]

        if isinstance(text, str):
            return embeddings[0]
        return embeddings

    @lru_cache(maxsize=1000)
    def embed_sync(self, text: str) -> np.ndarray:
        """
        Synchronous embedding with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.provider == "ollama":
            return self._embed_ollama(text)
        elif self.provider == "local":
            return self._embed_local(text)
        else:
            # For OpenAI, we need to run async in sync context
            import asyncio

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._embed_openai(text))

    async def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.embed(batch)
            embeddings.extend(
                batch_embeddings if isinstance(batch_embeddings, list) else [batch_embeddings]
            )

        return embeddings
