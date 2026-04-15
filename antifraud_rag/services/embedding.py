import logging
from typing import List, Optional

import httpx

from antifraud_rag.core.config import Settings
from antifraud_rag.core.constants import EMBEDDING_TIMEOUT
from antifraud_rag.core.exceptions import EmbeddingError
from antifraud_rag.services.cache import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Calls the remote Embedding API and caches results to avoid redundant
    round-trips for identical texts.

    Args:
        settings: Application settings (URL, API key, model name, dimension).
        cache: Optional :class:`EmbeddingCache` instance.  Pass ``None`` to
            disable caching entirely (useful in tests or one-off scripts).
    """

    def __init__(self, settings: Settings, cache: Optional[EmbeddingCache] = None) -> None:
        self.url = settings.EMBEDDING_MODEL_URL
        self.api_key = settings.EMBEDDING_MODEL_API_KEY
        self.model = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._cache = cache

    async def get_embeddings(self, text: str) -> List[float]:
        """
        Return the embedding vector for *text*.

        If a cache is configured and a valid entry exists, the cached
        vector is returned immediately without hitting the API.  On a
        cache miss the API is called and the result is stored for future
        requests.
        """
        if self._cache is not None:
            cache_key = EmbeddingCache.make_key(self.model, text)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

        embedding = await self._fetch_from_api(text)

        if self._cache is not None:
            await self._cache.set(cache_key, embedding)  # type: ignore[possibly-undefined]

        return embedding

    async def _fetch_from_api(self, text: str) -> List[float]:
        """Make the actual HTTP request to the embedding endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"input": text, "model": self.model},
                    timeout=EMBEDDING_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["data"][0]["embedding"]

                if len(embedding) != self.dimension:
                    raise EmbeddingError(
                        f"Embedding dimension mismatch: expected {self.dimension}, "
                        f"got {len(embedding)}"
                    )

                return embedding
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Embedding API error: {str(e)}")
