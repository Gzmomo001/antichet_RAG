import logging
from typing import List, Optional, Union

import httpx

from antifraud_rag.core.config import Settings
from antifraud_rag.core.constants import EMBEDDING_TIMEOUT
from antifraud_rag.core.exceptions import EmbeddingError
from antifraud_rag.services.cache import CacheStats, EmbeddingCache

logger = logging.getLogger(__name__)

_UNSET = object()


class EmbeddingService:
    """
    Calls the remote Embedding API with **automatic in-process caching**.

    By default a new :class:`EmbeddingCache` is created from ``settings``
    (``EMBEDDING_CACHE_MAX_SIZE`` / ``EMBEDDING_CACHE_TTL_SECONDS``).
    You never need to configure caching manually — it just works.

    To **explicitly disable** caching, pass ``cache=None``.

    Args:
        settings: Application settings (URL, API key, model, dimension,
            cache size / TTL).
        cache: Controls caching behaviour.

            * *not provided* (default) — auto-create from ``settings``
            * ``EmbeddingCache(...)`` — use the supplied instance
            * ``None`` — disable caching entirely
    """

    def __init__(
        self,
        settings: Settings,
        cache: Union[EmbeddingCache, None, object] = _UNSET,
    ) -> None:
        self.url = settings.EMBEDDING_MODEL_URL
        self.api_key = settings.EMBEDDING_MODEL_API_KEY
        self.model = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

        if cache is _UNSET:
            self._cache: Optional[EmbeddingCache] = EmbeddingCache(
                max_size=settings.EMBEDDING_CACHE_MAX_SIZE,
                ttl_seconds=settings.EMBEDDING_CACHE_TTL_SECONDS,
            )
            logger.debug(
                "Embedding cache auto-created (max_size=%d, ttl=%ds)",
                settings.EMBEDDING_CACHE_MAX_SIZE,
                settings.EMBEDDING_CACHE_TTL_SECONDS,
            )
        else:
            self._cache = cache  # type: ignore[assignment]

    @property
    def cache_stats(self) -> Optional[CacheStats]:
        """Return cache performance counters, or ``None`` if caching is disabled."""
        return self._cache.stats if self._cache is not None else None

    async def get_embeddings(self, text: str) -> List[float]:
        """
        Return the embedding vector for *text*.

        If a cache is configured and a valid entry exists, the cached
        vector is returned immediately without hitting the API.  On a
        cache miss the API is called and the result is stored for future
        requests.
        """
        cache_key: Optional[str] = None
        if self._cache is not None:
            cache_key = EmbeddingCache.make_key(self.model, text)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

        embedding = await self._fetch_from_api(text)

        if self._cache is not None and cache_key is not None:
            await self._cache.set(cache_key, embedding)

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
                embedding: List[float] = data["data"][0]["embedding"]
                if len(embedding) != self.dimension:
                    raise EmbeddingError(
                        f"Embedding dimension mismatch: expected {self.dimension}, "
                        f"got {len(embedding)}"
                    )
                return embedding
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise EmbeddingError(f"Embedding API error: {str(e)}") from e
