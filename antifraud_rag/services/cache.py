"""
In-process LRU embedding cache.

Avoids repeated calls to the Embedding API for identical texts by storing
text → vector mappings in memory with optional TTL-based expiry.

Design notes:
- Thread-safe for async code via asyncio.Lock.
- LRU eviction (OrderedDict): least-recently-used entry is dropped when
  the cache reaches max_size.
- Cache key = SHA-256(model_name + ":" + text) – model-aware, so switching
  embedding models never serves stale vectors.
- TTL of 0 means entries never expire.
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    embedding: List[float]
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class CacheStats:
    """Snapshot of cache performance counters."""

    hits: int
    misses: int
    evictions: int
    size: int

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were served from cache (0.0–1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, size={self.size}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


class EmbeddingCache:
    """
    Async-safe LRU cache for embedding vectors.

    Usage::

        cache = EmbeddingCache(max_size=1000, ttl_seconds=86400)

        # Inside EmbeddingService (or any async context)
        key = EmbeddingCache.make_key(model_name, text)
        vector = await cache.get(key)
        if vector is None:
            vector = await call_embedding_api(text)
            await cache.set(key, vector)

    Args:
        max_size: Maximum number of entries. When the limit is reached the
            least-recently-used entry is evicted. Must be > 0.
        ttl_seconds: Seconds an entry lives before it is treated as a miss.
            Pass 0 (default-off) to disable expiry entirely.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 86400) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be a positive integer, got {max_size}")
        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {ttl_seconds}")

        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(model_name: str, text: str) -> str:
        """
        Compute a stable, model-aware cache key.

        Incorporating the model name means that changing embedding models
        (e.g. ada-002 → text-embedding-3-small) automatically invalidates
        all cached entries for the old model.
        """
        payload = f"{model_name}:{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def get(self, key: str) -> Optional[List[float]]:
        """
        Return the cached embedding for *key*, or ``None`` on miss/expiry.

        On a hit the entry is moved to the MRU position (LRU promotion).
        Expired entries are removed and counted as misses.
        """
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            if self._is_expired(entry):
                del self._store[key]
                self._misses += 1
                logger.debug("Embedding cache: expired entry evicted (key=…%s)", key[-8:])
                return None

            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Embedding cache hit (key=…%s)", key[-8:])
            return entry.embedding

    async def set(self, key: str, embedding: List[float]) -> None:
        """
        Store *embedding* under *key*.

        If *key* already exists, the entry is refreshed in-place and
        promoted to the MRU position.  If the store is full, the LRU
        entry is evicted before the new entry is inserted.
        """
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = _CacheEntry(embedding=embedding)
                return

            self._store[key] = _CacheEntry(embedding=embedding)

            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("Embedding cache: LRU eviction (key=…%s)", evicted_key[-8:])

    async def clear(self) -> None:
        """Remove all entries from the cache (counters are preserved)."""
        async with self._lock:
            self._store.clear()
        logger.debug("Embedding cache cleared")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CacheStats:
        """Return a point-in-time snapshot of cache performance counters."""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            size=len(self._store),
        )

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: _CacheEntry) -> bool:
        if self._ttl <= 0:
            return False
        return (time.monotonic() - entry.created_at) > self._ttl
