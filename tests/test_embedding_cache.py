"""
Unit tests for antifraud_rag/services/cache.py - EmbeddingCache.

Covers:
- make_key uniqueness and stability
- cache miss → None returned
- cache hit → value returned, LRU order updated
- LRU eviction when max_size exceeded
- TTL expiry treated as a miss
- TTL = 0 means never expire
- clear() empties the store
- stats counters (hits, misses, evictions, hit_rate)
- concurrent async access (asyncio.Lock)
- constructor validation
"""

import asyncio
import time

import pytest

from antifraud_rag.services.cache import CacheStats, EmbeddingCache


class TestMakeKey:
    def test_same_model_and_text_produces_same_key(self):
        k1 = EmbeddingCache.make_key("model-a", "hello world")
        k2 = EmbeddingCache.make_key("model-a", "hello world")
        assert k1 == k2

    def test_different_texts_produce_different_keys(self):
        k1 = EmbeddingCache.make_key("model-a", "foo")
        k2 = EmbeddingCache.make_key("model-a", "bar")
        assert k1 != k2

    def test_different_models_produce_different_keys(self):
        k1 = EmbeddingCache.make_key("model-a", "hello")
        k2 = EmbeddingCache.make_key("model-b", "hello")
        assert k1 != k2

    def test_key_is_64_hex_characters(self):
        key = EmbeddingCache.make_key("ada-002", "test input")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_empty_text_produces_stable_key(self):
        k1 = EmbeddingCache.make_key("model-a", "")
        k2 = EmbeddingCache.make_key("model-a", "")
        assert k1 == k2

    def test_unicode_text_handled(self):
        k1 = EmbeddingCache.make_key("model-a", "这是一段中文")
        k2 = EmbeddingCache.make_key("model-a", "这是一段中文")
        assert k1 == k2


class TestCacheMissAndHit:
    @pytest.mark.asyncio
    async def test_miss_on_empty_cache(self):
        cache = EmbeddingCache()
        key = EmbeddingCache.make_key("model", "text")
        assert await cache.get(key) is None

    @pytest.mark.asyncio
    async def test_hit_after_set(self):
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3]
        key = EmbeddingCache.make_key("model", "text")

        await cache.set(key, embedding)
        result = await cache.get(key)

        assert result == embedding

    @pytest.mark.asyncio
    async def test_hit_returns_identical_list(self):
        cache = EmbeddingCache()
        embedding = [float(i) for i in range(1536)]
        key = EmbeddingCache.make_key("ada-002", "fraud case")

        await cache.set(key, embedding)
        result = await cache.get(key)

        assert result == embedding

    @pytest.mark.asyncio
    async def test_different_keys_independent(self):
        cache = EmbeddingCache()
        vec_a = [1.0] * 3
        vec_b = [2.0] * 3
        key_a = EmbeddingCache.make_key("model", "text-a")
        key_b = EmbeddingCache.make_key("model", "text-b")

        await cache.set(key_a, vec_a)
        await cache.set(key_b, vec_b)

        assert await cache.get(key_a) == vec_a
        assert await cache.get(key_b) == vec_b

    @pytest.mark.asyncio
    async def test_set_overwrites_existing_entry(self):
        cache = EmbeddingCache()
        key = EmbeddingCache.make_key("model", "text")
        old_vec = [0.1] * 3
        new_vec = [0.9] * 3

        await cache.set(key, old_vec)
        await cache.set(key, new_vec)
        result = await cache.get(key)

        assert result == new_vec


class TestLRUEviction:
    @pytest.mark.asyncio
    async def test_lru_evicts_oldest_entry_when_full(self):
        cache = EmbeddingCache(max_size=3, ttl_seconds=0)
        keys = [EmbeddingCache.make_key("m", str(i)) for i in range(4)]

        for i, key in enumerate(keys[:3]):
            await cache.set(key, [float(i)])

        # This should evict keys[0] (LRU)
        await cache.set(keys[3], [99.0])

        assert await cache.get(keys[0]) is None
        assert await cache.get(keys[1]) is not None
        assert await cache.get(keys[2]) is not None
        assert await cache.get(keys[3]) is not None

    @pytest.mark.asyncio
    async def test_access_promotes_entry_above_eviction(self):
        cache = EmbeddingCache(max_size=3, ttl_seconds=0)
        keys = [EmbeddingCache.make_key("m", str(i)) for i in range(4)]

        for i, key in enumerate(keys[:3]):
            await cache.set(key, [float(i)])

        # Access keys[0] → it becomes MRU; keys[1] becomes LRU
        await cache.get(keys[0])

        # Inserting a 4th entry should evict keys[1] now
        await cache.set(keys[3], [99.0])

        assert await cache.get(keys[0]) is not None, "MRU entry must survive"
        assert await cache.get(keys[1]) is None, "LRU entry must be evicted"
        assert await cache.get(keys[2]) is not None
        assert await cache.get(keys[3]) is not None

    @pytest.mark.asyncio
    async def test_size_never_exceeds_max_size(self):
        max_size = 5
        cache = EmbeddingCache(max_size=max_size, ttl_seconds=0)

        for i in range(20):
            key = EmbeddingCache.make_key("m", str(i))
            await cache.set(key, [float(i)])

        assert cache.stats.size <= max_size


class TestTTLExpiry:
    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self, monkeypatch):
        cache = EmbeddingCache(max_size=10, ttl_seconds=1)
        key = EmbeddingCache.make_key("model", "text")
        await cache.set(key, [0.5])

        # Simulate time advancing past TTL
        future = time.monotonic() + 2
        monkeypatch.setattr("antifraud_rag.services.cache.time.monotonic", lambda: future)

        result = await cache.get(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_expired_entry_still_accessible(self, monkeypatch):
        cache = EmbeddingCache(max_size=10, ttl_seconds=100)
        key = EmbeddingCache.make_key("model", "text")
        await cache.set(key, [0.5])

        # Advance time, but less than TTL
        slightly_later = time.monotonic() + 10
        monkeypatch.setattr("antifraud_rag.services.cache.time.monotonic", lambda: slightly_later)

        result = await cache.get(key)
        assert result == [0.5]

    @pytest.mark.asyncio
    async def test_ttl_zero_means_never_expire(self, monkeypatch):
        cache = EmbeddingCache(max_size=10, ttl_seconds=0)
        key = EmbeddingCache.make_key("model", "text")
        await cache.set(key, [0.7])

        far_future = time.monotonic() + 999_999
        monkeypatch.setattr("antifraud_rag.services.cache.time.monotonic", lambda: far_future)

        result = await cache.get(key)
        assert result == [0.7]

    @pytest.mark.asyncio
    async def test_expired_entry_is_removed_from_store(self, monkeypatch):
        cache = EmbeddingCache(max_size=10, ttl_seconds=1)
        key = EmbeddingCache.make_key("model", "text")
        await cache.set(key, [0.5])

        future = time.monotonic() + 2
        monkeypatch.setattr("antifraud_rag.services.cache.time.monotonic", lambda: future)

        await cache.get(key)  # triggers expiry cleanup
        assert cache.stats.size == 0


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_empties_cache(self):
        cache = EmbeddingCache()
        for i in range(5):
            await cache.set(EmbeddingCache.make_key("m", str(i)), [float(i)])

        await cache.clear()

        assert cache.stats.size == 0

    @pytest.mark.asyncio
    async def test_clear_allows_reuse(self):
        cache = EmbeddingCache()
        key = EmbeddingCache.make_key("model", "text")
        await cache.set(key, [1.0])

        await cache.clear()
        assert await cache.get(key) is None

        await cache.set(key, [2.0])
        assert await cache.get(key) == [2.0]


class TestStats:
    @pytest.mark.asyncio
    async def test_initial_stats_are_zero(self):
        cache = EmbeddingCache()
        s = cache.stats
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.size == 0

    @pytest.mark.asyncio
    async def test_miss_increments_miss_counter(self):
        cache = EmbeddingCache()
        await cache.get(EmbeddingCache.make_key("m", "unknown"))
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    @pytest.mark.asyncio
    async def test_hit_increments_hit_counter(self):
        cache = EmbeddingCache()
        key = EmbeddingCache.make_key("m", "text")
        await cache.set(key, [0.1])
        await cache.get(key)
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

    @pytest.mark.asyncio
    async def test_eviction_increments_eviction_counter(self):
        cache = EmbeddingCache(max_size=2, ttl_seconds=0)
        for i in range(3):
            await cache.set(EmbeddingCache.make_key("m", str(i)), [float(i)])
        assert cache.stats.evictions == 1

    @pytest.mark.asyncio
    async def test_size_tracks_number_of_entries(self):
        cache = EmbeddingCache()
        assert cache.stats.size == 0
        for i in range(5):
            await cache.set(EmbeddingCache.make_key("m", str(i)), [float(i)])
        assert cache.stats.size == 5

    @pytest.mark.asyncio
    async def test_hit_rate_with_mixed_lookups(self):
        cache = EmbeddingCache()
        hit_key = EmbeddingCache.make_key("m", "hit")
        await cache.set(hit_key, [0.1])

        await cache.get(hit_key)  # hit
        await cache.get(EmbeddingCache.make_key("m", "miss"))  # miss

        assert cache.stats.hit_rate == pytest.approx(0.5)

    def test_hit_rate_zero_when_no_lookups(self):
        assert EmbeddingCache().stats.hit_rate == 0.0

    def test_cache_stats_repr(self):
        stats = CacheStats(hits=3, misses=1, evictions=0, size=3)
        assert "hit_rate" in repr(stats)
        assert "75.0%" in repr(stats)


class TestConstructorValidation:
    def test_negative_max_size_raises(self):
        with pytest.raises(ValueError, match="max_size"):
            EmbeddingCache(max_size=0)

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds"):
            EmbeddingCache(ttl_seconds=-1)

    def test_max_size_one_is_valid(self):
        cache = EmbeddingCache(max_size=1)
        assert cache.max_size == 1

    def test_ttl_zero_is_valid(self):
        cache = EmbeddingCache(ttl_seconds=0)
        assert cache.ttl_seconds == 0


class TestConcurrentAccess:
    @pytest.mark.asyncio
    async def test_concurrent_sets_do_not_corrupt_cache(self):
        """Multiple concurrent writes should not raise or produce torn state."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=0)

        async def write(i: int) -> None:
            key = EmbeddingCache.make_key("m", str(i))
            await cache.set(key, [float(i)] * 3)

        await asyncio.gather(*[write(i) for i in range(50)])

        assert cache.stats.size == 50

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self):
        """Interleaved reads and writes should not deadlock or corrupt data."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=0)
        key = EmbeddingCache.make_key("m", "shared")
        await cache.set(key, [0.0])

        async def read() -> None:
            await cache.get(key)

        async def write(i: int) -> None:
            await cache.set(key, [float(i)])

        tasks = [read() if i % 2 == 0 else write(i) for i in range(20)]
        await asyncio.gather(*tasks)

        result = await cache.get(key)
        assert result is not None
