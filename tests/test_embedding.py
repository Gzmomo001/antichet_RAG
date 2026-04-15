"""
Unit tests for antifraud_rag/services/embedding.py - EmbeddingService.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Response

from antifraud_rag.core.constants import EMBEDDING_TIMEOUT
from antifraud_rag.core.exceptions import EmbeddingError
from antifraud_rag.services.cache import EmbeddingCache
from antifraud_rag.services.embedding import EmbeddingService


class TestEmbeddingService:
    """Tests for the EmbeddingService class."""

    def test_embedding_service_initialization_with_settings(self, mock_settings):
        """Test EmbeddingService initializes with provided settings."""
        service = EmbeddingService(settings=mock_settings)
        assert service.url == mock_settings.EMBEDDING_MODEL_URL
        assert service.api_key == mock_settings.EMBEDDING_MODEL_API_KEY
        assert service.model == mock_settings.EMBEDDING_MODEL_NAME

    @pytest.mark.asyncio
    async def test_get_embeddings_success(self, mock_settings):
        """Test successful embedding retrieval."""
        service = EmbeddingService(settings=mock_settings)
        embedding = [0.1] * mock_settings.EMBEDDING_DIMENSION

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_response = MagicMock(spec=Response)
            mock_response.json.return_value = {"data": [{"embedding": embedding, "index": 0}]}
            mock_response.raise_for_status = MagicMock()
            mock_client_instance.post.return_value = mock_response

            result = await service.get_embeddings("test text")

            assert result == embedding
            mock_client_instance.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embeddings_calls_correct_endpoint(self, mock_settings):
        """Test embedding request is made to correct endpoint."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_response = MagicMock(spec=Response)
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * mock_settings.EMBEDDING_DIMENSION, "index": 0}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_client_instance.post.return_value = mock_response

            test_text = "fraud detection test"
            await service.get_embeddings(test_text)

            call_args = mock_client_instance.post.call_args
            assert call_args[0][0] == mock_settings.EMBEDDING_MODEL_URL

    @pytest.mark.asyncio
    async def test_get_embeddings_includes_auth_header(self, mock_settings):
        """Test embedding request includes authorization header."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_response = MagicMock(spec=Response)
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * mock_settings.EMBEDDING_DIMENSION, "index": 0}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_client_instance.post.return_value = mock_response

            await service.get_embeddings("test")

            call_kwargs = mock_client_instance.post.call_args[1]
            assert "Bearer" in call_kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_get_embeddings_includes_model_in_payload(self, mock_settings):
        """Test embedding request includes model in JSON payload."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_response = MagicMock(spec=Response)
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * mock_settings.EMBEDDING_DIMENSION, "index": 0}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_client_instance.post.return_value = mock_response

            await service.get_embeddings("test")

            call_kwargs = mock_client_instance.post.call_args[1]
            assert call_kwargs["json"]["model"] == mock_settings.EMBEDDING_MODEL_NAME

    @pytest.mark.asyncio
    async def test_get_embeddings_timeout(self, mock_settings):
        """Test embedding request has timeout configured."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_response = MagicMock(spec=Response)
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * mock_settings.EMBEDDING_DIMENSION, "index": 0}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_client_instance.post.return_value = mock_response

            await service.get_embeddings("test")

            call_kwargs = mock_client_instance.post.call_args[1]
            assert call_kwargs["timeout"] == EMBEDDING_TIMEOUT

    @pytest.mark.asyncio
    async def test_get_embeddings_raises_on_dimension_mismatch(self, mock_settings):
        """Test embedding raises when API returns unexpected vector size."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_response = MagicMock(spec=Response)
            mock_response.json.return_value = {"data": [{"embedding": [0.1] * 3, "index": 0}]}
            mock_response.raise_for_status = MagicMock()
            mock_client_instance.post.return_value = mock_response

            with pytest.raises(EmbeddingError, match="Embedding dimension mismatch"):
                await service.get_embeddings("test")

    @pytest.mark.asyncio
    async def test_get_embeddings_raises_on_http_error(self, mock_settings):
        """Test embedding raises EmbeddingError on HTTP error."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_response = MagicMock(spec=Response)
            mock_response.raise_for_status.side_effect = Exception("HTTP Error")
            mock_client_instance.post.return_value = mock_response

            with pytest.raises(EmbeddingError) as exc_info:
                await service.get_embeddings("test")

            assert "Embedding API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embeddings_raises_on_network_error(self, mock_settings):
        """Test embedding raises EmbeddingError on network error."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_class.return_value = mock_client_instance

            mock_client_instance.post.side_effect = Exception("Network error")

            with pytest.raises(EmbeddingError) as exc_info:
                await service.get_embeddings("test")

            assert "Embedding API error" in str(exc_info.value)


class TestEmbeddingServiceCacheAutoEnabled:
    """Tests verifying that caching is automatic without manual setup."""

    def _make_mock_client(self, embedding: list):
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_response = MagicMock(spec=Response)
        mock_response.json.return_value = {"data": [{"embedding": embedding, "index": 0}]}
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response
        return mock_client_instance

    @pytest.mark.asyncio
    async def test_cache_enabled_by_default(self, mock_settings):
        """EmbeddingService(settings) auto-creates a cache — no manual setup."""
        service = EmbeddingService(settings=mock_settings)
        assert service._cache is not None
        assert service.cache_stats is not None
        assert service.cache_stats.size == 0

    @pytest.mark.asyncio
    async def test_auto_cache_stores_result_on_first_call(self, mock_settings):
        """First call hits API and caches; second call skips API entirely."""
        service = EmbeddingService(settings=mock_settings)
        embedding = [0.1] * mock_settings.EMBEDDING_DIMENSION

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = self._make_mock_client(embedding)
            first = await service.get_embeddings("fraud text")

        with patch("httpx.AsyncClient") as mock_client_class_2:
            second = await service.get_embeddings("fraud text")
            mock_client_class_2.assert_not_called()

        assert first == second == embedding
        assert service.cache_stats.hits == 1
        assert service.cache_stats.misses == 1
        assert service.cache_stats.size == 1

    @pytest.mark.asyncio
    async def test_different_texts_cached_independently(self, mock_settings):
        """Distinct texts produce independent cache entries automatically."""
        service = EmbeddingService(settings=mock_settings)
        vec_a = [0.1] * mock_settings.EMBEDDING_DIMENSION
        vec_b = [0.9] * mock_settings.EMBEDDING_DIMENSION

        with patch("httpx.AsyncClient") as mock_a:
            mock_a.return_value = self._make_mock_client(vec_a)
            result_a = await service.get_embeddings("text-a")

        with patch("httpx.AsyncClient") as mock_b:
            mock_b.return_value = self._make_mock_client(vec_b)
            result_b = await service.get_embeddings("text-b")

        assert result_a == vec_a
        assert result_b == vec_b
        assert service.cache_stats.size == 2

    @pytest.mark.asyncio
    async def test_explicit_none_disables_cache(self, mock_settings):
        """Passing cache=None explicitly disables caching."""
        service = EmbeddingService(settings=mock_settings, cache=None)
        embedding = [0.5] * mock_settings.EMBEDDING_DIMENSION

        assert service._cache is None
        assert service.cache_stats is None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = self._make_mock_client(embedding)
            await service.get_embeddings("text")
            await service.get_embeddings("text")
            assert mock_client_class.call_count == 2

    @pytest.mark.asyncio
    async def test_custom_cache_instance_accepted(self, mock_settings):
        """A user-supplied EmbeddingCache instance is used as-is."""
        custom_cache = EmbeddingCache(max_size=5, ttl_seconds=60)
        service = EmbeddingService(settings=mock_settings, cache=custom_cache)

        assert service._cache is custom_cache
        assert service._cache.max_size == 5

    @pytest.mark.asyncio
    async def test_api_error_does_not_pollute_cache(self, mock_settings):
        """Failed API calls must not write anything into the cache."""
        service = EmbeddingService(settings=mock_settings)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.post.side_effect = Exception("API down")
            mock_client_class.return_value = mock_client_instance

            with pytest.raises(EmbeddingError):
                await service.get_embeddings("text")

        assert service.cache_stats.size == 0

    @pytest.mark.asyncio
    async def test_cache_uses_settings_max_size(self, mock_settings):
        """Auto-created cache respects EMBEDDING_CACHE_MAX_SIZE from settings."""
        service = EmbeddingService(settings=mock_settings)
        assert service._cache is not None
        assert service._cache.max_size == mock_settings.EMBEDDING_CACHE_MAX_SIZE

    @pytest.mark.asyncio
    async def test_cache_uses_settings_ttl(self, mock_settings):
        """Auto-created cache respects EMBEDDING_CACHE_TTL_SECONDS from settings."""
        service = EmbeddingService(settings=mock_settings)
        assert service._cache is not None
        assert service._cache.ttl_seconds == mock_settings.EMBEDDING_CACHE_TTL_SECONDS
