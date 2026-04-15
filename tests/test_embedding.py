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


class TestEmbeddingServiceWithCache:
    """Tests for EmbeddingService cache integration."""

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
    async def test_cache_miss_calls_api_and_stores_result(self, mock_settings):
        """On first call the API is invoked and the result is cached."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=0)
        service = EmbeddingService(settings=mock_settings, cache=cache)
        embedding = [0.1] * mock_settings.EMBEDDING_DIMENSION

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = self._make_mock_client(embedding)
            result = await service.get_embeddings("fraud text")

        assert result == embedding
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0
        assert cache.stats.size == 1

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api_call(self, mock_settings):
        """Second call with identical text must not hit the API."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=0)
        service = EmbeddingService(settings=mock_settings, cache=cache)
        embedding = [0.2] * mock_settings.EMBEDDING_DIMENSION

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = self._make_mock_client(embedding)
            first = await service.get_embeddings("fraud text")

        with patch("httpx.AsyncClient") as mock_client_class_2:
            second = await service.get_embeddings("fraud text")
            mock_client_class_2.assert_not_called()

        assert first == second
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    @pytest.mark.asyncio
    async def test_different_texts_have_separate_cache_entries(self, mock_settings):
        """Distinct texts must produce independent cache entries."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=0)
        service = EmbeddingService(settings=mock_settings, cache=cache)
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
        assert cache.stats.size == 2

    @pytest.mark.asyncio
    async def test_no_cache_always_calls_api(self, mock_settings):
        """When cache=None the API is called every time."""
        service = EmbeddingService(settings=mock_settings, cache=None)
        embedding = [0.5] * mock_settings.EMBEDDING_DIMENSION

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = self._make_mock_client(embedding)
            await service.get_embeddings("text")
            await service.get_embeddings("text")
            assert mock_client_class.call_count == 2

    @pytest.mark.asyncio
    async def test_api_error_does_not_cache_result(self, mock_settings):
        """Failed API calls must not pollute the cache."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=0)
        service = EmbeddingService(settings=mock_settings, cache=cache)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.post.side_effect = Exception("API down")
            mock_client_class.return_value = mock_client_instance

            with pytest.raises(EmbeddingError):
                await service.get_embeddings("text")

        assert cache.stats.size == 0
