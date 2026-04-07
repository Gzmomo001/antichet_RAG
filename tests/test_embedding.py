"""
Unit tests for app/services/embedding.py - EmbeddingService.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Response

from app.services.embedding import EmbeddingService


class TestEmbeddingService:
    """Tests for the EmbeddingService class."""

    def test_embedding_service_initialization(self, mock_settings):
        """Test EmbeddingService initializes with correct settings."""
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()
            assert service.url == mock_settings.EMBEDDING_MODEL_URL
            assert service.api_key == mock_settings.EMBEDDING_MODEL_API_KEY
            assert service.model == mock_settings.EMBEDDING_MODEL_NAME

    @pytest.mark.asyncio
    async def test_get_embeddings_success(self, mock_settings, mock_async_client):
        """Test successful embedding retrieval."""
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__.return_value = mock_client_instance
                mock_client_instance.__aexit__.return_value = None
                mock_client_class.return_value = mock_client_instance

                mock_response = MagicMock(spec=Response)
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
                }
                mock_response.raise_for_status = MagicMock()
                mock_client_instance.post.return_value = mock_response

                result = await service.get_embeddings("test text")

                assert result == [0.1, 0.2, 0.3]
                mock_client_instance.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embeddings_calls_correct_endpoint(self, mock_settings):
        """Test embedding request is made to correct endpoint."""
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__.return_value = mock_client_instance
                mock_client_instance.__aexit__.return_value = None
                mock_client_class.return_value = mock_client_instance

                mock_response = MagicMock(spec=Response)
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1536, "index": 0}]
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
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__.return_value = mock_client_instance
                mock_client_instance.__aexit__.return_value = None
                mock_client_class.return_value = mock_client_instance

                mock_response = MagicMock(spec=Response)
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1536, "index": 0}]
                }
                mock_response.raise_for_status = MagicMock()
                mock_client_instance.post.return_value = mock_response

                await service.get_embeddings("test")

                call_kwargs = mock_client_instance.post.call_args[1]
                assert "Bearer" in call_kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_get_embeddings_includes_model_in_payload(self, mock_settings):
        """Test embedding request includes model in JSON payload."""
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__.return_value = mock_client_instance
                mock_client_instance.__aexit__.return_value = None
                mock_client_class.return_value = mock_client_instance

                mock_response = MagicMock(spec=Response)
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1536, "index": 0}]
                }
                mock_response.raise_for_status = MagicMock()
                mock_client_instance.post.return_value = mock_response

                await service.get_embeddings("test")

                call_kwargs = mock_client_instance.post.call_args[1]
                assert call_kwargs["json"]["model"] == mock_settings.EMBEDDING_MODEL_NAME

    @pytest.mark.asyncio
    async def test_get_embeddings_timeout(self, mock_settings):
        """Test embedding request has timeout configured."""
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__.return_value = mock_client_instance
                mock_client_instance.__aexit__.return_value = None
                mock_client_class.return_value = mock_client_instance

                mock_response = MagicMock(spec=Response)
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1536, "index": 0}]
                }
                mock_response.raise_for_status = MagicMock()
                mock_client_instance.post.return_value = mock_response

                await service.get_embeddings("test")

                call_kwargs = mock_client_instance.post.call_args[1]
                assert call_kwargs["timeout"] == 10.0

    @pytest.mark.asyncio
    async def test_get_embeddings_raises_on_http_error(self, mock_settings):
        """Test embedding raises RuntimeError on HTTP error."""
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__.return_value = mock_client_instance
                mock_client_instance.__aexit__.return_value = None
                mock_client_class.return_value = mock_client_instance

                mock_response = MagicMock(spec=Response)
                mock_response.raise_for_status.side_effect = Exception("HTTP Error")
                mock_client_instance.post.return_value = mock_response

                with pytest.raises(RuntimeError) as exc_info:
                    await service.get_embeddings("test")

                assert "Embedding API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embeddings_raises_on_network_error(self, mock_settings):
        """Test embedding raises RuntimeError on network error."""
        with patch("app.services.embedding.settings", mock_settings):
            service = EmbeddingService()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__.return_value = mock_client_instance
                mock_client_instance.__aexit__.return_value = None
                mock_client_class.return_value = mock_client_instance

                mock_client_instance.post.side_effect = Exception("Network error")

                with pytest.raises(RuntimeError) as exc_info:
                    await service.get_embeddings("test")

                assert "Embedding API error" in str(exc_info.value)


class TestEmbeddingServiceSingleton:
    """Tests for the module-level embedding_service singleton."""

    def test_embedding_service_singleton_exists(self):
        """Test embedding_service singleton is created."""
        from app.services.embedding import embedding_service

        assert embedding_service is not None
        assert isinstance(embedding_service, EmbeddingService)
