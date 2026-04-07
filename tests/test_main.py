"""
Unit tests for app/main.py - FastAPI application.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.main import app, verify_api_key


class TestHealthCheck:
    """Tests for the health_check endpoint."""

    def test_health_check_returns_healthy(self, test_client):
        """Test health check returns healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_health_check_no_auth_required(self, test_client):
        """Test health check does not require authentication."""
        response = test_client.get("/health")
        assert response.status_code == 200


class TestVerifyApiKey:
    """Tests for the verify_api_key dependency."""

    def test_verify_api_key_valid(self, test_client, mock_settings):
        """Test verify_api_key accepts valid API key."""
        with patch("app.main.settings", mock_settings):
            response = test_client.get(
                "/api/v1/analyze", headers={"x-api-key": mock_settings.API_KEY}
            )
            # Will return 422 (validation error) not 401 since /analyze needs body
            assert response.status_code != 401

    def test_verify_api_key_missing(self, test_client):
        """Test verify_api_key raises 401 when missing."""
        response = test_client.post("/api/v1/analyze", json={"request": {"text": "test"}})
        assert response.status_code == 401

    def test_verify_api_key_invalid(self, test_client):
        """Test verify_api_key raises 401 for invalid key."""
        response = test_client.post(
            "/api/v1/analyze",
            json={"request": {"text": "test"}},
            headers={"x-api-key": "invalid-key"},
        )
        assert response.status_code == 401


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_title(self):
        """Test app has correct title."""
        assert app.title == "Anti-Fraud RAG System"

    def test_app_version(self):
        """Test app has correct version."""
        assert app.version == "0.1.0"

    def test_app_has_analyze_router(self):
        """Test app includes analyze router."""
        route_paths = [route.path for route in app.routes]
        assert any("/api/v1/analyze" in path for path in route_paths)

    def test_app_has_data_router(self):
        """Test app includes data injection router."""
        route_paths = [r.path for r in app.routes]
        assert any("/api/v1/data" in path for path in route_paths)

    def test_app_has_health_endpoint(self):
        """Test app has health endpoint."""
        route_paths = [route.path for route in app.routes]
        assert any("/health" in path for path in route_paths)


class TestApiEndpointsAuth:
    """Tests for API endpoint authentication."""

    def test_analyze_endpoint_requires_auth(self, test_client):
        """Test /api/v1/analyze requires authentication."""
        response = test_client.post("/api/v1/analyze", json={"request": {"text": "test"}})
        assert response.status_code == 401

    def test_data_case_endpoint_requires_auth(self, test_client):
        """Test /api/v1/data/case requires authentication."""
        response = test_client.post("/api/v1/data/case", json={"case": {"description": "test"}})
        assert response.status_code == 401

    def test_data_tip_endpoint_requires_auth(self, test_client):
        """Test /api/v1/data/tip requires authentication."""
        response = test_client.post(
            "/api/v1/data/tip", json={"tip": {"title": "test", "content": "test"}}
        )
        assert response.status_code == 401


class TestVerifyApiKeyFunction:
    """Unit tests for verify_api_key function."""

    @pytest.mark.asyncio
    async def test_verify_api_key_with_valid_key(self, mock_settings):
        """Test verify_api_key returns key when valid."""
        with patch("app.main.settings", mock_settings):
            result = await verify_api_key(mock_settings.API_KEY)
            assert result == mock_settings.API_KEY

    @pytest.mark.asyncio
    async def test_verify_api_key_with_invalid_key(self, mock_settings):
        """Test verify_api_key raises HTTPException when invalid."""
        with patch("app.main.settings", mock_settings):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key("wrong-key")
            assert exc_info.value.status_code == 401
            assert "Invalid API Key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_api_key_with_none(self, mock_settings):
        """Test verify_api_key raises HTTPException when None."""
        with patch("app.main.settings", mock_settings):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(None)
            assert exc_info.value.status_code == 401
