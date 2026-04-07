"""
Pytest configuration and shared fixtures for Anti-Fraud RAG tests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import Settings
from app.db.models import Case, Tip
from app.main import app

# Test database URL (in-memory SQLite for unit tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    return Settings(
        EMBEDDING_MODEL_URL="https://api.test.com/embeddings",
        EMBEDDING_MODEL_API_KEY="test-api-key",
        EMBEDDING_MODEL_NAME="test-embedding-model",
        EMBEDDING_DIMENSION=1536,
        PORT=8000,
        HIGH_RISK_THRESHOLD=0.85,
        API_KEY="test-api-key-for-testing",
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
    )


@pytest.fixture
def mock_embedding_response() -> dict:
    """Mock response from embedding API."""
    return {
        "data": [
            {
                "embedding": [0.1] * 1536,  # Mock 1536-dimensional embedding
                "index": 0,
            }
        ],
        "model": "test-embedding-model",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


@pytest.fixture
def mock_async_client(mock_embedding_response: dict) -> AsyncMock:
    """Create a mock async HTTP client."""
    mock_client = AsyncMock()

    # Mock the response
    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = mock_embedding_response
    mock_response.raise_for_status = MagicMock()

    # Configure the mock client's post method
    mock_client.post = AsyncMock(return_value=mock_response)

    return mock_client


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for synchronous testing."""
    return TestClient(app)


@pytest.fixture
def sample_case() -> Case:
    """Create a sample Case for testing."""
    case = Case(
        id="12345678-1234-5678-1234-567812345678",
        description="This is a test fraud case about phone scams",
        fraud_type="phone_scam",
        amount=1000.00,
        keywords=["phone", "scam", "fraud"],
        embedding=[0.1] * 1536,
    )
    return case


@pytest.fixture
def sample_tip() -> Tip:
    """Create a sample Tip for testing."""
    tip = Tip(
        id="87654321-4321-8765-4321-876543218765",
        title="Phone Scam Warning",
        content="Be careful of phone scams asking for personal information",
        category="phone_fraud",
        keywords=["phone", "scam", "warning"],
        embedding=[0.2] * 1536,
    )
    return tip


@pytest.fixture
def sample_cases_list() -> list:
    """Create a list of sample cases for RRF fusion testing."""
    cases = []
    for i in range(5):
        case = Case(
            id=f"12345678-1234-5678-1234-{i:012d}",
            description=f"Test case {i} for fraud detection",
            fraud_type=f"type_{i}",
            amount=100.0 * (i + 1),
            keywords=[f"keyword_{i}"],
            embedding=[0.1 * (i + 1)] * 1536,
        )
        cases.append(case)
    return cases


@pytest.fixture
def sample_analysis_request() -> dict:
    """Create a sample analysis request body."""
    return {
        "request": {
            "text": "Someone called me asking for my bank details",
            "source": "user_submission",
            "metadata": {"user_id": "user123", "channel": "web"},
        }
    }


@pytest.fixture
def sample_case_create_request() -> dict:
    """Create a sample case create request body."""
    return {
        "case": {
            "description": "A new fraud case to inject",
            "fraud_type": "investment_scam",
            "amount": 5000.00,
            "keywords": ["investment", "scam"],
        }
    }


@pytest.fixture
def sample_tip_create_request() -> dict:
    """Create a sample tip create request body."""
    return {
        "tip": {
            "title": "Investment Scam Alert",
            "content": "Watch out for investments promising high returns",
            "category": "investment_fraud",
            "keywords": ["investment", "alert"],
        }
    }
