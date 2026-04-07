"""
Unit tests for app/api/v1/data.py - data injection endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.api.v1.data import inject_case, inject_tip
from app.db.models import Case, Tip
from app.schemas.data import CaseCreate, CaseCreateRequest, TipCreate, TipCreateRequest


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def mock_case_request():
    """Create a mock case creation request."""
    return CaseCreateRequest(
        case=CaseCreate(
            description="New fraud case for testing",
            fraud_type="investment_scam",
            amount=5000.00,
            keywords=["investment", "scam"],
        )
    )


@pytest.fixture
def mock_tip_request():
    """Create a mock tip creation request."""
    return TipCreateRequest(
        tip=TipCreate(
            title="Test Tip",
            content="Test content for tip",
            category="test_category",
            keywords=["test", "tip"],
        )
    )


@pytest.fixture
def mock_created_case():
    """Create a mock created case with ID."""
    case = MagicMock(spec=Case)
    case.id = UUID("12345678-1234-5678-1234-567812345678")
    case.description = "New fraud case for testing"
    case.fraud_type = "investment_scam"
    case.amount = 5000.00
    case.keywords = ["investment", "scam"]
    return case


@pytest.fixture
def mock_created_tip():
    """Create a mock created tip with ID."""
    tip = MagicMock(spec=Tip)
    tip.id = UUID("87654321-4321-8765-4321-876543218765")
    tip.title = "Test Tip"
    tip.content = "Test content for tip"
    tip.category = "test_category"
    tip.keywords = ["test", "tip"]
    return tip


class TestInjectCase:
    """Tests for the inject_case endpoint."""

    @pytest.mark.asyncio
    async def test_inject_case_success(self, mock_db, mock_case_request, mock_created_case):
        """Test successful case injection."""
        mock_embedding = [0.1] * 1536

        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            mock_db.add = MagicMock()
            mock_db.commit = AsyncMock()
            mock_db.refresh = AsyncMock(
                side_effect=lambda x: setattr(x, "id", mock_created_case.id)
            )

            response = await inject_case(mock_case_request, mock_db)

            assert response["status"] == "success"
            assert response["message"] == "案例注入成功"
            assert "case_id" in response
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_case_gets_embedding(self, mock_db, mock_case_request, mock_created_case):
        """Test case injection obtains embedding for description."""
        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=[0.1] * 1536)

            mock_db.add = MagicMock()
            mock_db.commit = AsyncMock()
            mock_db.refresh = AsyncMock(
                side_effect=lambda x: setattr(x, "id", mock_created_case.id)
            )

            await inject_case(mock_case_request, mock_db)

            mock_embed.get_embeddings.assert_called_once_with(mock_case_request.case.description)

    @pytest.mark.asyncio
    async def test_inject_case_rollback_on_error(self, mock_db, mock_case_request):
        """Test case injection rolls back on error."""
        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=[0.1] * 1536)
            mock_db.add = MagicMock(side_effect=Exception("DB Error"))
            mock_db.rollback = AsyncMock()

            with pytest.raises(HTTPException) as exc_info:
                await inject_case(mock_case_request, mock_db)

            assert exc_info.value.status_code == 500
            mock_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_case_embedding_error(self, mock_db, mock_case_request):
        """Test case injection handles embedding error."""
        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(side_effect=RuntimeError("Embedding API error"))

            with pytest.raises(HTTPException) as exc_info:
                await inject_case(mock_case_request, mock_db)

            assert exc_info.value.status_code == 500


class TestInjectTip:
    """Tests for the inject_tip endpoint."""

    @pytest.mark.asyncio
    async def test_inject_tip_success(self, mock_db, mock_tip_request, mock_created_tip):
        """Test successful tip injection."""
        mock_embedding = [0.2] * 1536

        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            mock_db.add = MagicMock()
            mock_db.commit = AsyncMock()
            mock_db.refresh = AsyncMock(side_effect=lambda x: setattr(x, "id", mock_created_tip.id))

            response = await inject_tip(mock_tip_request, mock_db)

            assert response["status"] == "success"
            assert response["message"] == "知识注入成功"
            assert "tip_id" in response
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_tip_combines_title_and_content(
        self, mock_db, mock_tip_request, mock_created_tip
    ):
        """Test tip injection combines title and content for embedding."""
        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=[0.2] * 1536)

            mock_db.add = MagicMock()
            mock_db.commit = AsyncMock()
            mock_db.refresh = AsyncMock(side_effect=lambda x: setattr(x, "id", mock_created_tip.id))

            await inject_tip(mock_tip_request, mock_db)

            expected_text = f"{mock_tip_request.tip.title} {mock_tip_request.tip.content}"
            mock_embed.get_embeddings.assert_called_once_with(expected_text)

    @pytest.mark.asyncio
    async def test_inject_tip_rollback_on_error(self, mock_db, mock_tip_request):
        """Test tip injection rolls back on error."""
        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=[0.2] * 1536)
            mock_db.add = MagicMock(side_effect=Exception("DB Error"))
            mock_db.rollback = AsyncMock()

            with pytest.raises(HTTPException) as exc_info:
                await inject_tip(mock_tip_request, mock_db)

            assert exc_info.value.status_code == 500
            mock_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_tip_embedding_error(self, mock_db, mock_tip_request):
        """Test tip injection handles embedding error."""
        with patch("app.api.v1.data.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(side_effect=RuntimeError("Embedding API error"))

            with pytest.raises(HTTPException) as exc_info:
                await inject_tip(mock_tip_request, mock_db)

            assert exc_info.value.status_code == 500
