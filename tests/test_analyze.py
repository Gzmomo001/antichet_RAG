"""
Unit tests for app/api/v1/analyze.py - analyze endpoint.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.api.v1.analyze import analyze_risk
from app.db.models import Case, Tip
from app.schemas.data import AnalysisRequest, AnalysisRequestBody


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def mock_analysis_request():
    """Create a mock analysis request."""
    return AnalysisRequestBody(
        request=AnalysisRequest(
            text="Someone is asking for my bank account",
            source="test",
        )
    )


@pytest.fixture
def mock_case():
    """Create a mock case for testing."""
    case = MagicMock(spec=Case)
    case.id = UUID("12345678-1234-5678-1234-567812345678")
    case.description = "Bank account phishing scam"
    case.fraud_type = "phishing"
    case.keywords = ["bank", "account", "phishing"]
    return case


@pytest.fixture
def mock_tip():
    """Create a mock tip for testing."""
    tip = MagicMock(spec=Tip)
    tip.id = UUID("87654321-4321-8765-4321-876543218765")
    tip.title = "Bank Security Tip"
    tip.content = "Never share your bank details"
    return tip


class TestAnalyzeRisk:
    """Tests for the analyze_risk endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_risk_direct_hit_high_score(
        self, mock_db, mock_analysis_request, mock_case, mock_tip
    ):
        """Test analyze returns Direct_Hit when score exceeds threshold."""
        mock_embedding = [0.9] * 1536

        with patch("app.api.v1.analyze.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            with patch("app.api.v1.analyze.RetrievalService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service

                # BM25 returns one case
                mock_service.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.95)])
                # Vector returns same case with high score
                mock_service.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.98)])
                # RRF fusion returns combined high score
                mock_service.rrf_fusion = MagicMock(
                    return_value=[{"item": mock_case, "score": 0.95}]
                )
                # Tips search
                mock_service.search_tips = AsyncMock(return_value=[mock_tip])

                response = await analyze_risk(mock_analysis_request, mock_db)

                assert response.result_type == "Direct_Hit"

    @pytest.mark.asyncio
    async def test_analyze_risk_rag_prompt_lower_score(
        self, mock_db, mock_analysis_request, mock_case, mock_tip
    ):
        """Test analyze returns RAG_Prompt when score is below threshold."""
        mock_embedding = [0.3] * 1536

        with patch("app.api.v1.analyze.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            with patch("app.api.v1.analyze.RetrievalService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service

                mock_service.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.4)])
                mock_service.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.5)])
                mock_service.rrf_fusion = MagicMock(
                    return_value=[
                        {"item": mock_case, "score": 0.5}  # Below 0.85 threshold
                    ]
                )
                mock_service.search_tips = AsyncMock(return_value=[mock_tip])

                response = await analyze_risk(mock_analysis_request, mock_db)

                assert response.result_type == "RAG_Prompt"
                assert response.data.rrf_score == 0.5

    @pytest.mark.asyncio
    async def test_analyze_risk_no_results(self, mock_db, mock_analysis_request, mock_tip):
        """Test analyze returns RAG_Prompt with empty results."""
        mock_embedding = [0.1] * 1536

        with patch("app.api.v1.analyze.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            with patch("app.api.v1.analyze.RetrievalService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service

                mock_service.search_cases_bm25 = AsyncMock(return_value=[])
                mock_service.search_cases_vector = AsyncMock(return_value=[])
                mock_service.rrf_fusion = MagicMock(return_value=[])
                mock_service.search_tips = AsyncMock(return_value=[mock_tip])

                response = await analyze_risk(mock_analysis_request, mock_db)

                assert response.result_type == "RAG_Prompt"
                assert response.data.rrf_score == 0.0
                assert response.data.risk_level == "LOW"

    @pytest.mark.asyncio
    async def test_analyze_risk_embedding_error(self, mock_db, mock_analysis_request):
        """Test analyze raises HTTPException when embedding fails."""
        with patch("app.api.v1.analyze.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(side_effect=RuntimeError("Embedding API error"))

            with pytest.raises(HTTPException) as exc_info:
                await analyze_risk(mock_analysis_request, mock_db)

            assert exc_info.value.status_code == 502
            assert "Embedding error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_analyze_risk_risk_level_medium_for_mid_scores(
        self, mock_db, mock_analysis_request, mock_case, mock_tip
    ):
        """Test risk level is MEDIUM for mid-range scores (0.5 - 0.85)."""
        mock_embedding = [0.6] * 1536

        with patch("app.api.v1.analyze.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            with patch("app.api.v1.analyze.RetrievalService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service

                mock_service.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.7)])
                mock_service.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.7)])
                mock_service.rrf_fusion = MagicMock(
                    return_value=[{"item": mock_case, "score": 0.6}]
                )
                mock_service.search_tips = AsyncMock(return_value=[mock_tip])

                response = await analyze_risk(mock_analysis_request, mock_db)

                assert response.result_type == "RAG_Prompt"
                assert response.data.risk_level == "MEDIUM"

    @pytest.mark.asyncio
    async def test_analyze_risk_risk_level_low_for_low_scores(
        self, mock_db, mock_analysis_request, mock_case, mock_tip
    ):
        """Test risk level is LOW for low scores (< 0.5)."""
        mock_embedding = [0.3] * 1536

        with patch("app.api.v1.analyze.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            with patch("app.api.v1.analyze.RetrievalService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service

                mock_service.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.4)])
                mock_service.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.3)])
                mock_service.rrf_fusion = MagicMock(
                    return_value=[{"item": mock_case, "score": 0.35}]
                )
                mock_service.search_tips = AsyncMock(return_value=[mock_tip])

                response = await analyze_risk(mock_analysis_request, mock_db)

                assert response.result_type == "RAG_Prompt"
                assert response.data.risk_level == "LOW"

    @pytest.mark.asyncio
    async def test_analyze_risk_includes_context(
        self, mock_db, mock_analysis_request, mock_case, mock_tip
    ):
        """Test analyze response includes relevant context."""
        mock_embedding = [0.3] * 1536

        with patch("app.api.v1.analyze.embedding_service") as mock_embed:
            mock_embed.get_embeddings = AsyncMock(return_value=mock_embedding)

            with patch("app.api.v1.analyze.RetrievalService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service

                mock_service.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.5)])
                mock_service.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.5)])
                mock_service.rrf_fusion = MagicMock(
                    return_value=[{"item": mock_case, "score": 0.5}]
                )
                mock_service.search_tips = AsyncMock(return_value=[mock_tip])

                response = await analyze_risk(mock_analysis_request, mock_db)

                assert len(response.data.context.relevant_cases) == 1
                assert len(response.data.context.anti_fraud_tips) == 1
                assert response.data.context.anti_fraud_tips[0]["title"] == "Bank Security Tip"
