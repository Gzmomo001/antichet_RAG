"""
Unit tests for antifraud_rag/main.py - AntiFraudRAG class.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from antifraud_rag import AntiFraudRAG, Settings
from antifraud_rag.db.models import Case, Tip


@pytest.fixture
def mock_settings():
    return Settings(
        EMBEDDING_MODEL_URL="https://api.test.com/embeddings",
        EMBEDDING_MODEL_API_KEY="test-key",
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        HIGH_RISK_THRESHOLD=0.85,
    )


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_embedding_service(mock_settings):
    service = MagicMock()
    service.get_embeddings = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def mock_case():
    case = MagicMock(spec=Case)
    case.id = UUID("12345678-1234-5678-1234-567812345678")
    case.description = "Test fraud case"
    case.fraud_type = "phishing"
    case.keywords = ["test"]
    return case


@pytest.fixture
def mock_tip():
    tip = MagicMock(spec=Tip)
    tip.id = UUID("87654321-4321-8765-4321-876543218765")
    tip.title = "Test Tip"
    tip.content = "Test content"
    return tip


class TestAntiFraudRAGClass:
    """Tests for AntiFraudRAG class."""

    def test_init_with_default_settings(self, mock_db):
        """Test AntiFraudRAG initializes with default settings."""
        with patch("antifraud_rag.main.Settings") as mock_settings_class:
            mock_settings_class.return_value = MagicMock(HIGH_RISK_THRESHOLD=0.85)
            rag = AntiFraudRAG(mock_db)
            assert rag.db == mock_db

    def test_init_with_custom_settings(self, mock_db, mock_settings):
        """Test AntiFraudRAG initializes with custom settings."""
        rag = AntiFraudRAG(mock_db, settings=mock_settings)
        assert rag.settings == mock_settings

    def test_init_with_custom_embedding_service(
        self, mock_db, mock_settings, mock_embedding_service
    ):
        """Test AntiFraudRAG initializes with custom embedding service."""
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )
        assert rag.embedding_service == mock_embedding_service

    @pytest.mark.asyncio
    async def test_analyze_returns_direct_hit(
        self, mock_db, mock_settings, mock_embedding_service, mock_case, mock_tip
    ):
        """Test analyze returns Direct_Hit when score exceeds threshold."""
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.9)])
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.95)])
            mock_retrieval.rrf_fusion = MagicMock(return_value=[{"item": mock_case, "score": 0.95}])
            mock_retrieval.search_tips = AsyncMock(return_value=[mock_tip])

            response = await rag.analyze("test query")

            assert response.result_type == "Direct_Hit"
            assert response.data.risk_level == "HIGH"

    @pytest.mark.asyncio
    async def test_analyze_returns_rag_prompt(
        self, mock_db, mock_settings, mock_embedding_service, mock_case, mock_tip
    ):
        """Test analyze returns RAG_Prompt when score is below threshold."""
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.6)])
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.7)])
            mock_retrieval.rrf_fusion = MagicMock(return_value=[{"item": mock_case, "score": 0.6}])
            mock_retrieval.search_tips = AsyncMock(return_value=[mock_tip])

            response = await rag.analyze("test query")

            assert response.result_type == "RAG_Prompt"
            assert response.data.risk_level == "MEDIUM"

    @pytest.mark.asyncio
    async def test_add_case_success(self, mock_db, mock_settings, mock_embedding_service):
        """Test add_case creates a case with embedding."""
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        mock_case = MagicMock(spec=Case)
        mock_case.id = UUID("12345678-1234-5678-1234-567812345678")

        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock(return_value=mock_case)

        with patch("antifraud_rag.main.Case") as mock_case_class:
            mock_case_class.return_value = mock_case
            await rag.add_case(description="test description", fraud_type="phishing")

            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_tip_success(self, mock_db, mock_settings, mock_embedding_service):
        """Test add_tip creates a tip with embedding."""
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        mock_tip = MagicMock(spec=Tip)
        mock_tip.id = UUID("87654321-4321-8765-4321-876543218765")

        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock(return_value=mock_tip)

        with patch("antifraud_rag.main.Tip") as mock_tip_class:
            mock_tip_class.return_value = mock_tip
            await rag.add_tip(title="Test Tip", content="Test content")

            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar_cases(
        self, mock_db, mock_settings, mock_embedding_service, mock_case
    ):
        """Test search_similar_cases returns formatted results."""
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.95)])

            results = await rag.search_similar_cases("test query", limit=5)

            assert len(results) == 1
            assert results[0]["case_id"] == mock_case.id
            assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_hybrid_search(self, mock_db, mock_settings, mock_embedding_service, mock_case):
        """Test hybrid_search returns RRF fused results."""
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.9)])
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.95)])
            mock_retrieval.rrf_fusion = MagicMock(return_value=[{"item": mock_case, "score": 0.92}])

            results = await rag.hybrid_search("test query", limit=10)

            assert len(results) == 1
            assert results[0]["rrf_score"] == 0.92

    @pytest.mark.asyncio
    async def test_analyze_returns_low_risk_when_no_matches(
        self, mock_db, mock_settings, mock_embedding_service, mock_tip
    ):
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[])
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[])
            mock_retrieval.rrf_fusion = MagicMock(return_value=[])
            mock_retrieval.search_tips = AsyncMock(return_value=[mock_tip])

            response = await rag.analyze("innocent query")

            assert response.result_type == "RAG_Prompt"
            assert response.data.risk_level == "LOW"
            assert response.data.rrf_score == 0.0

    @pytest.mark.asyncio
    async def test_analyze_returns_medium_when_score_between_thresholds(
        self, mock_db, mock_settings, mock_embedding_service, mock_case, mock_tip
    ):
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.6)])
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.7)])
            mock_retrieval.rrf_fusion = MagicMock(return_value=[{"item": mock_case, "score": 0.6}])
            mock_retrieval.search_tips = AsyncMock(return_value=[mock_tip])

            response = await rag.analyze("suspicious query")

            assert response.result_type == "RAG_Prompt"
            assert response.data.risk_level == "MEDIUM"

    @pytest.mark.asyncio
    async def test_analyze_returns_low_when_score_below_half(
        self, mock_db, mock_settings, mock_embedding_service, mock_case, mock_tip
    ):
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[(mock_case, 0.3)])
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[(mock_case, 0.4)])
            mock_retrieval.rrf_fusion = MagicMock(return_value=[{"item": mock_case, "score": 0.3}])
            mock_retrieval.search_tips = AsyncMock(return_value=[mock_tip])

            response = await rag.analyze("vaguely suspicious query")

            assert response.result_type == "RAG_Prompt"
            assert response.data.risk_level == "LOW"

    @pytest.mark.asyncio
    async def test_analyze_raises_on_embedding_error(
        self, mock_db, mock_settings, mock_embedding_service
    ):
        mock_embedding_service.get_embeddings = AsyncMock(side_effect=Exception("API down"))

        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with pytest.raises(RuntimeError, match="Embedding error"):
            await rag.analyze("test query")

    @pytest.mark.asyncio
    async def test_add_case_with_optional_fields(
        self, mock_db, mock_settings, mock_embedding_service
    ):
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        mock_case = MagicMock(spec=Case)
        mock_case.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock(return_value=mock_case)

        with patch("antifraud_rag.main.Case") as mock_case_class:
            mock_case_class.return_value = mock_case
            result = await rag.add_case(
                description="test",
                fraud_type="phishing",
                amount=9999.99,
                keywords=["urgent", "transfer"],
            )

            mock_case_class.assert_called_once_with(
                description="test",
                fraud_type="phishing",
                amount=9999.99,
                keywords=["urgent", "transfer"],
                embedding=[0.1] * 1536,
            )
            assert result == mock_case

    @pytest.mark.asyncio
    async def test_add_tip_uses_title_and_content_for_embedding(
        self, mock_db, mock_settings, mock_embedding_service
    ):
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        mock_tip = MagicMock(spec=Tip)
        mock_tip.id = UUID("87654321-4321-8765-4321-876543218765")
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock(return_value=mock_tip)

        with patch("antifraud_rag.main.Tip") as mock_tip_class:
            mock_tip_class.return_value = mock_tip
            await rag.add_tip(title="Title", content="Content", category="general")

            mock_embedding_service.get_embeddings.assert_called_once_with("Title Content")

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_limit(
        self, mock_db, mock_settings, mock_embedding_service
    ):
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        mock_case = MagicMock(spec=Case)
        mock_case.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_case.description = "Case"
        mock_case.fraud_type = "scam"

        fused_results = [
            {"item": mock_case, "score": 0.9},
            {"item": mock_case, "score": 0.8},
            {"item": mock_case, "score": 0.7},
        ]

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[])
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[])
            mock_retrieval.rrf_fusion = MagicMock(return_value=fused_results)

            results = await rag.hybrid_search("test", limit=2)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_similar_cases_empty_results(
        self, mock_db, mock_settings, mock_embedding_service
    ):
        rag = AntiFraudRAG(
            mock_db, settings=mock_settings, embedding_service=mock_embedding_service
        )

        with patch.object(rag, "retrieval_service") as mock_retrieval:
            mock_retrieval.search_cases_vector = AsyncMock(return_value=[])

            results = await rag.search_similar_cases("test query")
            assert results == []


class TestAntiFraudRAGIntegration:
    """Integration tests showing usage pattern."""

    @pytest.mark.asyncio
    async def test_full_usage_pattern(self, mock_settings):
        """Test full usage pattern with AntiFraudRAG class."""
        mock_db = AsyncMock()

        with patch("antifraud_rag.main.EmbeddingService") as mock_embed_class:
            mock_embed = MagicMock()
            mock_embed.get_embeddings = AsyncMock(return_value=[0.1] * 1536)
            mock_embed_class.return_value = mock_embed

            with patch("antifraud_rag.main.RetrievalService") as mock_retrieval_class:
                mock_retrieval = MagicMock()
                mock_retrieval.search_cases_bm25 = AsyncMock(return_value=[])
                mock_retrieval.search_cases_vector = AsyncMock(return_value=[])
                mock_retrieval.rrf_fusion = MagicMock(return_value=[])
                mock_retrieval.search_tips = AsyncMock(return_value=[])
                mock_retrieval_class.return_value = mock_retrieval

                rag = AntiFraudRAG(mock_db, settings=mock_settings)

                response = await rag.analyze("test query")

                assert response.result_type == "RAG_Prompt"
