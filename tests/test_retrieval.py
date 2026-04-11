"""
Unit tests for antifraud_rag/services/retrieval.py - RetrievalService.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from antifraud_rag.db.models import Case, Tip
from antifraud_rag.services.retrieval import RetrievalService


class TestRetrievalServiceRRF:
    """Tests for RRF (Reciprocal Rank Fusion) algorithm."""

    def test_rrf_fusion_empty_results(self):
        """Test RRF fusion with empty results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        result = service.rrf_fusion([], [])
        assert result == []

    def test_rrf_fusion_bm25_only(self):
        """Test RRF fusion with only BM25 results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case1 = MagicMock(spec=Case)
        mock_case1.id = "case1"
        mock_case2 = MagicMock(spec=Case)
        mock_case2.id = "case2"

        bm25_results = [(mock_case1, 0.9), (mock_case2, 0.7)]

        result = service.rrf_fusion(bm25_results, [])

        assert len(result) == 2
        assert result[0]["item"].id == "case1"
        assert result[1]["item"].id == "case2"

    def test_rrf_fusion_vector_only(self):
        """Test RRF fusion with only vector results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case1 = MagicMock(spec=Case)
        mock_case1.id = "case1"
        mock_case2 = MagicMock(spec=Case)
        mock_case2.id = "case2"

        vector_results = [(mock_case2, 0.9), (mock_case1, 0.8)]

        result = service.rrf_fusion([], vector_results)

        assert len(result) == 2
        assert result[0]["item"].id == "case2"
        assert result[1]["item"].id == "case1"

    def test_rrf_fusion_combined_results(self):
        """Test RRF fusion with both BM25 and vector results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case1 = MagicMock(spec=Case)
        mock_case1.id = "case1"
        mock_case2 = MagicMock(spec=Case)
        mock_case2.id = "case2"
        mock_case3 = MagicMock(spec=Case)
        mock_case3.id = "case3"

        bm25_results = [
            (mock_case1, 0.9),
            (mock_case2, 0.7),
        ]
        vector_results = [
            (mock_case2, 0.95),
            (mock_case3, 0.85),
        ]

        result = service.rrf_fusion(bm25_results, vector_results)

        assert len(result) == 3
        result_ids = [r["item"].id for r in result]
        assert result_ids[0] == "case2"

    def test_rrf_fusion_normalize_parameter(self):
        """Test normalize parameter controls score normalization."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case = MagicMock(spec=Case)
        mock_case.id = "case1"

        result_normalized = service.rrf_fusion([(mock_case, 1.0)], [], normalize=True)
        result_raw = service.rrf_fusion([(mock_case, 1.0)], [], normalize=False)

        assert result_normalized[0]["score"] > result_raw[0]["score"]

    def test_rrf_fusion_combined_item_scores_higher(self):
        """Item in both lists scores higher than item in only one list."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case_a = MagicMock(spec=Case)
        mock_case_a.id = "case_a"
        mock_case_b = MagicMock(spec=Case)
        mock_case_b.id = "case_b"

        bm25_results = [(mock_case_a, 0.9)]
        vector_results = [(mock_case_a, 0.8), (mock_case_b, 0.7)]

        result = service.rrf_fusion(bm25_results, vector_results)

        assert len(result) == 2
        assert result[0]["item"].id == "case_a"
        assert result[0]["score"] > result[1]["score"]

    def test_rrf_fusion_score_calculation(self):
        """Test RRF score calculation formula (normalized)."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case = MagicMock(spec=Case)
        mock_case.id = "case1"

        result = service.rrf_fusion([(mock_case, 1.0)], [])
        raw_score = 1 / (60 + 0 + 1)
        max_score = 2 / (60 + 1)
        expected_score = raw_score / max_score
        assert result[0]["score"] == pytest.approx(expected_score)

    def test_rrf_fusion_results_sorted_by_score(self):
        """Test RRF fusion results are sorted by score in descending order."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_cases = []
        for i in range(5):
            mock_case = MagicMock(spec=Case)
            mock_case.id = f"case{i}"
            mock_cases.append(mock_case)

        bm25_results = [(mock_cases[0], 1), (mock_cases[1], 1)]
        vector_results = [(mock_cases[2], 1), (mock_cases[3], 1), (mock_cases[4], 1)]

        result = service.rrf_fusion(bm25_results, vector_results)

        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_fusion_item_structure(self):
        """Test RRF fusion result item structure."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case = MagicMock(spec=Case)
        mock_case.id = "test_case"

        result = service.rrf_fusion([(mock_case, 1.0)], [])

        assert len(result) == 1
        item = result[0]
        assert "item" in item
        assert "score" in item
        assert item["item"].id == "test_case"


class TestRetrievalServiceVector:
    """Tests for vector search functionality."""

    @pytest.mark.asyncio
    async def test_search_cases_vector_returns_results(self):
        """Test vector search returns properly formatted results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case = MagicMock(spec=Case)
        mock_case.id = "test_case"

        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_case, 0.95)]
        mock_db.execute.return_value = mock_result

        query_embedding = [0.1] * 1536
        results = await service.search_cases_vector(query_embedding)

        assert len(results) == 1
        assert results[0][0].id == "test_case"
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_cases_vector_empty_results(self):
        """Test vector search with no results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.execute.return_value = mock_result

        results = await service.search_cases_vector([0.1] * 1536)
        assert results == []


class TestRetrievalServiceBM25:
    """Tests for BM25 search functionality."""

    @pytest.mark.asyncio
    async def test_search_cases_bm25_returns_results(self):
        """Test BM25 search returns properly formatted results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_case = MagicMock(spec=Case)
        mock_case.id = "bm25_test_case"

        mock_result1 = MagicMock()
        mock_result1.all.return_value = [("bm25_test_case", 0.8)]
        mock_result2 = MagicMock()
        mock_result2.scalars.return_value.all.return_value = [mock_case]

        mock_db.execute.side_effect = [mock_result1, mock_result2]

        results = await service.search_cases_bm25("test query")

        assert len(results) == 1
        assert results[0][0].id == "bm25_test_case"

    @pytest.mark.asyncio
    async def test_search_cases_bm25_empty_results(self):
        """Test BM25 search with no results."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.execute.return_value = mock_result

        results = await service.search_cases_bm25("nonexistent query")
        assert results == []


class TestRetrievalServiceTips:
    """Tests for tips search functionality."""

    @pytest.mark.asyncio
    async def test_search_tips_returns_tips(self):
        """Test tips search returns list of tips."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_tip1 = MagicMock(spec=Tip)
        mock_tip1.id = "tip1"
        mock_tip2 = MagicMock(spec=Tip)
        mock_tip2.id = "tip2"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_tip1, mock_tip2]
        mock_db.execute.return_value = mock_result

        results = await service.search_tips("test query", [0.1] * 1536)

        assert len(results) == 2
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_tips_respects_limit(self):
        """Test tips search respects limit parameter."""
        mock_db = AsyncMock()
        service = RetrievalService(mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        await service.search_tips("test", [0.1] * 1536, limit=3)

        mock_db.execute.assert_called_once()
