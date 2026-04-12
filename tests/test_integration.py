"""
Integration tests for FraudAnalyzer — full pipeline tests with real internal components.

These tests differ from unit tests in that they:
- Use the REAL FraudAnalyzer.analyze() / hybrid_search() / etc. methods
- Use the REAL RetrievalService.rrf_fusion() algorithm (not mocked)
- Use the REAL prompt builders (build_rag_prompt, build_matched_cases, etc.)
- Use the REAL Pydantic schemas for response validation
- Only mock external boundaries: embedding API and DB query execution

This catches integration bugs like:
- Mismatched data flow between components
- Wrong field names in prompt builders
- Schema validation errors
- Incorrect RRF score thresholds for risk levels

NOTE on RRF scores vs thresholds:
  RRF scores are now normalized: raw_score / (2/(k+1)), giving range [0, 1].
  HIGH_RISK_THRESHOLD=0.85 and MEDIUM threshold 0.5 are meaningful with normalized scores.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from antifraud_rag import FraudAnalyzer, Settings
from antifraud_rag.core.enums import ResultType, RiskLevel
from antifraud_rag.core.exceptions import EmbeddingError
from antifraud_rag.db.models import Case, Tip
from antifraud_rag.schemas import AnalysisResponse, DirectHitData, RAGPromptData
from antifraud_rag.services.embedding import EmbeddingService


@pytest.fixture
def settings():
    return Settings(
        EMBEDDING_MODEL_URL="https://api.test.com/embeddings",
        EMBEDDING_MODEL_API_KEY="test-key",
        HIGH_RISK_THRESHOLD=0.85,
    )


@pytest.fixture
def fake_embedding(settings):
    return [0.1] * settings.EMBEDDING_DIMENSION


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


def _make_embedding_service(fake_embedding):
    service = MagicMock(spec=EmbeddingService)
    service.get_embeddings = AsyncMock(return_value=fake_embedding)
    return service


def _make_case(
    case_id="12345678-1234-5678-1234-567812345678",
    description="Phone scam: caller pretends to be police and asks for bank transfer",
    fraud_type="phone_scam",
    amount=50000.0,
    keywords=None,
):
    case = MagicMock(spec=Case)
    case.id = UUID(case_id) if isinstance(case_id, str) else case_id
    case.description = description
    case.fraud_type = fraud_type
    case.amount = amount
    case.keywords = keywords or ["phone", "police", "transfer"]
    return case


def _make_tip(
    tip_id="87654321-4321-8765-4321-876543218765",
    title="Phone Scam Warning",
    content="Never transfer money to someone claiming to be police over the phone",
    category="phone_fraud",
    keywords=None,
):
    tip = MagicMock(spec=Tip)
    tip.id = UUID(tip_id) if isinstance(tip_id, str) else tip_id
    tip.title = title
    tip.content = content
    tip.category = category
    tip.keywords = keywords or []
    return tip


def _patch_retrieval(analyzer, *, bm25=None, vector=None, tips=None, rrf=None):
    """Patch retrieval methods. If rrf is provided, rrf_fusion is also patched (real otherwise)."""
    if bm25 is not None:
        patcher_bm25 = patch.object(
            analyzer.retrieval_service, "search_cases_bm25", return_value=bm25
        )
    else:
        patcher_bm25 = patch.object(
            analyzer.retrieval_service, "search_cases_bm25", return_value=[]
        )

    if vector is not None:
        patcher_vector = patch.object(
            analyzer.retrieval_service, "search_cases_vector", return_value=vector
        )
    else:
        patcher_vector = patch.object(
            analyzer.retrieval_service, "search_cases_vector", return_value=[]
        )

    if tips is not None:
        patcher_tips = patch.object(analyzer.retrieval_service, "search_tips", return_value=tips)
    else:
        patcher_tips = patch.object(analyzer.retrieval_service, "search_tips", return_value=[])

    patchers = [patcher_bm25, patcher_vector, patcher_tips]
    if rrf is not None:
        patchers.append(patch.object(analyzer.retrieval_service, "rrf_fusion", return_value=rrf))

    for p in patchers:
        p.start()

    def cleanup():
        for p in patchers:
            p.stop()

    return cleanup


class TestAnalyzeEndToEnd:
    """End-to-end tests for FraudAnalyzer.analyze() with real RRF + prompts."""

    @pytest.mark.asyncio
    async def test_high_risk_direct_hit_full_pipeline(self, mock_db, settings, fake_embedding):
        """
        Full pipeline: embedding -> BM25 + vector -> real RRF -> Direct_Hit.
        rrf_fusion is mocked with score > threshold to exercise Direct_Hit path.
        RRF + prompt building is still real (only threshold bypassed).
        """
        case = _make_case()
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            analyzer,
            bm25=[(case, 0.9)],
            vector=[(case, 0.95)],
            rrf=[{"item": case, "score": 0.9}],
        )
        try:
            response = await analyzer.analyze(
                "I got a call from police asking me to transfer money"
            )

            embedding_service.get_embeddings.assert_called_once()
            assert isinstance(response, AnalysisResponse)
            assert response.result_type == ResultType.DIRECT_HIT.value
            assert isinstance(response.data, DirectHitData)
            assert response.data.risk_level == RiskLevel.HIGH.value
            assert len(response.data.matched_cases) >= 1
            assert response.data.matched_cases[0].description == case.description
            assert response.data.recommended_action == "停止一切操作，立即报警"
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_medium_risk_rag_prompt_full_pipeline(self, mock_db, settings, fake_embedding):
        """
        Full pipeline: RRF score > 0.5 but < threshold -> MEDIUM -> RAG_Prompt.
        rrf_fusion is mocked with score=0.7 (0.5 < 0.7 < 0.85) to exercise MEDIUM path.
        Verifies real prompt text contains case + tip data from the pipeline.
        """
        case = _make_case()
        tip = _make_tip()
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            analyzer,
            tips=[tip],
            rrf=[{"item": case, "score": 0.7}],
        )
        try:
            response = await analyzer.analyze("Someone called me about my account")

            assert response.result_type == ResultType.RAG_PROMPT.value
            assert isinstance(response.data, RAGPromptData)
            assert response.data.risk_level == RiskLevel.MEDIUM.value
            assert response.data.rrf_score == 0.7
            assert "你是一个专业的反诈骗助手" in response.data.prompt
            assert case.description in response.data.prompt
            assert tip.title in response.data.prompt
            assert len(response.data.context.relevant_cases) >= 1
            assert len(response.data.context.anti_fraud_tips) == 1
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_low_risk_no_cases_full_pipeline(self, mock_db, settings, fake_embedding):
        """
        Full pipeline: no cases found -> LOW risk -> RAG_Prompt with tips only.
        """
        tip = _make_tip()
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(analyzer, tips=[tip])
        try:
            response = await analyzer.analyze("What's the weather today?")

            assert response.result_type == ResultType.RAG_PROMPT.value
            assert response.data.risk_level == RiskLevel.LOW.value
            assert response.data.rrf_score == 0.0
            assert "你是一个专业的反诈骗助手" in response.data.prompt
            assert tip.title in response.data.prompt
            assert response.data.context.relevant_cases == []
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_low_risk_score_below_half_full_pipeline(self, mock_db, settings, fake_embedding):
        """
        Full pipeline: only vector returns one result, normalized RRF score = 0.5 -> LOW.
        Real RRF math: (1/(k+0+1)) / (2/(k+1)) = 0.5, which is not > 0.5.
        """
        case = _make_case()
        tip = _make_tip()
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            analyzer,
            bm25=[],
            vector=[(case, 0.3)],
            tips=[tip],
        )
        try:
            response = await analyzer.analyze("maybe slightly suspicious")

            assert response.result_type == ResultType.RAG_PROMPT.value
            assert response.data.risk_level == RiskLevel.LOW.value
            assert response.data.rrf_score == pytest.approx(0.5)
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_embedding_failure_propagates(self, mock_db, settings):
        """Embedding failure -> EmbeddingError from analyze()."""
        embedding_service = MagicMock(spec=EmbeddingService)
        embedding_service.get_embeddings = AsyncMock(side_effect=EmbeddingError("API timeout"))
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        with pytest.raises(EmbeddingError, match="Failed to get embedding"):
            await analyzer.analyze("test query")


class TestAddCaseEndToEnd:
    """End-to-end tests for add_case with real Case model construction."""

    @pytest.mark.asyncio
    async def test_add_case_creates_real_case_object(self, mock_db, settings, fake_embedding):
        """add_case: real Case object with correct fields + embedding."""
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("12345678-1234-5678-1234-567812345678"))
        )

        await analyzer.add_case(
            description="Phone fraud case",
            fraud_type="phone_scam",
            amount=10000.0,
            keywords=["phone", "fraud"],
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()

        added_obj = mock_db.add.call_args[0][0]
        assert isinstance(added_obj, Case)
        assert added_obj.description == "Phone fraud case"
        assert added_obj.fraud_type == "phone_scam"
        assert added_obj.amount == 10000.0
        assert added_obj.keywords == ["phone", "fraud"]
        assert added_obj.embedding == fake_embedding

        embedding_service.get_embeddings.assert_called_once_with("Phone fraud case")

    @pytest.mark.asyncio
    async def test_add_case_minimal_fields(self, mock_db, settings, fake_embedding):
        """add_case: only required field -> None optionals."""
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("12345678-1234-5678-1234-567812345678"))
        )

        await analyzer.add_case(description="Simple case")

        added_obj = mock_db.add.call_args[0][0]
        assert isinstance(added_obj, Case)
        assert added_obj.description == "Simple case"
        assert added_obj.fraud_type is None
        assert added_obj.amount is None
        assert added_obj.keywords is None


class TestAddTipEndToEnd:
    """End-to-end tests for add_tip with real Tip model construction."""

    @pytest.mark.asyncio
    async def test_add_tip_concatenates_title_and_content(self, mock_db, settings, fake_embedding):
        """add_tip: embedding called with 'title content'."""
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("87654321-4321-8765-4321-876543218765"))
        )

        await analyzer.add_tip(
            title="Beware of Phishing",
            content="Never click suspicious links",
            category="phishing",
            keywords=["phishing", "links"],
        )

        added_obj = mock_db.add.call_args[0][0]
        assert isinstance(added_obj, Tip)
        assert added_obj.title == "Beware of Phishing"
        assert added_obj.content == "Never click suspicious links"
        assert added_obj.category == "phishing"
        assert added_obj.keywords == ["phishing", "links"]
        assert added_obj.embedding == fake_embedding

        embedding_service.get_embeddings.assert_called_once_with(
            "Beware of Phishing Never click suspicious links"
        )


class TestHybridSearchEndToEnd:
    """End-to-end tests for hybrid_search with real RRF fusion."""

    @pytest.mark.asyncio
    async def test_case_in_both_lists_ranks_highest(self, mock_db, settings, fake_embedding):
        """
        Case in both BM25 and vector should outrank case in only one.
        Real RRF: 1/(k+rank+1) + 1/(k+rank+1) vs 1/(k+rank+1).
        """
        case_a = _make_case(
            case_id="11111111-1111-1111-1111-111111111111",
            description="Case A: phone scam",
        )
        case_b = _make_case(
            case_id="22222222-2222-2222-2222-222222222222",
            description="Case B: email phishing",
        )

        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            analyzer,
            bm25=[(case_a, 0.9)],
            vector=[(case_a, 0.8), (case_b, 0.7)],
        )
        try:
            results = await analyzer.hybrid_search("phone scam", limit=10)

            assert len(results) == 2
            assert results[0]["case_id"] == case_a.id
            assert results[0]["rrf_score"] > results[1]["rrf_score"]
            assert results[0]["description"] == "Case A: phone scam"
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_db, settings, fake_embedding):
        """hybrid_search: no matches -> empty list."""
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(analyzer)
        try:
            results = await analyzer.hybrid_search("harmless query")
            assert results == []
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_limit_truncates_results(self, mock_db, settings, fake_embedding):
        """hybrid_search: limit=3 on 5 results -> only 3 returned."""
        cases = [
            _make_case(
                case_id=f"11111111-1111-1111-1111-{i:012d}",
                description=f"Case {i}",
            )
            for i in range(5)
        ]

        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            analyzer, vector=[(c, 0.9 - i * 0.1) for i, c in enumerate(cases)]
        )
        try:
            results = await analyzer.hybrid_search("scam", limit=3)
            assert len(results) == 3
        finally:
            cleanup()


class TestSearchSimilarCasesEndToEnd:
    """End-to-end tests for search_similar_cases."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_dicts(self, mock_db, settings, fake_embedding):
        """search_similar_cases: returns list of dicts with expected keys."""
        case = _make_case()
        embedding_service = _make_embedding_service(fake_embedding)
        analyzer = FraudAnalyzer(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(analyzer, vector=[(case, 0.95)])
        try:
            results = await analyzer.search_similar_cases("phone scam", limit=5)

            assert len(results) == 1
            assert results[0]["case_id"] == case.id
            assert results[0]["description"] == case.description
            assert results[0]["fraud_type"] == case.fraud_type
            assert results[0]["score"] == 0.95
        finally:
            cleanup()
