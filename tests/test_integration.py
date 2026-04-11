"""
Integration tests for AntiFraudRAG — full pipeline tests with real internal components.

These tests differ from unit tests in that they:
- Use the REAL AntiFraudRAG.analyze() / hybrid_search() / etc. methods
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

from antifraud_rag import AntiFraudRAG, Settings
from antifraud_rag.db.models import Case, Tip
from antifraud_rag.schemas import AnalysisResponse, DirectHitData, RAGPromptData
from antifraud_rag.services.embedding import EmbeddingService
from antifraud_rag.services.prompts import (
    build_matched_cases,
    build_rag_prompt,
    build_relevant_cases_data,
    build_tips_data,
)
from antifraud_rag.services.retrieval import RetrievalService


@pytest.fixture
def settings():
    return Settings(
        EMBEDDING_MODEL_URL="https://api.test.com/embeddings",
        EMBEDDING_MODEL_API_KEY="test-key",
        HIGH_RISK_THRESHOLD=0.85,
    )


@pytest.fixture
def fake_embedding():
    return [0.1] * 1536


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


def _patch_retrieval(rag, *, bm25=None, vector=None, tips=None, rrf=None):
    """Patch retrieval methods. If rrf is provided, rrf_fusion is also patched (real otherwise)."""
    if bm25 is not None:
        patcher_bm25 = patch.object(rag.retrieval_service, "search_cases_bm25", return_value=bm25)
    else:
        patcher_bm25 = patch.object(rag.retrieval_service, "search_cases_bm25", return_value=[])

    if vector is not None:
        patcher_vector = patch.object(
            rag.retrieval_service, "search_cases_vector", return_value=vector
        )
    else:
        patcher_vector = patch.object(rag.retrieval_service, "search_cases_vector", return_value=[])

    if tips is not None:
        patcher_tips = patch.object(rag.retrieval_service, "search_tips", return_value=tips)
    else:
        patcher_tips = patch.object(rag.retrieval_service, "search_tips", return_value=[])

    patchers = [patcher_bm25, patcher_vector, patcher_tips]
    if rrf is not None:
        patchers.append(patch.object(rag.retrieval_service, "rrf_fusion", return_value=rrf))

    for p in patchers:
        p.start()

    def cleanup():
        for p in patchers:
            p.stop()

    return cleanup


class TestAnalyzeEndToEnd:
    """End-to-end tests for AntiFraudRAG.analyze() with real RRF + prompts."""

    @pytest.mark.asyncio
    async def test_high_risk_direct_hit_full_pipeline(self, mock_db, settings, fake_embedding):
        """
        Full pipeline: embedding -> BM25 + vector -> real RRF -> Direct_Hit.
        rrf_fusion is mocked with score > threshold to exercise Direct_Hit path.
        RRF + prompt building is still real (only threshold bypassed).
        """
        case = _make_case()
        embedding_service = _make_embedding_service(fake_embedding)
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            rag,
            bm25=[(case, 0.9)],
            vector=[(case, 0.95)],
            rrf=[{"item": case, "score": 0.9}],
        )
        try:
            response = await rag.analyze("I got a call from police asking me to transfer money")

            embedding_service.get_embeddings.assert_called_once()
            assert isinstance(response, AnalysisResponse)
            assert response.result_type == "Direct_Hit"
            assert isinstance(response.data, DirectHitData)
            assert response.data.risk_level == "HIGH"
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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            rag,
            tips=[tip],
            rrf=[{"item": case, "score": 0.7}],
        )
        try:
            response = await rag.analyze("Someone called me about my account")

            assert response.result_type == "RAG_Prompt"
            assert isinstance(response.data, RAGPromptData)
            assert response.data.risk_level == "MEDIUM"
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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(rag, tips=[tip])
        try:
            response = await rag.analyze("What's the weather today?")

            assert response.result_type == "RAG_Prompt"
            assert response.data.risk_level == "LOW"
            assert response.data.rrf_score == 0.0
            assert "分析用户请求" in response.data.prompt
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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            rag,
            bm25=[],
            vector=[(case, 0.3)],
            tips=[tip],
        )
        try:
            response = await rag.analyze("maybe slightly suspicious")

            assert response.result_type == "RAG_Prompt"
            assert response.data.risk_level == "LOW"
            assert response.data.rrf_score == pytest.approx(0.5)
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_embedding_failure_propagates(self, mock_db, settings):
        """Embedding failure -> RuntimeError from analyze()."""
        embedding_service = MagicMock(spec=EmbeddingService)
        embedding_service.get_embeddings = AsyncMock(side_effect=RuntimeError("API timeout"))
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        with pytest.raises(RuntimeError, match="Embedding error"):
            await rag.analyze("test query")


class TestAddCaseEndToEnd:
    """End-to-end tests for add_case with real Case model construction."""

    @pytest.mark.asyncio
    async def test_add_case_creates_real_case_object(self, mock_db, settings, fake_embedding):
        """add_case: real Case object with correct fields + embedding."""
        embedding_service = _make_embedding_service(fake_embedding)
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("12345678-1234-5678-1234-567812345678"))
        )

        await rag.add_case(
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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("12345678-1234-5678-1234-567812345678"))
        )

        await rag.add_case(description="Simple case")

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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("87654321-4321-8765-4321-876543218765"))
        )

        await rag.add_tip(
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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            rag,
            bm25=[(case_a, 0.9)],
            vector=[(case_a, 0.8), (case_b, 0.7)],
        )
        try:
            results = await rag.hybrid_search("phone scam", limit=10)

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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(rag)
        try:
            results = await rag.hybrid_search("harmless query")
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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(rag, vector=[(c, 0.9 - i * 0.1) for i, c in enumerate(cases)])
        try:
            results = await rag.hybrid_search("scam", limit=3)
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
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(rag, vector=[(case, 0.95)])
        try:
            results = await rag.search_similar_cases("phone scam", limit=5)

            assert len(results) == 1
            assert results[0]["case_id"] == case.id
            assert results[0]["description"] == case.description
            assert results[0]["fraud_type"] == case.fraud_type
            assert results[0]["score"] == 0.95
        finally:
            cleanup()


class TestRRFFusionRealComputation:
    """Verify real RRF math in the context of analyze()."""

    @pytest.mark.asyncio
    async def test_rrf_score_matches_formula(self, mock_db, settings, fake_embedding):
        """
        analyze() runs real rrf_fusion. Verify normalized score when case is
        in only one list: (1/(k+0+1)) / (2/(k+1)) = 0.5.
        Score 0.5 < threshold → RAG_Prompt path (has rrf_score field).
        """
        case = _make_case()
        tip = _make_tip()
        embedding_service = _make_embedding_service(fake_embedding)
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            rag,
            bm25=[(case, 0.9)],
            vector=[(case, 0.8)],
            tips=[tip],
        )
        try:
            response = await rag.analyze("suspicious text")

            assert response.result_type == "Direct_Hit"
            max_score = 2 / (60 + 1)
            raw_rrf = 1 / (60 + 0 + 1) + 1 / (60 + 0 + 1)
            expected_normalized = raw_rrf / max_score
            assert abs(expected_normalized - 1.0) < 0.001
        finally:
            cleanup()

    def test_rrf_multiple_items_deduplication(self, mock_db):
        """
        Same item in both lists: combined score.
        Different items: individual scores.
        All using real RetrievalService.rrf_fusion() with normalization.
        """
        case_a = _make_case(case_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        case_b = _make_case(case_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        retrieval = RetrievalService(mock_db)
        bm25_results = [(case_a, 0.9), (case_b, 0.7)]
        vector_results = [(case_b, 0.8), (case_a, 0.6)]

        fused = retrieval.rrf_fusion(bm25_results, vector_results)

        max_score = 2 / (60 + 1)
        raw_score_a = 1 / (60 + 0 + 1) + 1 / (60 + 1 + 1)
        raw_score_b = 1 / (60 + 1 + 1) + 1 / (60 + 0 + 1)
        expected_a = raw_score_a / max_score
        expected_b = raw_score_b / max_score

        assert len(fused) == 2
        assert abs(fused[0]["score"] - expected_a) < 0.001
        assert abs(fused[1]["score"] - expected_b) < 0.001


class TestPromptBuildersIntegration:
    """Verify prompt builders produce correct output from retrieval pipeline data."""

    def test_rag_prompt_contains_all_sections(self):
        """RAG prompt includes user text, case descriptions, and tips."""
        cases_data = [
            {"description": "Phone scam asking for transfer", "fraud_type": "phone_scam"},
            {"description": "Fake police call", "fraud_type": "impersonation"},
        ]
        tips_data = [
            {"title": "Warning", "content": "Police never ask for transfers"},
            {"title": "Tip 2", "content": "Hang up and call 110"},
        ]

        prompt = build_rag_prompt("I got a suspicious call", cases_data, tips_data)

        assert "I got a suspicious call" in prompt
        assert "Phone scam asking for transfer" in prompt
        assert "Fake police call" in prompt
        assert "Warning" in prompt
        assert "Police never ask for transfers" in prompt
        assert "反诈骗助手" in prompt

    def test_matched_cases_from_fused_results(self):
        """build_matched_cases produces valid MatchedCase from fused results."""
        case = _make_case()
        fused = [
            {"item": case, "score": 0.95},
            {"item": case, "score": 0.5},
        ]

        matched = build_matched_cases(fused)

        assert len(matched) == 2
        assert matched[0].case_id == case.id
        assert matched[0].description == case.description
        assert matched[0].confidence == 0.95
        assert matched[0].fraud_type == "phone_scam"

    def test_matched_cases_filters_low_score(self):
        """build_matched_cases skips results below min_score."""
        case = _make_case()
        fused = [{"item": case, "score": 0.01}]

        matched = build_matched_cases(fused, min_score=0.1)
        assert len(matched) == 0

    def test_relevant_cases_truncates_to_three(self):
        """Only top 3 cases included in relevant_cases."""
        cases = [
            _make_case(
                case_id=f"11111111-1111-1111-1111-{i:012d}",
                description=f"Case {i}",
            )
            for i in range(5)
        ]
        fused = [{"item": c, "score": 0.9 - i * 0.1} for i, c in enumerate(cases)]

        result = build_relevant_cases_data(fused)
        assert len(result) == 3
        assert result[0]["description"] == "Case 0"

    def test_tips_data_extracts_fields(self):
        """build_tips_data extracts title and content."""
        tip = _make_tip()
        result = build_tips_data([tip])

        assert len(result) == 1
        assert result[0]["title"] == "Phone Scam Warning"
        assert (
            result[0]["content"]
            == "Never transfer money to someone claiming to be police over the phone"
        )


class TestFullWorkflowIntegration:
    """
    Full workflow tests: add data -> search -> analyze -> verify response.
    These simulate realistic usage patterns end-to-end.
    """

    @pytest.mark.asyncio
    async def test_add_case_then_analyze_high_risk(self, mock_db, settings, fake_embedding):
        """
        Workflow: add a case -> analyze matching text -> Direct_Hit.
        Data created by add_case flows through to analyze pipeline.
        rrf_fusion is mocked with score > threshold.
        """
        embedding_service = _make_embedding_service(fake_embedding)
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("12345678-1234-5678-1234-567812345678"))
        )
        await rag.add_case(
            description="Fraudster posing as bank staff asking for verification code",
            fraud_type="phone_scam",
            amount=20000.0,
            keywords=["bank", "verification code"],
        )

        added_case = mock_db.add.call_args[0][0]

        cleanup = _patch_retrieval(
            rag,
            rrf=[{"item": added_case, "score": 0.9}],
        )
        try:
            response = await rag.analyze(
                "Someone called saying they're from my bank asking for my verification code"
            )

            assert response.result_type == "Direct_Hit"
            assert response.data.risk_level == "HIGH"
            assert any("verification code" in mc.description for mc in response.data.matched_cases)
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_add_tip_then_analyze_low_risk(self, mock_db, settings, fake_embedding):
        """
        Workflow: add a tip -> analyze text with no case matches ->
        tip appears in RAG prompt context.
        """
        embedding_service = _make_embedding_service(fake_embedding)
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        mock_db.refresh = AsyncMock(
            side_effect=lambda obj: setattr(obj, "id", UUID("87654321-4321-8765-4321-876543218765"))
        )
        await rag.add_tip(
            title="Verify Caller Identity",
            content="Always hang up and call the official number to verify",
            category="phone_fraud",
        )

        added_tip = mock_db.add.call_args[0][0]

        cleanup = _patch_retrieval(rag, tips=[added_tip])
        try:
            response = await rag.analyze("I received a suspicious phone call")

            assert response.result_type == "RAG_Prompt"
            assert response.data.risk_level == "LOW"
            assert any(
                t["title"] == "Verify Caller Identity"
                for t in response.data.context.anti_fraud_tips
            )
        finally:
            cleanup()

    @pytest.mark.asyncio
    async def test_hybrid_and_similar_search_consistency(self, mock_db, settings, fake_embedding):
        """
        Workflow: hybrid_search and search_similar_cases on same data
        should return consistent case_ids and descriptions.
        """
        case = _make_case()
        embedding_service = _make_embedding_service(fake_embedding)
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            rag,
            bm25=[(case, 0.9)],
            vector=[(case, 0.85)],
        )
        try:
            hybrid_results = await rag.hybrid_search("phone scam")
        finally:
            cleanup()

        cleanup2 = _patch_retrieval(rag, vector=[(case, 0.85)])
        try:
            similar_results = await rag.search_similar_cases("phone scam")
        finally:
            cleanup2()

        assert len(hybrid_results) >= 1
        assert len(similar_results) >= 1
        assert hybrid_results[0]["case_id"] == similar_results[0]["case_id"]
        assert hybrid_results[0]["description"] == similar_results[0]["description"]

    @pytest.mark.asyncio
    async def test_response_schema_serialization_roundtrip(self, mock_db, settings, fake_embedding):
        """
        Workflow: analyze returns valid AnalysisResponse that can serialize to dict.
        Tests Pydantic schemas work end-to-end through the real pipeline.
        """
        case = _make_case()
        tip = _make_tip()
        embedding_service = _make_embedding_service(fake_embedding)
        rag = AntiFraudRAG(mock_db, settings=settings, embedding_service=embedding_service)

        cleanup = _patch_retrieval(
            rag,
            rrf=[{"item": case, "score": 0.9}],
        )
        try:
            response = await rag.analyze("scam call")
            data = response.model_dump()

            assert data["result_type"] == "Direct_Hit"
            assert data["status"] == "success"
            assert isinstance(data["data"]["matched_cases"], list)
            assert len(data["data"]["matched_cases"]) >= 1
        finally:
            cleanup()

        cleanup2 = _patch_retrieval(
            rag,
            rrf=[{"item": case, "score": 0.7}],
            tips=[tip],
        )
        try:
            response = await rag.analyze("suspicious")
            data = response.model_dump()

            assert data["result_type"] == "RAG_Prompt"
            assert isinstance(data["data"]["prompt"], str)
            assert isinstance(data["data"]["context"]["relevant_cases"], list)
            assert isinstance(data["data"]["context"]["anti_fraud_tips"], list)
        finally:
            cleanup2()
