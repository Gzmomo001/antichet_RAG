"""
Unit tests for app/schemas/data.py - Pydantic models/schemas.
"""

from uuid import UUID

import pytest
from pydantic import ValidationError

from app.schemas.data import (
    AnalysisRequest,
    AnalysisRequestBody,
    AnalysisRequestMetadata,
    AnalysisResponse,
    CaseCreate,
    CaseCreateRequest,
    DirectHitData,
    MatchedCase,
    RAGPromptContext,
    RAGPromptData,
    TipCreate,
    TipCreateRequest,
)


class TestAnalysisRequestMetadata:
    """Tests for AnalysisRequestMetadata schema."""

    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = AnalysisRequestMetadata(user_id="user123", channel="mobile")
        assert metadata.user_id == "user123"
        assert metadata.channel == "mobile"

    def test_metadata_optional_fields(self):
        """Test metadata with optional fields omitted."""
        metadata = AnalysisRequestMetadata()
        assert metadata.user_id is None
        assert metadata.channel == "web"  # default

    def test_metadata_default_channel(self):
        """Test metadata channel has correct default."""
        metadata = AnalysisRequestMetadata(user_id="user456")
        assert metadata.channel == "web"


class TestAnalysisRequest:
    """Tests for AnalysisRequest schema."""

    def test_valid_request(self):
        """Test valid analysis request."""
        request = AnalysisRequest(
            text="Test fraud text",
            source="api",
            metadata=AnalysisRequestMetadata(user_id="user123"),
        )
        assert request.text == "Test fraud text"
        assert request.source == "api"
        assert request.metadata.user_id == "user123"

    def test_request_required_text_field(self):
        """Test analysis request requires text field."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest()
        assert any(err["loc"] == ("text",) for err in exc_info.value.errors())

    def test_request_optional_metadata(self):
        """Test analysis request with optional metadata."""
        request = AnalysisRequest(text="Test text")
        assert request.metadata is None
        assert request.source == "user_submission"  # default


class TestAnalysisRequestBody:
    """Tests for AnalysisRequestBody schema."""

    def test_valid_body(self):
        """Test valid request body."""
        body = AnalysisRequestBody(request=AnalysisRequest(text="Test text"))
        assert body.request.text == "Test text"

    def test_body_requires_request(self):
        """Test body requires request field."""
        with pytest.raises(ValidationError):
            AnalysisRequestBody()


class TestMatchedCase:
    """Tests for MatchedCase schema."""

    def test_valid_matched_case(self):
        """Test valid matched case."""
        case = MatchedCase(
            case_id=UUID("12345678-1234-5678-1234-567812345678"),
            description="Fraud case description",
            confidence=0.95,
            fraud_type="phone_scam",
            key_indicators=["urgent", "money"],
        )
        assert case.confidence == 0.95
        assert case.fraud_type == "phone_scam"

    def test_matched_case_defaults(self):
        """Test matched case default values."""
        case = MatchedCase(
            case_id=UUID("12345678-1234-5678-1234-567812345678"),
            description="Description",
            confidence=0.5,
        )
        assert case.fraud_type is None
        assert case.key_indicators == []

    def test_matched_case_requires_fields(self):
        """Test matched case requires case_id, description, and confidence."""
        with pytest.raises(ValidationError):
            MatchedCase(description="Test")


class TestDirectHitData:
    """Tests for DirectHitData schema."""

    def test_valid_direct_hit_data(self):
        """Test valid direct hit data."""
        case = MatchedCase(
            case_id=UUID("12345678-1234-5678-1234-567812345678"),
            description="Test",
            confidence=0.9,
        )
        data = DirectHitData(
            risk_level="HIGH",
            matched_cases=[case],
            recommended_action="报警",
        )
        assert data.risk_level == "HIGH"
        assert len(data.matched_cases) == 1

    def test_direct_hit_defaults(self):
        """Test direct hit data default values."""
        case = MatchedCase(
            case_id=UUID("12345678-1234-5678-1234-567812345678"),
            description="Test",
            confidence=0.9,
        )
        data = DirectHitData(matched_cases=[case])
        assert data.risk_level == "HIGH"
        assert data.recommended_action == "停止一切操作，立即报警"


class TestRAGPromptContext:
    """Tests for RAGPromptContext schema."""

    def test_valid_context(self):
        """Test valid RAG prompt context."""
        context = RAGPromptContext(
            relevant_cases=[{"description": "Case 1", "fraud_type": "scam"}],
            anti_fraud_tips=[{"title": "Tip 1", "content": "Be careful"}],
        )
        assert len(context.relevant_cases) == 1
        assert len(context.anti_fraud_tips) == 1

    def test_context_empty_lists(self):
        """Test context with empty lists."""
        context = RAGPromptContext(relevant_cases=[], anti_fraud_tips=[])
        assert context.relevant_cases == []
        assert context.anti_fraud_tips == []


class TestRAGPromptData:
    """Tests for RAGPromptData schema."""

    def test_valid_rag_data(self):
        """Test valid RAG prompt data."""
        data = RAGPromptData(
            risk_level="MEDIUM",
            rrf_score=0.75,
            prompt="分析这个请求",
            context=RAGPromptContext(relevant_cases=[], anti_fraud_tips=[]),
        )
        assert data.risk_level == "MEDIUM"
        assert data.rrf_score == 0.75

    def test_rag_data_defaults(self):
        """Test RAG data default values."""
        data = RAGPromptData(
            rrf_score=0.5,
            prompt="Test prompt",
            context=RAGPromptContext(relevant_cases=[], anti_fraud_tips=[]),
        )
        assert data.risk_level == "MEDIUM"


class TestAnalysisResponse:
    """Tests for AnalysisResponse schema."""

    def test_direct_hit_response(self):
        """Test AnalysisResponse with DirectHit data."""
        case = MatchedCase(
            case_id=UUID("12345678-1234-5678-1234-567812345678"),
            description="Test",
            confidence=0.9,
        )
        response = AnalysisResponse(
            status="success",
            result_type="Direct_Hit",
            data=DirectHitData(matched_cases=[case]),
        )
        assert response.status == "success"
        assert response.result_type == "Direct_Hit"

    def test_rag_prompt_response(self):
        """Test AnalysisResponse with RAG prompt data."""
        response = AnalysisResponse(
            status="success",
            result_type="RAG_Prompt",
            data=RAGPromptData(
                rrf_score=0.6,
                prompt="分析",
                context=RAGPromptContext(relevant_cases=[], anti_fraud_tips=[]),
            ),
        )
        assert response.result_type == "RAG_Prompt"

    def test_response_default_status(self):
        """Test response has default status."""
        response = AnalysisResponse(
            result_type="Direct_Hit",
            data=DirectHitData(matched_cases=[]),
        )
        assert response.status == "success"


class TestCaseCreate:
    """Tests for CaseCreate schema."""

    def test_valid_case_create(self):
        """Test valid case creation."""
        case = CaseCreate(
            description="New fraud case",
            fraud_type="investment_scam",
            amount=5000.00,
            keywords=["investment", "high_return"],
        )
        assert case.description == "New fraud case"
        assert case.amount == 5000.00

    def test_case_create_minimal(self):
        """Test case creation with minimal fields."""
        case = CaseCreate(description="Minimal case")
        assert case.description == "Minimal case"
        assert case.fraud_type is None
        assert case.amount is None
        assert case.keywords == []

    def test_case_create_requires_description(self):
        """Test case creation requires description."""
        with pytest.raises(ValidationError):
            CaseCreate()


class TestCaseCreateRequest:
    """Tests for CaseCreateRequest schema."""

    def test_valid_case_request(self):
        """Test valid case create request."""
        request = CaseCreateRequest(case=CaseCreate(description="Test case"))
        assert request.case.description == "Test case"


class TestTipCreate:
    """Tests for TipCreate schema."""

    def test_valid_tip_create(self):
        """Test valid tip creation."""
        tip = TipCreate(
            title="Warning Title",
            content="Warning content",
            category="fraud",
            keywords=["warning", "alert"],
        )
        assert tip.title == "Warning Title"
        assert tip.category == "fraud"

    def test_tip_create_minimal(self):
        """Test tip creation with minimal fields."""
        tip = TipCreate(title="Title", content="Content")
        assert tip.title == "Title"
        assert tip.content == "Content"
        assert tip.category is None
        assert tip.keywords == []

    def test_tip_create_requires_title_and_content(self):
        """Test tip creation requires title and content."""
        with pytest.raises(ValidationError):
            TipCreate(title="Only title")

        with pytest.raises(ValidationError):
            TipCreate(content="Only content")


class TestTipCreateRequest:
    """Tests for TipCreateRequest schema."""

    def test_valid_tip_request(self):
        """Test valid tip create request."""
        request = TipCreateRequest(tip=TipCreate(title="Test", content="Test content"))
        assert request.tip.title == "Test"
