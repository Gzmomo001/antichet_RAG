"""
Unit tests for antifraud_rag/services/prompts.py - Prompt building utilities.
"""

from unittest.mock import MagicMock
from uuid import UUID

from antifraud_rag.schemas import MatchedCase
from antifraud_rag.services.prompts import (
    build_matched_cases,
    build_rag_prompt,
    build_relevant_cases_data,
    build_tips_data,
)


class TestBuildMatchedCases:
    def test_returns_matched_case_objects(self):
        mock_case = MagicMock()
        mock_case.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_case.description = "Phone scam case"
        mock_case.fraud_type = "phone_scam"
        mock_case.keywords = ["urgent", "transfer"]

        fused_results = [{"item": mock_case, "score": 0.95}]
        result = build_matched_cases(fused_results)

        assert len(result) == 1
        assert isinstance(result[0], MatchedCase)
        assert result[0].case_id == mock_case.id
        assert result[0].description == "Phone scam case"
        assert result[0].confidence == 0.95
        assert result[0].fraud_type == "phone_scam"
        assert result[0].key_indicators == ["urgent", "transfer"]

    def test_filters_by_min_score(self):
        mock_case = MagicMock()
        mock_case.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_case.description = "Low score case"
        mock_case.fraud_type = None
        mock_case.keywords = None

        fused_results = [{"item": mock_case, "score": 0.05}]
        result = build_matched_cases(fused_results, min_score=0.1)

        assert len(result) == 0

    def test_limits_to_top_three(self):
        fused_results = []
        for i in range(5):
            mock_case = MagicMock()
            mock_case.id = UUID(f"12345678-1234-5678-1234-{i:012d}")
            mock_case.description = f"Case {i}"
            mock_case.fraud_type = "scam"
            mock_case.keywords = []
            fused_results.append({"item": mock_case, "score": 0.9 - i * 0.1})

        result = build_matched_cases(fused_results)
        assert len(result) == 3

    def test_empty_input(self):
        result = build_matched_cases([])
        assert result == []

    def test_keywords_none_uses_empty_list(self):
        mock_case = MagicMock()
        mock_case.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_case.description = "Case"
        mock_case.fraud_type = None
        mock_case.keywords = None

        fused_results = [{"item": mock_case, "score": 0.9}]
        result = build_matched_cases(fused_results)

        assert result[0].key_indicators == []


class TestBuildRelevantCasesData:
    def test_builds_dict_list(self):
        mock_case = MagicMock()
        mock_case.description = "Scam description"
        mock_case.fraud_type = "phishing"

        fused_results = [{"item": mock_case, "score": 0.8}]
        result = build_relevant_cases_data(fused_results)

        assert len(result) == 1
        assert result[0] == {"description": "Scam description", "fraud_type": "phishing"}

    def test_limits_to_top_three(self):
        fused_results = []
        for i in range(5):
            mock_case = MagicMock()
            mock_case.description = f"Case {i}"
            mock_case.fraud_type = f"type_{i}"
            fused_results.append({"item": mock_case, "score": 0.5})

        result = build_relevant_cases_data(fused_results)
        assert len(result) == 3

    def test_empty_input(self):
        result = build_relevant_cases_data([])
        assert result == []


class TestBuildTipsData:
    def test_builds_dict_list_from_tips(self):
        tip1 = MagicMock()
        tip1.title = "Tip 1"
        tip1.content = "Content 1"
        tip2 = MagicMock()
        tip2.title = "Tip 2"
        tip2.content = "Content 2"

        result = build_tips_data([tip1, tip2])
        assert len(result) == 2
        assert result[0] == {"title": "Tip 1", "content": "Content 1"}
        assert result[1] == {"title": "Tip 2", "content": "Content 2"}

    def test_empty_input(self):
        result = build_tips_data([])
        assert result == []


class TestBuildRAGPrompt:
    def test_prompt_contains_all_sections(self):
        relevant_cases = [{"description": "Case about phone scam"}]
        tips = [{"title": "Warning", "content": "Do not share OTP"}]

        prompt = build_rag_prompt("可疑短信", relevant_cases, tips)

        assert "可疑短信" in prompt
        assert "Case about phone scam" in prompt
        assert "Warning: Do not share OTP" in prompt
        assert "反诈骗助手" in prompt

    def test_prompt_with_empty_cases_and_tips(self):
        prompt = build_rag_prompt("test query", [], [])

        assert "test query" in prompt
        assert "相关案例" in prompt
        assert "反诈知识" in prompt

    def test_prompt_with_multiple_cases_and_tips(self):
        cases = [
            {"description": "Case 1"},
            {"description": "Case 2"},
        ]
        tips = [
            {"title": "Tip A", "content": "Advice A"},
            {"title": "Tip B", "content": "Advice B"},
        ]

        prompt = build_rag_prompt("query", cases, tips)

        assert "Case 1" in prompt
        assert "Case 2" in prompt
        assert "Tip A: Advice A" in prompt
        assert "Tip B: Advice B" in prompt
