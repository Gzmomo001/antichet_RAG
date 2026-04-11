"""
Prompt building utilities for Anti-Fraud RAG system.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from antifraud_rag.core.constants import MIN_MATCH_SCORE, TOP_RESULTS_COUNT
from antifraud_rag.schemas import MatchedCase

if TYPE_CHECKING:
    from antifraud_rag.db.models import Tip


def build_matched_cases(
    fused_results: List[Dict[str, Any]], min_score: float = MIN_MATCH_SCORE
) -> List[MatchedCase]:
    """
    Build a list of MatchedCase objects from fused search results.
    """
    matched_cases = []
    for res in fused_results[:TOP_RESULTS_COUNT]:
        if res["score"] > min_score:
            case = res["item"]
            matched_cases.append(
                MatchedCase(
                    case_id=case.id,
                    description=case.description,
                    confidence=res["score"],
                    fraud_type=case.fraud_type,
                    key_indicators=case.keywords or [],
                )
            )
    return matched_cases


def build_relevant_cases_data(
    fused_results: List[Dict[str, Any]],
    limit: int = TOP_RESULTS_COUNT,
) -> List[Dict[str, Any]]:
    """
    Build a list of relevant case data dictionaries from fused results.
    """
    return [
        {"description": res["item"].description, "fraud_type": res["item"].fraud_type}
        for res in fused_results[:limit]
    ]


def build_tips_data(tips: List["Tip"]) -> List[Dict[str, Any]]:
    """
    Build a list of tip data dictionaries from Tip objects.
    """
    return [{"title": t.title, "content": t.content} for t in tips]


def build_rag_prompt(
    text: str,
    relevant_cases: List[Dict[str, Any]],
    tips: List[Dict[str, Any]],
) -> str:
    """
    Build a RAG prompt for the anti-fraud analysis.
    """
    cases_text = "\n".join([f"- {c['description']}" for c in relevant_cases])
    tips_text = "\n".join([f"- {t['title']}: {t['content']}" for t in tips])

    return f"""你是一个专业的反诈骗助手。请根据以下案例信息和反诈知识，
分析用户遇到的情况是否属于诈骗，并给出专业的判断和建议。

【用户咨询】
{text}

【相关案例】
{cases_text}

【反诈知识】
{tips_text}

请给出你的分析："""
