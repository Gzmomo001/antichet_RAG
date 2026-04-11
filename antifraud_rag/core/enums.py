"""
Enumerations for Anti-Fraud RAG system.
"""

from enum import Enum


class ResultType(str, Enum):
    DIRECT_HIT = "Direct_Hit"
    RAG_PROMPT = "RAG_Prompt"


class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
