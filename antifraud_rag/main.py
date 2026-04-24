"""Backward-compatible exports for the historical main module path."""

from antifraud_rag.analyzer import AntiFraudRAG, FraudAnalyzer

__all__ = ["FraudAnalyzer", "AntiFraudRAG"]
