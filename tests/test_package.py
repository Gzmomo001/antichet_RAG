"""
Unit tests for antifraud_rag/__init__.py - Package exports.
"""

from antifraud_rag import (
    AnalysisResponse,
    AntiFraudRAG,
    Case,
    DirectHitData,
    MatchedCase,
    RAGPromptContext,
    RAGPromptData,
    Settings,
    Tip,
    __version__,
)


class TestPackageExports:
    def test_version_exists(self):
        assert __version__ == "1.0.0"

    def test_antifraud_rag_class_exported(self):
        assert AntiFraudRAG is not None

    def test_settings_class_exported(self):
        assert Settings is not None

    def test_case_model_exported(self):
        assert Case is not None

    def test_tip_model_exported(self):
        assert Tip is not None

    def test_schema_classes_exported(self):
        assert AnalysisResponse is not None
        assert DirectHitData is not None
        assert MatchedCase is not None
        assert RAGPromptContext is not None
        assert RAGPromptData is not None
