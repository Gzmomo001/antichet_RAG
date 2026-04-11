"""
Custom exceptions for Anti-Fraud RAG system.
"""


class AntiFraudError(RuntimeError):
    """Base exception for Anti-Fraud RAG system."""

    pass


class EmbeddingError(AntiFraudError):
    """Raised when embedding API fails."""

    pass


class DatabaseNotInitializedError(AntiFraudError):
    """Raised when database is not initialized."""

    pass
