import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Column, Computed, DateTime, Index, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import declarative_base

from antifraud_rag.core.constants import EMBEDDING_DIMENSION as DEFAULT_EMBEDDING_DIMENSION


@dataclass(frozen=True)
class ModelRegistry:
    """A set of SQLAlchemy models bound to a specific embedding dimension."""

    base: Any
    case_model: Any
    tip_model: Any
    embedding_dimension: int


@lru_cache(maxsize=None)
def get_model_registry(
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> ModelRegistry:
    """Return isolated SQLAlchemy models for a specific embedding dimension."""
    base = declarative_base()

    class Case(base):
        __tablename__ = "cases_table"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        description = Column(Text, nullable=False)
        fraud_type = Column(String(100))
        amount = Column(Numeric(15, 2))
        keywords = Column(ARRAY(String))
        embedding = Column(Vector(embedding_dimension))

        # TSVector for full-text search (BM25)
        content_tsv = Column(
            TSVECTOR,
            Computed("to_tsvector('english', description)", persisted=True),
        )

        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), onupdate=func.now())

        __table_args__ = (
            Index(
                "idx_cases_embedding",
                "embedding",
                postgresql_using="ivfflat",
                postgresql_with={"lists": 100},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
            Index("idx_cases_tsv", "content_tsv", postgresql_using="gin"),
        )

    class Tip(base):
        __tablename__ = "tips_table"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        title = Column(String(500), nullable=False)
        content = Column(Text, nullable=False)
        category = Column(String(100))
        keywords = Column(ARRAY(String))
        embedding = Column(Vector(embedding_dimension))

        content_tsv = Column(
            TSVECTOR,
            Computed("to_tsvector('english', title || ' ' || content)", persisted=True),
        )

        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), onupdate=func.now())

        __table_args__ = (
            Index(
                "idx_tips_embedding",
                "embedding",
                postgresql_using="ivfflat",
                postgresql_with={"lists": 50},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
            Index("idx_tips_tsv", "content_tsv", postgresql_using="gin"),
        )

    return ModelRegistry(
        base=base,
        case_model=Case,
        tip_model=Tip,
        embedding_dimension=embedding_dimension,
    )


_DEFAULT_MODEL_REGISTRY = get_model_registry(DEFAULT_EMBEDDING_DIMENSION)
Base = _DEFAULT_MODEL_REGISTRY.base
Case = _DEFAULT_MODEL_REGISTRY.case_model
Tip = _DEFAULT_MODEL_REGISTRY.tip_model


def configure_embedding_dimension(dimension: int) -> ModelRegistry:
    """Return isolated models configured for the requested embedding dimension."""
    return get_model_registry(dimension)


def get_embedding_dimension(model: Any = Case) -> int:
    """Return the embedding dimension for a specific model class."""
    return model.__table__.columns["embedding"].type.dim
