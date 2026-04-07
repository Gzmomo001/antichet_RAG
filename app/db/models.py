import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Column, Computed, DateTime, Index, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Case(Base):
    __tablename__ = "cases_table"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    description = Column(Text, nullable=False)
    fraud_type = Column(String(100))
    amount = Column(Numeric(15, 2))
    keywords = Column(ARRAY(String))
    embedding = Column(Vector(1536))  # Default dimension

    # TSVector for full-text search (BM25)
    # Note: Using 'simple' or 'english' as placeholder.
    # In production with zhparser, it would be 'zhparser'.
    content_tsv = Column(TSVECTOR, Computed("to_tsvector('english', description)", persisted=True))

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


class Tip(Base):
    __tablename__ = "tips_table"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100))
    keywords = Column(ARRAY(String))
    embedding = Column(Vector(1536))

    content_tsv = Column(
        TSVECTOR, Computed("to_tsvector('english', title || ' ' || content)", persisted=True)
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
