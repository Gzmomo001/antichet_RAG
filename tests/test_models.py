"""
Unit tests for antifraud_rag/db/models.py - SQLAlchemy models.
"""

from antifraud_rag.core.constants import EMBEDDING_DIMENSION as DEFAULT_EMBEDDING_DIMENSION
from antifraud_rag.db.models import (
    Base,
    Case,
    Tip,
    configure_embedding_dimension,
    get_embedding_dimension,
)


class TestCaseModel:
    def test_case_tablename(self):
        assert Case.__tablename__ == "cases_table"

    def test_case_has_required_columns(self):
        columns = {c.name for c in Case.__table__.columns}
        assert "id" in columns
        assert "description" in columns
        assert "fraud_type" in columns
        assert "amount" in columns
        assert "keywords" in columns
        assert "embedding" in columns
        assert "content_tsv" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_case_id_default_uuid(self):
        col = Case.__table__.columns["id"]
        assert col.default is not None

    def test_case_description_not_nullable(self):
        col = Case.__table__.columns["description"]
        assert not col.nullable

    def test_case_table_args_has_indexes(self):
        index_names = {idx.name for idx in Case.__table__.indexes}
        assert "idx_cases_embedding" in index_names
        assert "idx_cases_tsv" in index_names

    def test_case_instantiation(self):
        case = Case(
            description="Test fraud",
            fraud_type="phishing",
            amount=500.00,
            keywords=["test"],
        )
        assert case.description == "Test fraud"
        assert case.fraud_type == "phishing"
        assert case.amount == 500.00
        assert case.keywords == ["test"]

    def test_case_inherits_from_base(self):
        assert issubclass(Case, Base)

    def test_case_embedding_uses_default_dimension(self):
        assert Case.__table__.columns["embedding"].type.dim == DEFAULT_EMBEDDING_DIMENSION


class TestTipModel:
    def test_tip_tablename(self):
        assert Tip.__tablename__ == "tips_table"

    def test_tip_has_required_columns(self):
        columns = {c.name for c in Tip.__table__.columns}
        assert "id" in columns
        assert "title" in columns
        assert "content" in columns
        assert "category" in columns
        assert "keywords" in columns
        assert "embedding" in columns
        assert "content_tsv" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_tip_title_not_nullable(self):
        col = Tip.__table__.columns["title"]
        assert not col.nullable

    def test_tip_content_not_nullable(self):
        col = Tip.__table__.columns["content"]
        assert not col.nullable

    def test_tip_table_args_has_indexes(self):
        index_names = {idx.name for idx in Tip.__table__.indexes}
        assert "idx_tips_embedding" in index_names
        assert "idx_tips_tsv" in index_names

    def test_tip_instantiation(self):
        tip = Tip(
            title="Test Tip",
            content="Test content",
            category="phone_fraud",
            keywords=["test"],
        )
        assert tip.title == "Test Tip"
        assert tip.content == "Test content"
        assert tip.category == "phone_fraud"

    def test_tip_inherits_from_base(self):
        assert issubclass(Tip, Base)

    def test_tip_embedding_dimension_can_be_reconfigured(self):
        model_registry = configure_embedding_dimension(2048)

        assert get_embedding_dimension(model_registry.case_model) == 2048
        assert get_embedding_dimension(model_registry.tip_model) == 2048
        assert get_embedding_dimension(Case) == DEFAULT_EMBEDDING_DIMENSION
        assert get_embedding_dimension(Tip) == DEFAULT_EMBEDDING_DIMENSION
