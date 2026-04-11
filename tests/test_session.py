"""
Unit tests for antifraud_rag/db/session.py - Database session management.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from antifraud_rag.core.config import Settings
from antifraud_rag.db import session as session_module


@pytest.fixture(autouse=True)
def reset_session_globals():
    original_engine = session_module.engine
    original_factory = session_module.async_session_factory
    yield
    session_module.engine = original_engine
    session_module.async_session_factory = original_factory


class TestInitEngine:
    def test_init_engine_creates_engine_and_factory(self):
        settings = Settings(
            EMBEDDING_MODEL_URL="https://test.com",
            EMBEDDING_MODEL_API_KEY="test-key",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/testdb",
        )

        with (
            patch("antifraud_rag.db.session.create_async_engine") as mock_create_engine,
            patch("antifraud_rag.db.session.async_sessionmaker") as mock_sessionmaker,
        ):
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_factory = MagicMock()
            mock_sessionmaker.return_value = mock_factory

            session_module.init_engine(settings)

            mock_create_engine.assert_called_once_with(
                settings.DATABASE_URL,
                echo=False,
                pool_pre_ping=True,
            )
            mock_sessionmaker.assert_called_once_with(
                mock_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            assert session_module.engine == mock_engine
            assert session_module.async_session_factory == mock_factory

    def test_init_engine_can_be_called_multiple_times(self):
        settings = Settings(
            EMBEDDING_MODEL_URL="https://test.com",
            EMBEDDING_MODEL_API_KEY="test-key",
        )

        with (
            patch("antifraud_rag.db.session.create_async_engine") as mock_create_engine,
            patch("antifraud_rag.db.session.async_sessionmaker"),
        ):
            mock_create_engine.return_value = MagicMock()
            session_module.init_engine(settings)
            session_module.init_engine(settings)

            assert mock_create_engine.call_count == 2


class TestGetSession:
    def test_get_session_raises_when_not_initialized(self):
        session_module.async_session_factory = None

        with pytest.raises(RuntimeError, match="Database not initialized"):
            session_module.get_session()

    def test_get_session_returns_session_from_factory(self):
        mock_factory = MagicMock()
        mock_session = MagicMock()
        mock_factory.return_value = mock_session
        session_module.async_session_factory = mock_factory

        result = session_module.get_session()

        assert result == mock_session
        mock_factory.assert_called_once()
