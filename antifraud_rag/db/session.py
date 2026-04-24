"""
Database session management for Anti-Fraud RAG system.
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from antifraud_rag.core.config import Settings
from antifraud_rag.core.exceptions import DatabaseNotInitializedError

logger = logging.getLogger(__name__)

# Global engine and session factory (initialized via init_engine)
engine: Optional[AsyncEngine] = None
async_session_factory: Optional[async_sessionmaker] = None


def init_engine(settings: Settings) -> None:
    """
    Initialize the async database engine and session factory.

    Safe to call multiple times — the previous engine, if any, is disposed
    to avoid leaking connection pools.
    """
    global engine, async_session_factory

    # Dispose the previous engine to avoid leaking its connection pool if
    # ``init_engine`` is called more than once.
    if engine is not None:
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # We're inside a running event loop; schedule disposal.
                loop.create_task(engine.dispose())
            else:
                asyncio.run(engine.dispose())
        except Exception as exc:  # pragma: no cover - defensive, non-fatal
            logger.warning("Failed to dispose previous engine: %s", exc)

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


def get_session() -> AsyncSession:
    """
    Get a new async database session.
    Usage:
        async with get_session() as session:
            ...
    """
    if async_session_factory is None:
        raise DatabaseNotInitializedError("Database not initialized. Call init_engine() first.")
    return async_session_factory()


async def dispose_engine() -> None:
    """Dispose the current engine and clear the session factory."""
    global engine, async_session_factory
    if engine is not None:
        await engine.dispose()
    engine = None
    async_session_factory = None
