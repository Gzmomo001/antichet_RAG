"""
Database session management for Anti-Fraud RAG system.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from antifraud_rag.core.config import Settings
from antifraud_rag.core.exceptions import DatabaseNotInitializedError

# Global engine and session factory (initialized via init_db)
engine = None
async_session_factory = None


def init_engine(settings: Settings) -> None:
    """
    Initialize the async database engine and session factory.
    Call this once at application startup.
    """
    global engine, async_session_factory

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
