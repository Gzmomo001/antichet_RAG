import argparse
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from antifraud_rag.core.config import Settings
from antifraud_rag.core.constants import EMBEDDING_DIMENSION as DEFAULT_EMBEDDING_DIMENSION
from antifraud_rag.db.models import configure_embedding_dimension


async def init_db(settings: Settings):
    model_registry = configure_embedding_dimension(settings.EMBEDDING_DIMENSION)

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(model_registry.base.metadata.create_all)
        print(
            f"Database tables created successfully with embedding dimension "
            f"{settings.EMBEDDING_DIMENSION}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize database tables")
    parser.add_argument("--db-url", required=True, help="Database URL")
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSION,
        help="Embedding dimension for both Case and Tip vectors",
    )
    args = parser.parse_args()

    settings = Settings(
        EMBEDDING_MODEL_URL="https://placeholder.com",
        EMBEDDING_MODEL_API_KEY="placeholder",
        DATABASE_URL=args.db_url,
        EMBEDDING_DIMENSION=args.embedding_dimension,
    )
    asyncio.run(init_db(settings))
