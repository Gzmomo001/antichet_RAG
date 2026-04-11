import argparse
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from antifraud_rag.core.config import Settings
from antifraud_rag.db.models import Base


async def init_db(settings: Settings):
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        print("Database tables created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize database tables")
    parser.add_argument("--db-url", required=True, help="Database URL")
    args = parser.parse_args()

    settings = Settings(
        EMBEDDING_MODEL_URL="https://placeholder.com",
        EMBEDDING_MODEL_API_KEY="placeholder",
        DATABASE_URL=args.db_url,
    )
    asyncio.run(init_db(settings))
