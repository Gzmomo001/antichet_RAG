# AGENTS.md — Anti-Fraud RAG Library

## Project

Python library for anti-fraud RAG (Retrieval-Augmented Generation) with hybrid search (BM25 + vector + RRF), PostgreSQL/pgvector backend. **Not a FastAPI service** — import and use programmatically.

**Note**: README.md describes a FastAPI service that doesn't exist. Dockerfile is broken (references `app.main:app`). Actual entrypoint is `AntiFraudRAG` class.

## Quick Commands

```bash
make install       # Install deps via uv sync
make test          # Run pytest (auto-sets mock env vars)
make lint          # ruff check antifraud_rag tests
make fmt           # ruff format + fix
make ci            # lint + test
```

## Configuration

All config is passed explicitly via `Settings` (no env vars). `EMBEDDING_MODEL_URL` and `EMBEDDING_MODEL_API_KEY` are required.

```python
settings = Settings(
    EMBEDDING_MODEL_URL="https://your-embedding-api.com/v1/embeddings",
    EMBEDDING_MODEL_API_KEY="your-key",
    DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db",  # optional
)
```

## Architecture

```
antifraud_rag/
├── main.py              # AntiFraudRAG class (core entrypoint)
├── schemas.py           # Pydantic response schemas
├── core/config.py       # Settings (Pydantic BaseModel)
├── db/
│   ├── models.py        # Case, Tip (SQLAlchemy + pgvector)
│   └── session.py       # Async engine config
└── services/
    ├── retrieval.py     # BM25, vector search, RRF fusion (k=60)
    ├── embedding.py     # Embedding API client
    └── prompts.py       # Prompt construction helpers
```

## Usage

```python
from antifraud_rag import AntiFraudRAG, Settings
from antifraud_rag.db.session import get_session

settings = Settings(
    EMBEDDING_MODEL_URL="...",
    EMBEDDING_MODEL_API_KEY="..."
)

async with get_session() as db:
    rag = AntiFraudRAG(db, settings=settings)
    
    result = await rag.analyze("可疑文本...")
    await rag.add_case(description="...", fraud_type="电信诈骗")
    await rag.add_tip(title="...", content="...")
```

## Database Setup

```bash
# Docker Compose (starts postgres with pgvector)
docker-compose up -d

# Init tables + pgvector extension
uv run python scripts/init_db.py --db-url postgresql+asyncpg://user:pass@localhost:5432/antifraud
```

English tokenizer used for tsvector; zhparser commented out in `init_db.py`.

## Key Implementation

- **RRF fusion**: k=60, combines BM25 + vector rankings
- **High-risk threshold**: score > 0.85 → Direct_Hit, else RAG_Prompt
- **Tip embedding**: uses `title + " " + content` (differs from Case)
- **Async**: SQLAlchemy with asyncpg driver

## Testing

```bash
make test-cov       # Coverage report
make test-config    # Settings tests only
make test-services  # Retrieval + embedding
```

Pytest asyncio_mode = "auto" (no explicit markers needed).