import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import text

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_settings = None
_engine_initialized = False

TextInput = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]
TitleInput = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
CategoryInput = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
KeywordInput = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
SearchQuery = Annotated[str, Query(min_length=1, max_length=5000)]
SearchLimit = Annotated[int, Query(ge=1, le=50)]


def get_settings():
    global _settings
    if _settings is None:
        from antifraud_rag import Settings

        _settings = Settings(
            EMBEDDING_MODEL_URL=os.environ.get("EMBEDDING_MODEL_URL", ""),
            EMBEDDING_MODEL_API_KEY=os.environ.get("EMBEDDING_MODEL_API_KEY", ""),
            EMBEDDING_MODEL_NAME=os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002"),
            EMBEDDING_DIMENSION=int(os.environ.get("EMBEDDING_DIMENSION", "1536")),
            HIGH_RISK_THRESHOLD=float(os.environ.get("HIGH_RISK_THRESHOLD", "0.85")),
            DATABASE_URL=os.environ.get(
                "DATABASE_URL",
                "postgresql+asyncpg://{}:{}@localhost:5432/{}".format(
                    os.environ.get("POSTGRES_USER", "antifraud_user"),
                    os.environ.get("POSTGRES_PASSWORD", "antifraud_pass"),
                    os.environ.get("POSTGRES_DB", "antifraud_db"),
                ),
            ),
        )
    return _settings


def get_cors_allow_origins() -> List[str]:
    raw_origins = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


async def verify_database_ready() -> None:
    from antifraud_rag.db.session import get_session

    async with get_session() as db:
        await db.execute(text("SELECT 1"))
        await db.execute(text("SELECT 1 FROM cases_table LIMIT 1"))
        await db.execute(text("SELECT 1 FROM tips_table LIMIT 1"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine_initialized
    try:
        from antifraud_rag.db.session import init_engine

        s = get_settings()
        if s.EMBEDDING_MODEL_URL and s.DATABASE_URL:
            init_engine(s)
            await verify_database_ready()
            _engine_initialized = True
            logger.info("Engine initialized and database verified successfully")
        else:
            logger.warning("Engine not initialized: missing EMBEDDING_MODEL_URL or DATABASE_URL")
    except Exception:
        _engine_initialized = False
        logger.exception("Could not initialize API engine")
    try:
        yield
    finally:
        try:
            from antifraud_rag.db.session import dispose_engine

            await dispose_engine()
        finally:
            _engine_initialized = False


app = FastAPI(
    title="AntiCheat RAG API",
    description="反欺诈 RAG 系统 REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: TextInput
    source: CategoryInput = "user_submission"


class AddCaseRequest(BaseModel):
    description: TextInput
    fraud_type: Optional[CategoryInput] = None
    amount: Optional[float] = Field(default=None, ge=0)
    keywords: Optional[List[KeywordInput]] = Field(default=None, max_length=50)


class AddTipRequest(BaseModel):
    title: TitleInput
    content: TextInput
    category: Optional[CategoryInput] = None
    keywords: Optional[List[KeywordInput]] = Field(default=None, max_length=50)


def require_ready() -> None:
    if not _engine_initialized:
        raise HTTPException(
            status_code=503,
            detail="服务未就绪：请先配置 .env 文件并确保数据库连接正常，然后重启服务",
        )


def raise_internal_error(operation: str):
    logger.exception("%s failed", operation)
    raise HTTPException(status_code=500, detail=f"{operation}失败，请查看服务端日志")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "engine_initialized": _engine_initialized,
    }


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    require_ready()
    try:
        from antifraud_rag import FraudAnalyzer
        from antifraud_rag.db.session import get_session

        s = get_settings()
        async with get_session() as db:
            analyzer = FraudAnalyzer(db, settings=s)
            result = await analyzer.analyze(request.text)
            return result.model_dump()
    except HTTPException:
        raise
    except Exception:
        raise_internal_error("文本分析")


@app.post("/api/cases")
async def add_case(request: AddCaseRequest):
    require_ready()
    try:
        from antifraud_rag import FraudAnalyzer
        from antifraud_rag.db.session import get_session

        s = get_settings()
        async with get_session() as db:
            analyzer = FraudAnalyzer(db, settings=s)
            case = await analyzer.add_case(
                description=request.description,
                fraud_type=request.fraud_type,
                amount=request.amount,
                keywords=request.keywords,
            )
            return {
                "id": str(case.id),
                "description": case.description,
                "fraud_type": case.fraud_type,
                "amount": case.amount,
                "keywords": case.keywords or [],
            }
    except HTTPException:
        raise
    except Exception:
        raise_internal_error("添加案例")


@app.post("/api/tips")
async def add_tip(request: AddTipRequest):
    require_ready()
    try:
        from antifraud_rag import FraudAnalyzer
        from antifraud_rag.db.session import get_session

        s = get_settings()
        async with get_session() as db:
            analyzer = FraudAnalyzer(db, settings=s)
            tip = await analyzer.add_tip(
                title=request.title,
                content=request.content,
                category=request.category,
                keywords=request.keywords,
            )
            return {
                "id": str(tip.id),
                "title": tip.title,
                "content": tip.content,
                "category": tip.category,
                "keywords": tip.keywords or [],
            }
    except HTTPException:
        raise
    except Exception:
        raise_internal_error("添加知识")


@app.get("/api/cases/search")
async def search_cases(query: SearchQuery, limit: SearchLimit = 5):
    require_ready()
    try:
        from antifraud_rag import FraudAnalyzer
        from antifraud_rag.db.session import get_session

        s = get_settings()
        async with get_session() as db:
            analyzer = FraudAnalyzer(db, settings=s)
            results = await analyzer.search_similar_cases(query, limit=limit)
            return [
                {
                    "case_id": str(r["case_id"]),
                    "description": r["description"],
                    "fraud_type": r["fraud_type"],
                    "score": r["score"],
                }
                for r in results
            ]
    except HTTPException:
        raise
    except Exception:
        raise_internal_error("案例搜索")


@app.get("/api/search")
async def hybrid_search(query: SearchQuery, limit: SearchLimit = 10):
    require_ready()
    try:
        from antifraud_rag import FraudAnalyzer
        from antifraud_rag.db.session import get_session

        s = get_settings()
        async with get_session() as db:
            analyzer = FraudAnalyzer(db, settings=s)
            results = await analyzer.hybrid_search(query, limit=limit)
            return [
                {
                    "case_id": str(r["case_id"]),
                    "description": r["description"],
                    "fraud_type": r["fraud_type"],
                    "rrf_score": r["rrf_score"],
                }
                for r in results
            ]
    except HTTPException:
        raise
    except Exception:
        raise_internal_error("混合搜索")
