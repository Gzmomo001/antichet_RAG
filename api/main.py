import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_settings = None
_engine_initialized = False


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine_initialized
    try:
        from antifraud_rag.db.session import init_engine
        s = get_settings()
        if s.EMBEDDING_MODEL_URL and s.DATABASE_URL:
            init_engine(s)
            _engine_initialized = True
            print("Engine initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize engine: {e}")
    yield


app = FastAPI(
    title="AntiCheat RAG API",
    description="反欺诈 RAG 系统 REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    source: str = "user_submission"


class AddCaseRequest(BaseModel):
    description: str
    fraud_type: Optional[str] = None
    amount: Optional[float] = None
    keywords: Optional[List[str]] = None


class AddTipRequest(BaseModel):
    title: str
    content: str
    category: Optional[str] = None
    keywords: Optional[List[str]] = None


def require_ready():
    if not _engine_initialized:
        raise HTTPException(
            status_code=503,
            detail="服务未就绪：请先配置 .env 文件并确保数据库连接正常，然后重启服务",
        )


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cases/search")
async def search_cases(query: str, limit: int = 5):
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def hybrid_search(query: str, limit: int = 10):
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
