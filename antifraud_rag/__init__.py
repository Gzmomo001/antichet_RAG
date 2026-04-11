"""
Anti-Fraud RAG System - 反欺诈 RAG 系统

一个基于混合搜索（BM25 + 向量 + RRF）的反欺诈分析系统。

使用示例:
    from antifraud_rag import FraudAnalyzer, Settings

    # 初始化 (所有配置通过参数传入)
    settings = Settings(
        EMBEDDING_MODEL_URL="https://your-api.com/v1/embeddings",
        EMBEDDING_MODEL_API_KEY="your-api-key",
    )
    analyzer = FraudAnalyzer(db_session, settings=settings)

    # 分析文本风险
    result = await analyzer.analyze("这是一个可疑的电话...")

    # 添加案例
    await analyzer.add_case(description="...", fraud_type="电信诈骗")

    # 添加知识
    await analyzer.add_tip(title="...", content="...")
"""

from antifraud_rag.analyzer import AntiFraudRAG, FraudAnalyzer
from antifraud_rag.core.config import Settings
from antifraud_rag.core.enums import ResultType, RiskLevel
from antifraud_rag.core.exceptions import (
    AntiFraudError,
    DatabaseNotInitializedError,
    EmbeddingError,
)
from antifraud_rag.db.models import Case, Tip
from antifraud_rag.schemas import (
    AnalysisResponse,
    DirectHitData,
    MatchedCase,
    RAGPromptContext,
    RAGPromptData,
)

__all__ = [
    "FraudAnalyzer",
    "AntiFraudRAG",
    "Settings",
    "Case",
    "Tip",
    "AnalysisResponse",
    "DirectHitData",
    "MatchedCase",
    "RAGPromptContext",
    "RAGPromptData",
    "ResultType",
    "RiskLevel",
    "AntiFraudError",
    "EmbeddingError",
    "DatabaseNotInitializedError",
]

__version__ = "1.0.0"
