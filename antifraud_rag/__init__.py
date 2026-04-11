"""
Anti-Fraud RAG System - 反欺诈 RAG 系统

一个基于混合搜索（BM25 + 向量 + RRF）的反欺诈分析系统。

使用示例:
    from antifraud_rag import AntiFraudRAG, Settings

    # 初始化 (所有配置通过参数传入)
    settings = Settings(
        EMBEDDING_MODEL_URL="https://your-api.com/v1/embeddings",
        EMBEDDING_MODEL_API_KEY="your-api-key",
    )
    rag = AntiFraudRAG(db_session, settings=settings)

    # 分析文本风险
    result = await rag.analyze("这是一个可疑的电话...")

    # 添加案例
    await rag.add_case(description="...", fraud_type="电信诈骗")

    # 添加知识
    await rag.add_tip(title="...", content="...")
"""

from antifraud_rag.core.config import Settings
from antifraud_rag.db.models import Case, Tip
from antifraud_rag.main import AntiFraudRAG
from antifraud_rag.schemas import (
    AnalysisResponse,
    DirectHitData,
    MatchedCase,
    RAGPromptContext,
    RAGPromptData,
)

__all__ = [
    "AntiFraudRAG",
    "Settings",
    "Case",
    "Tip",
    "AnalysisResponse",
    "DirectHitData",
    "MatchedCase",
    "RAGPromptContext",
    "RAGPromptData",
]

__version__ = "1.0.0"
