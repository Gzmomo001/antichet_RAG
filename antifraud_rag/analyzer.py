import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from antifraud_rag.core.config import Settings
from antifraud_rag.core.constants import (
    DEFAULT_SEARCH_LIMIT,
    MEDIUM_RISK_THRESHOLD,
)
from antifraud_rag.core.enums import ResultType, RiskLevel
from antifraud_rag.core.exceptions import EmbeddingError
from antifraud_rag.db.models import configure_embedding_dimension
from antifraud_rag.schemas import (
    AnalysisResponse,
    DirectHitData,
    RAGPromptContext,
    RAGPromptData,
)
from antifraud_rag.services.cache import EmbeddingCache
from antifraud_rag.services.embedding import EmbeddingService
from antifraud_rag.services.prompts import (
    build_matched_cases,
    build_rag_prompt,
    build_relevant_cases_data,
    build_tips_data,
)
from antifraud_rag.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class FraudAnalyzer:
    """
    反欺诈 RAG 系统核心类。

    使用方法:
        from antifraud_rag import FraudAnalyzer

        # 初始化（缓存默认开启，参数来自 settings）
        analyzer = FraudAnalyzer(db_session, settings=my_settings)

        # 关闭缓存
        analyzer = FraudAnalyzer(db_session, settings=my_settings,
                                  embedding_cache=None)

        # 自定义缓存参数
        from antifraud_rag import EmbeddingCache
        cache = EmbeddingCache(max_size=500, ttl_seconds=3600)
        analyzer = FraudAnalyzer(db_session, settings=my_settings,
                                  embedding_cache=cache)

        # 分析文本风险
        result = await analyzer.analyze("这是一个可疑的电话...")

        # 查看缓存命中统计
        print(analyzer.embedding_service.cache_stats)
    """

    def __init__(
        self,
        db: AsyncSession,
        settings: Settings,
        embedding_service: Optional[EmbeddingService] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
    ):
        """
        Args:
            db: Async SQLAlchemy session.
            settings: Application settings.
            embedding_service: Pre-built :class:`EmbeddingService` to use.
                When supplied, *embedding_cache* is ignored (the provided
                service is used as-is with whatever cache it was configured
                with).
            embedding_cache: Cache to attach to the default
                :class:`EmbeddingService`.  Pass an :class:`EmbeddingCache`
                instance to use custom parameters, or ``None`` to disable
                caching.  Ignored when *embedding_service* is given.
                Defaults to a new :class:`EmbeddingCache` built from
                ``settings.EMBEDDING_CACHE_MAX_SIZE`` /
                ``settings.EMBEDDING_CACHE_TTL_SECONDS``.
        """
        self.db = db
        self.settings = settings
        model_registry = configure_embedding_dimension(self.settings.EMBEDDING_DIMENSION)
        self.case_model = model_registry.case_model
        self.tip_model = model_registry.tip_model

        if embedding_service is not None:
            if embedding_cache is not None:
                logger.warning(
                    "Both embedding_service and embedding_cache were provided. "
                    "embedding_cache will be ignored because the supplied "
                    "embedding_service already has its own cache configuration."
                )
            self.embedding_service = embedding_service
        else:
            cache = (
                embedding_cache
                if embedding_cache is not None
                else EmbeddingCache(
                    max_size=self.settings.EMBEDDING_CACHE_MAX_SIZE,
                    ttl_seconds=self.settings.EMBEDDING_CACHE_TTL_SECONDS,
                )
            )
            self.embedding_service = EmbeddingService(settings=self.settings, cache=cache)

        self.retrieval_service = RetrievalService(
            db,
            case_model=self.case_model,
            tip_model=self.tip_model,
        )

    async def analyze(self, text: str) -> AnalysisResponse:
        """
        分析文本的欺诈风险。

        Args:
            text: 待分析的文本内容

        Returns:
            AnalysisResponse: 包含风险评估结果
                - Direct_Hit: 高风险，直接命中已知案例
                - RAG_Prompt: 中低风险，返回 RAG prompt 供进一步分析
        """
        query_embedding = await self._get_query_embedding(text)
        fused_results = await self._search_cases(text, query_embedding)

        if not fused_results:
            return await self._build_low_risk_response(text, query_embedding)

        top_score = fused_results[0]["score"]

        if top_score >= self.settings.HIGH_RISK_THRESHOLD:
            return self._build_high_risk_response(fused_results)

        return await self._build_medium_risk_response(
            text, fused_results, query_embedding, top_score
        )

    async def add_case(
        self,
        description: str,
        fraud_type: Optional[str] = None,
        amount: Optional[float] = None,
        keywords: Optional[List[str]] = None,
    ) -> Any:
        """
        添加案例到知识库。

        Args:
            description: 案例描述
            fraud_type: 诈骗类型
            amount: 涉案金额
            keywords: 关键词列表

        Returns:
            创建的案例对象
        """
        embedding = await self.embedding_service.get_embeddings(description)

        case = self.case_model(
            description=description,
            fraud_type=fraud_type,
            amount=amount,
            keywords=keywords,
            embedding=embedding,
        )

        self.db.add(case)
        await self.db.commit()
        await self.db.refresh(case)

        return case

    async def add_tip(
        self,
        title: str,
        content: str,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> Any:
        """
        添加反诈知识到知识库。

        Args:
            title: 知识标题
            content: 知识内容
            category: 知识分类
            keywords: 关键词列表

        Returns:
            创建的知识对象
        """
        embedding = await self.embedding_service.get_embeddings(f"{title} {content}")

        tip = self.tip_model(
            title=title,
            content=content,
            category=category,
            keywords=keywords,
            embedding=embedding,
        )

        self.db.add(tip)
        await self.db.commit()
        await self.db.refresh(tip)

        return tip

    async def search_similar_cases(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        搜索相似案例（向量搜索）。

        Args:
            query: 查询文本
            limit: 返回结果数量

        Returns:
            List[Dict]: 相似案例列表
        """
        embedding = await self.embedding_service.get_embeddings(query)
        results = await self.retrieval_service.search_cases_vector(embedding, limit)

        return [
            {
                "case_id": case.id,
                "description": case.description,
                "fraud_type": case.fraud_type,
                "score": score,
            }
            for case, score in results
        ]

    async def hybrid_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（BM25 + 向量搜索 + RRF融合）。

        Args:
            query: 查询文本
            limit: 返回结果数量

        Returns:
            List[Dict]: 融合后的搜索结果
        """
        embedding = await self.embedding_service.get_embeddings(query)

        bm25_cases = await self.retrieval_service.search_cases_bm25(query, limit)
        vector_cases = await self.retrieval_service.search_cases_vector(embedding, limit)

        fused_results = self.retrieval_service.rrf_fusion(bm25_cases, vector_cases)

        return [
            {
                "case_id": res["item"].id,
                "description": res["item"].description,
                "fraud_type": res["item"].fraud_type,
                "rrf_score": res["score"],
            }
            for res in fused_results[:limit]
        ]

    async def _get_query_embedding(self, text: str) -> List[float]:
        """获取查询文本的嵌入向量。"""
        try:
            return await self.embedding_service.get_embeddings(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise EmbeddingError(f"Failed to get embedding: {str(e)}")

    async def _search_cases(self, text: str, embedding: List[float]) -> List[Dict[str, Any]]:
        """执行混合搜索并返回融合结果。"""
        bm25_cases = await self.retrieval_service.search_cases_bm25(text)
        vector_cases = await self.retrieval_service.search_cases_vector(embedding)
        return self.retrieval_service.rrf_fusion(bm25_cases, vector_cases)

    async def _build_low_risk_response(self, text: str, embedding: List[float]) -> AnalysisResponse:
        """构建低风险响应。"""
        tips = await self.retrieval_service.search_tips(text, embedding)
        return self._build_rag_prompt_response(
            text=text,
            relevant_cases_data=[],
            tips_data=build_tips_data(tips),
            risk_level=RiskLevel.LOW.value,
            top_score=0.0,
        )

    def _build_high_risk_response(self, fused_results: List[Dict[str, Any]]) -> AnalysisResponse:
        """构建高风险响应。"""
        return AnalysisResponse(
            result_type=ResultType.DIRECT_HIT.value,
            data=DirectHitData(
                risk_level=RiskLevel.HIGH.value,
                matched_cases=build_matched_cases(fused_results),
            ),
        )

    async def _build_medium_risk_response(
        self,
        text: str,
        fused_results: List[Dict[str, Any]],
        embedding: List[float],
        top_score: float,
    ) -> AnalysisResponse:
        """构建中等风险响应。"""
        tips = await self.retrieval_service.search_tips(text, embedding)
        relevant_cases_data = build_relevant_cases_data(fused_results)
        tips_data = build_tips_data(tips)

        risk_level = (
            RiskLevel.MEDIUM.value if top_score > MEDIUM_RISK_THRESHOLD else RiskLevel.LOW.value
        )

        return self._build_rag_prompt_response(
            text=text,
            relevant_cases_data=relevant_cases_data,
            tips_data=tips_data,
            risk_level=risk_level,
            top_score=top_score,
        )

    def _build_rag_prompt_response(
        self,
        text: str,
        relevant_cases_data: List[Dict[str, Any]],
        tips_data: List[Dict[str, Any]],
        risk_level: str,
        top_score: float,
    ) -> AnalysisResponse:
        """构建统一的 RAG prompt 响应。"""
        return AnalysisResponse(
            result_type=ResultType.RAG_PROMPT.value,
            data=RAGPromptData(
                risk_level=risk_level,
                rrf_score=top_score,
                prompt=build_rag_prompt(text, relevant_cases_data, tips_data),
                context=RAGPromptContext(
                    relevant_cases=relevant_cases_data,
                    anti_fraud_tips=tips_data,
                ),
            ),
        )


AntiFraudRAG = FraudAnalyzer
