import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from antifraud_rag.core.config import Settings
from antifraud_rag.db.models import Case, Tip
from antifraud_rag.schemas import (
    AnalysisResponse,
    DirectHitData,
    RAGPromptContext,
    RAGPromptData,
)
from antifraud_rag.services.embedding import EmbeddingService
from antifraud_rag.services.prompts import (
    build_matched_cases,
    build_rag_prompt,
    build_relevant_cases_data,
    build_tips_data,
)
from antifraud_rag.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class AntiFraudRAG:
    """
    反欺诈 RAG 系统核心类。

    使用方法:
        from antifraud_rag import AntiFraudRAG

        # 初始化
        rag = AntiFraudRAG(db_session, settings=my_settings)

        # 分析文本风险
        result = await rag.analyze("这是一个可疑的电话...")

        # 添加案例到知识库
        await rag.add_case(description="...", fraud_type="电信诈骗")

        # 添加反诈知识
        await rag.add_tip(title="...", content="...")
    """

    def __init__(
        self,
        db: AsyncSession,
        settings: Settings,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.db = db
        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.retrieval_service = RetrievalService(db)

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
        try:
            query_embedding = await self.embedding_service.get_embeddings(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise RuntimeError(f"Embedding error: {str(e)}")

        bm25_cases = await self.retrieval_service.search_cases_bm25(text)
        vector_cases = await self.retrieval_service.search_cases_vector(query_embedding)

        fused_results = self.retrieval_service.rrf_fusion(bm25_cases, vector_cases)

        if not fused_results:
            tips = await self.retrieval_service.search_tips(text, query_embedding)
            return AnalysisResponse(
                result_type="RAG_Prompt",
                data=RAGPromptData(
                    risk_level="LOW",
                    rrf_score=0.0,
                    prompt=f"分析用户请求: {text}",
                    context=RAGPromptContext(
                        relevant_cases=[],
                        anti_fraud_tips=build_tips_data(tips),
                    ),
                ),
            )

        top_result = fused_results[0]
        top_score = top_result["score"]

        if top_score >= self.settings.HIGH_RISK_THRESHOLD:
            return AnalysisResponse(
                result_type="Direct_Hit",
                data=DirectHitData(
                    risk_level="HIGH",
                    matched_cases=build_matched_cases(fused_results),
                ),
            )
        else:
            tips = await self.retrieval_service.search_tips(text, query_embedding)
            relevant_cases_data = build_relevant_cases_data(fused_results)
            tips_data = build_tips_data(tips)

            return AnalysisResponse(
                result_type="RAG_Prompt",
                data=RAGPromptData(
                    risk_level="MEDIUM" if top_score > 0.5 else "LOW",
                    rrf_score=top_score,
                    prompt=build_rag_prompt(text, relevant_cases_data, tips_data),
                    context=RAGPromptContext(
                        relevant_cases=relevant_cases_data,
                        anti_fraud_tips=tips_data,
                    ),
                ),
            )

    async def add_case(
        self,
        description: str,
        fraud_type: Optional[str] = None,
        amount: Optional[float] = None,
        keywords: Optional[List[str]] = None,
    ) -> Case:
        """
        添加案例到知识库。

        Args:
            description: 案例描述
            fraud_type: 诈骗类型
            amount: 涉案金额
            keywords: 关键词列表

        Returns:
            Case: 创建的案例对象
        """
        embedding = await self.embedding_service.get_embeddings(description)

        case = Case(
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
    ) -> Tip:
        """
        添加反诈知识到知识库。

        Args:
            title: 知识标题
            content: 知识内容
            category: 知识分类
            keywords: 关键词列表

        Returns:
            Tip: 创建的知识对象
        """
        embedding = await self.embedding_service.get_embeddings(f"{title} {content}")

        tip = Tip(
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

    async def search_similar_cases(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
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

    async def hybrid_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
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
