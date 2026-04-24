from typing import Any, Dict, Hashable, List, Optional, Tuple, Type

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from antifraud_rag.db.models import Case, Tip


class RetrievalService:
    def __init__(
        self,
        db: AsyncSession,
        case_model: Optional[Type[Any]] = None,
        tip_model: Optional[Type[Any]] = None,
    ):
        self.db = db
        self.case_model = case_model if case_model is not None else Case
        self.tip_model = tip_model if tip_model is not None else Tip

    async def search_cases_vector(
        self, query_embedding: List[float], limit: int = 20
    ) -> List[Tuple[Any, float]]:
        query = (
            select(
                self.case_model,
                (1 - self.case_model.embedding.cosine_distance(query_embedding)).label("score"),
            )
            .order_by(self.case_model.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        result = await self.db.execute(query)
        return result.all()

    async def search_cases_bm25(
        self, query_text: str, limit: int = 20
    ) -> List[Tuple[Any, float]]:
        sql = text("""
            SELECT id, ts_rank(content_tsv, plainto_tsquery('simple', :query)) as score
            FROM cases_table
            WHERE content_tsv @@ plainto_tsquery('simple', :query)
            ORDER BY score DESC
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"query": query_text, "limit": limit})
        rows = result.all()

        if not rows:
            return []

        # Preserve the BM25 ranking order explicitly by iterating over ``rows``
        # (ordered by score DESC from the SQL query) rather than relying on
        # implicit dict insertion-order semantics.
        case_ids = [row[0] for row in rows]
        scores_map: Dict[str, float] = {str(row[0]): row[1] for row in rows}

        cases_query = select(self.case_model).where(self.case_model.id.in_(case_ids))
        cases_result = await self.db.execute(cases_query)
        cases = cases_result.scalars().all()

        cases_by_id = {str(case.id): case for case in cases}
        return [
            (cases_by_id[str(cid)], scores_map[str(cid)])
            for cid in case_ids
            if str(cid) in cases_by_id
        ]

    async def search_tips_vector(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Tuple[Any, float]]:
        """Vector-based tip search."""
        query = (
            select(
                self.tip_model,
                (1 - self.tip_model.embedding.cosine_distance(query_embedding)).label("score"),
            )
            .order_by(self.tip_model.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.all()

    async def search_tips_bm25(
        self, query_text: str, limit: int = 5
    ) -> List[Tuple[Any, float]]:
        """BM25-based tip search."""
        sql = text("""
            SELECT id, ts_rank(content_tsv, plainto_tsquery('simple', :query)) as score
            FROM tips_table
            WHERE content_tsv @@ plainto_tsquery('simple', :query)
            ORDER BY score DESC
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"query": query_text, "limit": limit})
        rows = result.all()

        if not rows:
            return []

        tip_ids = [row[0] for row in rows]
        scores_map: Dict[str, float] = {str(row[0]): row[1] for row in rows}

        tips_query = select(self.tip_model).where(self.tip_model.id.in_(tip_ids))
        tips_result = await self.db.execute(tips_query)
        tips = tips_result.scalars().all()

        tips_by_id = {str(tip.id): tip for tip in tips}
        return [
            (tips_by_id[str(tid)], scores_map[str(tid)])
            for tid in tip_ids
            if str(tid) in tips_by_id
        ]

    def rrf_fusion(
        self,
        bm25_results: List[Tuple[Any, float]],
        vector_results: List[Tuple[Any, float]],
        k: int = 60,
        normalize: bool = True,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion.

        Combines BM25 and vector rankings into a single score using
        ``score = Σ 1/(k + rank + 1)`` for each list an item appears in.

        When *normalize* is ``True`` (default) the raw RRF sum is divided by
        ``2 / (k + 1)``, the theoretical maximum score for a single item that
        appears at rank 0 in both input lists.  This maps scores to [0, 1].
        """
        fused_scores: Dict[Hashable, float] = {}
        items_by_id: Dict[Hashable, Any] = {}

        for rank, (item, _) in enumerate(bm25_results):
            fused_scores[item.id] = fused_scores.get(item.id, 0.0) + 1 / (k + rank + 1)
            items_by_id.setdefault(item.id, item)

        for rank, (item, _) in enumerate(vector_results):
            fused_scores[item.id] = fused_scores.get(item.id, 0.0) + 1 / (k + rank + 1)
            items_by_id.setdefault(item.id, item)

        normalization_factor = (2.0 / (k + 1)) if normalize else 1.0

        results = [
            {"item": items_by_id[item_id], "score": score / normalization_factor}
            for item_id, score in fused_scores.items()
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)

    async def search_tips(
        self, query_text: str, query_embedding: List[float], limit: int = 5
    ) -> List[Any]:
        """Hybrid tip search using BM25 + vector RRF fusion."""
        bm25_results = await self.search_tips_bm25(query_text, limit)
        vector_results = await self.search_tips_vector(query_embedding, limit)
        fused = self.rrf_fusion(bm25_results, vector_results)
        return [res["item"] for res in fused[:limit]]
