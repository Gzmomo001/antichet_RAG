from collections import defaultdict
from typing import Any, Dict, List, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from antifraud_rag.core.constants import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TIPS_LIMIT,
    RRF_K,
)
from antifraud_rag.db.models import Case, Tip


class RetrievalService:
    def __init__(self, db: AsyncSession, case_model: Any = Case, tip_model: Any = Tip):
        self.db = db
        self.case_model = case_model
        self.tip_model = tip_model

    async def _hydrate_ranked_results(
        self,
        model: Any,
        rows: List[Tuple[Any, float]],
    ) -> List[Tuple[Any, float]]:
        """Load ORM objects for ranked search rows while preserving ranking order."""
        if not rows:
            return []

        item_ids = [row[0] for row in rows]
        scores = {row[0]: row[1] for row in rows}

        items_query = select(model).where(model.id.in_(item_ids))
        items_result = await self.db.execute(items_query)
        items = items_result.scalars().all()
        items_by_id = {item.id: item for item in items}

        return [
            (items_by_id[item_id], scores[item_id])
            for item_id in item_ids
            if item_id in items_by_id
        ]

    async def search_cases_vector(
        self, query_embedding: List[float], limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Tuple[Case, float]]:
        # Vector search using pgvector cosine distance
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
        self, query_text: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Tuple[Case, float]]:
        # BM25 search using PostgreSQL ts_rank
        sql = text("""
            SELECT id, ts_rank(content_tsv, plainto_tsquery('english', :query)) as score
            FROM cases_table
            WHERE content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"query": query_text, "limit": limit})
        return await self._hydrate_ranked_results(self.case_model, result.all())

    async def search_tips_vector(
        self, query_embedding: List[float], limit: int = DEFAULT_TIPS_LIMIT
    ) -> List[Tuple[Tip, float]]:
        # Vector search using pgvector cosine distance
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
        self, query_text: str, limit: int = DEFAULT_TIPS_LIMIT
    ) -> List[Tuple[Tip, float]]:
        # BM25 search using PostgreSQL ts_rank
        sql = text("""
            SELECT id, ts_rank(content_tsv, plainto_tsquery('english', :query)) as score
            FROM tips_table
            WHERE content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"query": query_text, "limit": limit})
        return await self._hydrate_ranked_results(self.tip_model, result.all())

    def rrf_fusion(
        self,
        bm25_results: List[Tuple[Any, float]],
        vector_results: List[Tuple[Any, float]],
        k: int = RRF_K,
        normalize: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm to combine multiple search rankings.
        """
        # Using defaultdict to simplify scoring logic
        scores = defaultdict(lambda: {"item": None, "score": 0.0})

        # Process BM25 results
        for rank, (item, _) in enumerate(bm25_results):
            scores[item.id]["item"] = item
            scores[item.id]["score"] += 1.0 / (k + rank + 1)

        # Process Vector results
        for rank, (item, _) in enumerate(vector_results):
            scores[item.id]["item"] = item
            scores[item.id]["score"] += 1.0 / (k + rank + 1)

        # Sort by score descending
        results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        # Normalize scores if requested
        if normalize and results:
            normalization_factor = 2.0 / (k + 1)
            for result in results:
                result["score"] = result["score"] / normalization_factor

        return results

    async def search_tips(
        self, query_text: str, query_embedding: List[float], limit: int = DEFAULT_TIPS_LIMIT
    ) -> List[Tip]:
        # Hybrid search for tips using BM25 + vector + RRF fusion
        bm25_tips = await self.search_tips_bm25(query_text, limit)
        vector_tips = await self.search_tips_vector(query_embedding, limit)
        fused_results = self.rrf_fusion(bm25_tips, vector_tips)
        return [result["item"] for result in fused_results[:limit]]
