from typing import Any, Dict, List, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from antifraud_rag.db.models import Case, Tip


class RetrievalService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def search_cases_vector(
        self, query_embedding: List[float], limit: int = 20
    ) -> List[Tuple[Case, float]]:
        # Vector search using pgvector cosine distance
        query = (
            select(Case, (1 - Case.embedding.cosine_distance(query_embedding)).label("score"))
            .order_by(Case.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        result = await self.db.execute(query)
        return result.all()

    async def search_cases_bm25(self, query_text: str, limit: int = 20) -> List[Tuple[Case, float]]:
        # BM25 search using PostgreSQL ts_rank
        sql = text("""
            SELECT id, ts_rank(content_tsv, plainto_tsquery('english', :query)) as score
            FROM cases_table
            WHERE content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"query": query_text, "limit": limit})
        rows = result.all()

        if not rows:
            return []

        case_ids = [row[0] for row in rows]
        scores = {row[0]: row[1] for row in rows}

        cases_query = select(Case).where(Case.id.in_(case_ids))
        cases_result = await self.db.execute(cases_query)
        cases = cases_result.scalars().all()

        return [(case, scores[case.id]) for case in cases]

    def rrf_fusion(
        self,
        bm25_results: List[Tuple[Any, float]],
        vector_results: List[Tuple[Any, float]],
        k: int = 60,
        normalize: bool = True,
    ) -> List[Dict[str, Any]]:
        scores: Dict[int, Dict[str, Any]] = {}

        for rank, (item, _) in enumerate(bm25_results):
            scores[item.id] = {
                "item": item,
                "score": scores.get(item.id, {"score": 0})["score"] + 1 / (k + rank + 1),
            }

        for rank, (item, _) in enumerate(vector_results):
            scores[item.id] = {
                "item": item,
                "score": scores.get(item.id, {"score": 0})["score"] + 1 / (k + rank + 1),
            }

        results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        if normalize and results:
            max_score = 2 / (k + 1)
            for result in results:
                result["score"] = result["score"] / max_score

        return results

    async def search_tips(
        self, query_text: str, query_embedding: List[float], limit: int = 5
    ) -> List[Tip]:
        # Hybrid search for tips (simplified for MVP)
        query = select(Tip).order_by(Tip.embedding.cosine_distance(query_embedding)).limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()
