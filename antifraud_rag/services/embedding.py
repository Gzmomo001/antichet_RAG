import logging
from typing import List

import httpx

from antifraud_rag.core.config import Settings
from antifraud_rag.core.constants import EMBEDDING_TIMEOUT
from antifraud_rag.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.url = settings.EMBEDDING_MODEL_URL
        self.api_key = settings.EMBEDDING_MODEL_API_KEY
        self.model = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

    async def get_embeddings(self, text: str) -> List[float]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"input": text, "model": self.model},
                    timeout=EMBEDDING_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                # Assuming OpenAI compatible response format
                embedding = data["data"][0]["embedding"]

                if len(embedding) != self.dimension:
                    raise EmbeddingError(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}"
                    )

                return embedding
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Embedding API error: {str(e)}")
