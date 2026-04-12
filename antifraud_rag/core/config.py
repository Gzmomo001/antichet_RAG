from pydantic import BaseModel, ConfigDict, Field

from antifraud_rag.core.constants import EMBEDDING_DIMENSION as DEFAULT_EMBEDDING_DIMENSION


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    EMBEDDING_MODEL_URL: str
    EMBEDDING_MODEL_API_KEY: str
    EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, gt=0)
    HIGH_RISK_THRESHOLD: float = 0.85
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@db:5432/antifraud"
