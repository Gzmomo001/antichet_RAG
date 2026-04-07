from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Embedding Model API
    EMBEDDING_MODEL_URL: str
    EMBEDDING_MODEL_API_KEY: str
    EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536

    # Application
    PORT: int = 8000
    HIGH_RISK_THRESHOLD: float = 0.85
    API_KEY: str = "your-secret-api-key"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@db:5432/antifraud"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
