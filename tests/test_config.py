"""
Unit tests for antifraud_rag/core/config.py - Settings configuration.
"""

from antifraud_rag.core.config import Settings


class TestSettings:
    """Tests for the Settings class."""

    def test_settings_with_valid_env_vars(self):
        """Test Settings loads correctly with valid environment variables."""
        settings = Settings(
            EMBEDDING_MODEL_URL="https://api.openai.com/v1/embeddings",
            EMBEDDING_MODEL_API_KEY="test-key-123",
        )

        assert settings.EMBEDDING_MODEL_URL == "https://api.openai.com/v1/embeddings"
        assert settings.EMBEDDING_MODEL_API_KEY == "test-key-123"
        assert settings.EMBEDDING_MODEL_NAME == "text-embedding-ada-002"
        assert settings.EMBEDDING_DIMENSION == 1536

    def test_settings_with_custom_values(self):
        """Test Settings accepts custom values for all fields."""
        settings = Settings(
            EMBEDDING_MODEL_URL="https://custom.api.com/embeddings",
            EMBEDDING_MODEL_API_KEY="custom-key",
            EMBEDDING_MODEL_NAME="custom-embedding-model",
            EMBEDDING_DIMENSION=2048,
            HIGH_RISK_THRESHOLD=0.75,
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/testdb",
        )

        assert settings.EMBEDDING_MODEL_NAME == "custom-embedding-model"
        assert settings.EMBEDDING_DIMENSION == 2048
        assert settings.HIGH_RISK_THRESHOLD == 0.75
        assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@localhost:5432/testdb"

    def test_settings_default_values(self):
        """Test Settings has correct default values."""
        settings = Settings(
            EMBEDDING_MODEL_URL="https://api.test.com",
            EMBEDDING_MODEL_API_KEY="test-key",
        )

        assert settings.EMBEDDING_MODEL_NAME == "text-embedding-ada-002"
        assert settings.EMBEDDING_DIMENSION == 1536
        assert settings.HIGH_RISK_THRESHOLD == 0.85
        assert "postgresql" in settings.DATABASE_URL

    def test_settings_threshold_boundaries(self):
        """Test HIGH_RISK_THRESHOLD accepts valid boundary values."""
        settings_low = Settings(
            EMBEDDING_MODEL_URL="https://api.test.com",
            EMBEDDING_MODEL_API_KEY="test-key",
            HIGH_RISK_THRESHOLD=0.0,
        )
        assert settings_low.HIGH_RISK_THRESHOLD == 0.0

        settings_high = Settings(
            EMBEDDING_MODEL_URL="https://api.test.com",
            EMBEDDING_MODEL_API_KEY="test-key",
            HIGH_RISK_THRESHOLD=1.0,
        )
        assert settings_high.HIGH_RISK_THRESHOLD == 1.0

    def test_settings_extra_fields_ignored(self):
        """Test Settings ignores extra fields (extra='ignore')."""
        settings = Settings(
            EMBEDDING_MODEL_URL="https://api.test.com",
            EMBEDDING_MODEL_API_KEY="test-key",
            UNKNOWN_FIELD="ignored",
        )
        assert not hasattr(settings, "UNKNOWN_FIELD")

    def test_settings_dimension_accepts_positive_values(self):
        """Test EMBEDDING_DIMENSION accepts positive values."""
        settings = Settings(
            EMBEDDING_MODEL_URL="https://api.test.com",
            EMBEDDING_MODEL_API_KEY="test-key",
            EMBEDDING_DIMENSION=1,
        )
        assert settings.EMBEDDING_DIMENSION == 1
