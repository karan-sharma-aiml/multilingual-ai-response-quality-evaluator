"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.paths import DEFAULT_LOG_DIR, ENV_FILE

ALLOWED_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


ALLOWED_ENVIRONMENTS: frozenset[str] = frozenset({"development", "test", "production"})
PRODUCTION_ENVIRONMENT: str = "production"


class Settings(BaseSettings):
    """Typed application settings.

    Values are read from the process environment and, when present, from a
    ``.env`` file located at the project root.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Multilingual AI Response Quality Evaluator")
    debug: bool = Field(default=False)
    database_url: str = Field(default="sqlite:///./evaluator.db")
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default=str(DEFAULT_LOG_DIR))

    # Environment & security
    app_env: str = Field(default="development")
    enable_docs: bool = Field(default=True)
    auto_create_tables: bool = Field(default=True)
    api_keys: SecretStr | None = Field(default=None)
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=120, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    trust_proxy_headers: bool = Field(default=False)

    # Performance
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=60, ge=1)
    cache_max_entries: int = Field(default=256, ge=1)
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=20, ge=0)
    db_pool_timeout: int = Field(default=30, ge=1)
    db_pool_recycle: int = Field(default=1800, ge=-1)
    gemini_concurrency: int = Field(default=4, ge=1, le=32)

    # Analytics & reports
    low_quality_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    report_max_rows: int = Field(default=10_000, ge=1)
    report_pdf_max_rows: int = Field(default=500, ge=1)
    report_font_path: str | None = Field(default=None)

    # API
    cors_origins: str = Field(default="*")
    max_batch_rows: int = Field(default=500, ge=1)
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)

    # Gemini integration
    gemini_api_key: SecretStr | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_api_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta"
    )
    gemini_timeout_seconds: float = Field(default=20.0, gt=0)
    gemini_max_retries: int = Field(default=2, ge=0, le=10)
    gemini_retry_backoff_seconds: float = Field(default=1.0, ge=0)
    gemini_max_retry_delay_seconds: float = Field(default=10.0, ge=0)
    gemini_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    gemini_max_input_chars: int = Field(default=6000, ge=100)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(ALLOWED_LOG_LEVELS))
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}")
        return normalized


    @field_validator("app_env")
    @classmethod
    def _validate_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_ENVIRONMENTS:
            allowed = ", ".join(sorted(ALLOWED_ENVIRONMENTS))
            raise ValueError(f"APP_ENV must be one of: {allowed}")
        return normalized

    @model_validator(mode="after")
    def _validate_production_cors(self) -> "Settings":
        if self.is_production and "*" in self.cors_origin_list:
            raise ValueError(
                "CORS_ORIGINS must list explicit origins when APP_ENV=production."
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == PRODUCTION_ENVIRONMENT

    @property
    def api_key_list(self) -> list[str]:
        """Accepted API keys parsed from the comma-separated ``API_KEYS``."""
        if not self.api_keys:
            return []
        keys = [item.strip() for item in self.api_keys.get_secret_value().split(",")]
        return [item for item in keys if item]

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins parsed from the comma-separated ``CORS_ORIGINS``."""
        origins = [item.strip() for item in self.cors_origins.split(",")]
        return [item for item in origins if item] or ["*"]

    @property
    def gemini_configured(self) -> bool:
        """True when a non-empty Gemini API key is available."""
        return bool(
            self.gemini_api_key and self.gemini_api_key.get_secret_value().strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
