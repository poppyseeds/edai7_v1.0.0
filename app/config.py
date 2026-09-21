"""Central configuration. Values can be overridden with environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    max_iterations: int = 3
    default_synthetic_ratio: float = 0.3
    random_state: int = 42
    ctgan_epochs: int = 10
    tvae_epochs: int = 10
    max_upload_mb: int = 20
    min_rows: int = 20
    test_size: float = 0.25
    min_improvement: float = 0.005
    validation_pass_threshold: float = 0.55
    log_level: str = "INFO"
    generated_dir: str = "generated"
    uploads_dir: str = "datasets/uploads"
    models_dir: str = "models"


def get_settings() -> Settings:
    return Settings()
