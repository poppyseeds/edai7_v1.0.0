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

    max_iterations: int = 5
    default_synthetic_ratio: float = 0.3
    minority_target_ratio: float = 0.20
    moderate_imbalance_target_ratio: float = 0.40
    small_dataset_threshold: int = 1000
    small_dataset_augmentation_ratio: float = 0.50
    low_diversity_augmentation_ratio: float = 0.30
    general_augmentation_ratio: float = 0.20
    max_generation_ratio_per_iteration: float = 0.50
    max_total_synthetic_ratio: float = 1.50
    sample_count_increase_factor: float = 1.50
    sample_count_decrease_factor: float = 0.75
    random_state: int = 42
    # Leave unset to use CTGANGenerator's size-aware prototype default.
    ctgan_epochs: int | None = None
    tvae_epochs: int = 10
    max_upload_mb: int = 20
    min_rows: int = 20
    test_size: float = 0.25
    # A candidate must clear both thresholds before the pipeline claims a gain.
    min_improvement: float = 0.02
    min_absolute_improvement: float = 0.01
    validation_pass_threshold: float = 0.55
    log_level: str = "INFO"
    generated_dir: str = "generated"
    uploads_dir: str = "datasets/uploads"
    models_dir: str = "models"


def get_settings() -> Settings:
    return Settings()
