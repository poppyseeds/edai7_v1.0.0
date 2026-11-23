"""Central configuration. Values can be overridden with environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    max_iterations: int = Field(default=5, ge=1, le=50)
    # Unset preserves adaptive planning; an explicit value is a fixed-count override.
    default_synthetic_ratio: float | None = Field(default=None, ge=0)
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
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    # Leave unset to use the adapter's 100-epoch default.
    ctgan_epochs: int | None = Field(default=None, ge=1)
    tvae_epochs: int = Field(default=10, ge=1)
    max_upload_mb: int = Field(default=20, ge=1)
    min_rows: int = 20
    test_size: float = Field(default=0.25, gt=0, lt=1)
    # A candidate must clear both thresholds before the pipeline claims a gain.
    min_improvement: float = 0.02
    min_absolute_improvement: float = 0.01
    validation_pass_threshold: float = 0.55
    evaluation_sample_cap: int = 500
    enable_mmd: bool = True
    enable_discriminator_ensemble: bool = True
    cv_folds: int = 3
    enable_candidate_search: bool = True
    max_generator_candidates: int = 8
    max_search_trials: int = 8
    enable_experimental_generators: bool = False
    min_privacy_score: float = 0.70
    max_exact_match_rate: float = 0.05
    min_validity_score: float = 0.95
    min_distribution_score: float = 0.60
    log_level: str = "INFO"
    generated_dir: str = "generated"
    uploads_dir: str = "datasets/uploads"
    models_dir: str = "models"


def get_settings() -> Settings:
    return Settings()
