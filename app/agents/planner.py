"""Deterministic generation planning from DatasetAnalysis."""

from __future__ import annotations

from app.config import get_settings
from app.schemas.schemas import DatasetAnalysis, GenerationPlan
from app.utils.logging_config import get_logger

logger = get_logger("Planner")


class GenerationPlanner:
    def plan(
        self,
        analysis: DatasetAnalysis,
        preferred_generator: str | None = None,
        synthetic_ratio: float | None = None,
        random_state: int | None = None,
        ctgan_epochs: int | None = None,
        tvae_epochs: int | None = None,
    ) -> GenerationPlan:
        settings = get_settings()
        ratio = synthetic_ratio if synthetic_ratio is not None else settings.default_synthetic_ratio
        seed = random_state if random_state is not None else settings.random_state
        ctgan_epochs = ctgan_epochs if ctgan_epochs is not None else settings.ctgan_epochs
        tvae_epochs = tvae_epochs if tvae_epochs is not None else settings.tvae_epochs

        issues = set(analysis.issues)
        num_samples = max(int(analysis.rows * ratio), 20)
        strategy = "full_distribution"
        generator = "gaussian_copula"
        reason_parts: list[str] = []

        if analysis.class_distribution:
            counts = analysis.class_distribution
            majority = max(counts.values())
            minority = min(counts.values())
            if "severe_class_imbalance" in issues or "class_imbalance" in issues:
                strategy = "minority_oversampling"
                majority_n = int(majority * analysis.rows)
                minority_n = int(minority * analysis.rows)
                num_samples = max(num_samples, majority_n - minority_n, 20)
                reason_parts.append("Class imbalance detected; oversampling the minority class.")

        num_samples = min(num_samples, max(analysis.rows * 2, 40))

        if preferred_generator:
            generator = preferred_generator
            reason_parts.append(f"Generator overridden to {generator}.")
        elif "small_dataset" in issues:
            generator = "gaussian_copula"
            reason_parts.append("Small dataset: GaussianCopula is more stable than deep models.")
        elif "severe_class_imbalance" in issues:
            generator = "ctgan"
            reason_parts.append("Severe class imbalance: CTGAN is preferred for mixed tabular data.")
        elif "mixed_column_types" in issues:
            generator = "ctgan"
            reason_parts.append("Mixed numerical/categorical columns: CTGAN or TVAE is suitable.")
        elif len(analysis.numerical_columns) >= len(analysis.categorical_columns):
            generator = "gaussian_copula"
            reason_parts.append("Mostly numerical data: GaussianCopula is a reasonable first attempt.")
        else:
            generator = "tvae"
            reason_parts.append("Categorical-heavy table: TVAE is a reasonable first attempt.")

        epochs = None
        if generator == "ctgan":
            epochs = ctgan_epochs
        elif generator == "tvae":
            epochs = tvae_epochs

        if not reason_parts:
            reason_parts.append("Default generation strategy.")

        reason = " ".join(reason_parts)
        logger.info("Selected %s (%s samples)", generator, num_samples)
        return GenerationPlan(
            generator=generator,  # type: ignore[arg-type]
            num_samples=int(num_samples),
            reason=reason,
            target_column=analysis.target_column,
            target_strategy=strategy,
            epochs=epochs,
            random_state=seed,
            preserve_columns=list(analysis.column_names),
        )
