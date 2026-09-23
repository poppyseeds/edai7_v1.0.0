"""Deterministic generation planning from DatasetAnalysis."""

from __future__ import annotations

from app.config import get_settings
from app.agents.sample_planner import SamplePlanner
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
        seed = random_state if random_state is not None else settings.random_state
        ctgan_epochs = ctgan_epochs if ctgan_epochs is not None else settings.ctgan_epochs
        tvae_epochs = tvae_epochs if tvae_epochs is not None else settings.tvae_epochs

        issues = set(analysis.issues)
        sample_decision = SamplePlanner().plan(analysis)
        num_samples = sample_decision.samples_to_generate
        strategy = sample_decision.generation_mode
        generator = "gaussian_copula"
        reason_parts: list[str] = []

        if analysis.class_distribution:
            if "severe_class_imbalance" in issues or "class_imbalance" in issues:
                reason_parts.append("Class imbalance detected; planning targeted minority augmentation.")

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
        reason_parts.append(sample_decision.reason)

        reason = " ".join(reason_parts)
        logger.info("Selected %s (%s samples)", generator, num_samples)
        return GenerationPlan(
            generator=generator,  # type: ignore[arg-type]
            num_samples=int(num_samples),
            reason=reason,
            generation_needed=sample_decision.generation_needed,
            generation_mode=sample_decision.generation_mode,
            target_column=analysis.target_column,
            target_class=sample_decision.target_class,
            target_strategy=strategy,
            original_rows=sample_decision.original_rows,
            current_target_count=sample_decision.current_target_count,
            desired_target_count=sample_decision.desired_target_count,
            samples_to_generate=sample_decision.samples_to_generate,
            augmentation_ratio=sample_decision.augmentation_ratio,
            max_allowed_samples=sample_decision.max_allowed_samples,
            sample_count_reason=sample_decision.reason,
            sample_count_details=sample_decision.details,
            epochs=epochs,
            random_state=seed,
            preserve_columns=list(analysis.column_names),
        )
