"""Deterministic generation planning from DatasetAnalysis."""

from __future__ import annotations

from app.config import get_settings
from app.generators.base import GeneratorConfig
from app.generators.registry import require_generator
from app.generators.runtime import resolve_device
from app.agents.sample_planner import SamplePlanner
from app.agents.model_selector import rank_tabular_generators
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
        generator_config: GeneratorConfig | None = None,
    ) -> GenerationPlan:
        settings = get_settings()
        seed = random_state if random_state is not None else settings.random_state
        ctgan_epochs = ctgan_epochs if ctgan_epochs is not None else settings.ctgan_epochs
        tvae_epochs = tvae_epochs if tvae_epochs is not None else settings.tvae_epochs

        issues = set(analysis.issues)
        sample_decision = SamplePlanner().plan(analysis)
        if (
            synthetic_ratio is not None
            and sample_decision.generation_needed
            and sample_decision.generation_mode
            not in {"minority_augmentation", "moderate_minority_augmentation"}
        ):
            requested = max(0, int(round(analysis.rows * synthetic_ratio)))
            final, max_allowed, capped = SamplePlanner().cap_samples(requested, analysis.rows)
            sample_decision.samples_to_generate = final
            sample_decision.augmentation_ratio = final / max(analysis.rows, 1)
            sample_decision.max_allowed_samples = max_allowed
            sample_decision.generation_needed = final > 0
            sample_decision.details = {
                **sample_decision.details,
                "synthetic_ratio": synthetic_ratio,
                "synthetic_ratio_requested_rows": requested,
                "synthetic_ratio_applied": True,
                "capped": capped,
            }
        num_samples = sample_decision.samples_to_generate
        strategy = sample_decision.generation_mode
        reason_parts: list[str] = []

        if analysis.class_distribution:
            if "severe_class_imbalance" in issues or "class_imbalance" in issues:
                reason_parts.append("Class imbalance detected; planning targeted minority augmentation.")

        cfg = generator_config or GeneratorConfig(random_state=seed)
        if preferred_generator:
            # Preserve precise unknown/unavailable/experimental errors before
            # applying dataset compatibility ranking.
            require_generator(preferred_generator, settings.enable_experimental_generators)
        ranking = rank_tabular_generators(
            analysis, strategy, cfg.device, settings.enable_experimental_generators,
            preferred_generator=preferred_generator,
        )
        if not ranking:
            raise ValueError("No available tabular generator supports this dataset and generation request.")
        generator = ranking[0]["generator"]
        ranking_names = [item["generator"] for item in ranking]
        reason_parts.append(
            f"Dataset-aware ranking selected {generator}; candidates: {', '.join(ranking_names)}."
        )
        if ranking[0]["reasons"]:
            reason_parts.append("Top-model factors: " + "; ".join(ranking[0]["reasons"]) + ".")

        registration = require_generator(generator, settings.enable_experimental_generators)
        if registration.capabilities.execution == "in_process":
            resolve_device(cfg.device, registration.capabilities.supports_gpu, registration.capabilities.supports_cpu)
        allowed_options = registration.capabilities.configuration_options
        if cfg.batch_size is not None and "batch_size" not in allowed_options:
            raise ValueError(f"{generator} does not support batch_size.")

        epochs = None
        if generator == "ctgan":
            epochs = ctgan_epochs
        elif generator == "tvae":
            epochs = tvae_epochs
        if cfg.epochs is not None:
            if "epochs" not in allowed_options:
                raise ValueError(f"{generator} does not support epochs.")
            epochs = cfg.epochs

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
            sample_count_details={
                **sample_decision.details,
                "initial_sample_count": num_samples,
                "candidate_ranking": ranking_names,
                "candidate_scores": ranking,
                "selection_policy": "dataset_aware_then_validation_early_stop",
            },
            epochs=epochs,
            random_state=seed,
            preserve_columns=list(analysis.column_names),
            generator_config=cfg,
        )
