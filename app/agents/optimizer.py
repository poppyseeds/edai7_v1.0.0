"""Choose the next generator if augmentation did not improve utility."""

from __future__ import annotations

from app.config import get_settings
from app.schemas.schemas import (
    BenchmarkResult,
    GenerationPlan,
    OptimizationDecision,
    UnsupervisedUtilityResult,
    ValidationResult,
)
from app.utils.logging_config import get_logger

logger = get_logger("Optimizer")

GENERATOR_CYCLE = ("ctgan", "tvae", "gaussian_copula")


class OptimizationAgent:
    def _next_sample_count(
        self,
        current_plan: GenerationPlan,
        original_rows: int,
        improved_signal: bool,
    ) -> int:
        settings = get_settings()
        factor = (
            settings.sample_count_decrease_factor
            if improved_signal
            else settings.sample_count_increase_factor
        )
        requested = max(1, int(current_plan.num_samples * factor))
        max_allowed = max(1, int(original_rows * settings.max_generation_ratio_per_iteration))
        return min(requested, max_allowed)

    def decide(
        self,
        iteration: int,
        max_iterations: int,
        current_plan: GenerationPlan,
        benchmark: BenchmarkResult | None,
        validation: ValidationResult,
        tried_generators: list[str],
    ) -> OptimizationDecision:
        if benchmark is not None and benchmark.improved:
            logger.info("Improvement detected")
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason="Downstream performance improved; accepting this synthetic dataset.",
                iteration=iteration,
                accept=True,
            )

        if iteration >= max_iterations:
            logger.info("Maximum iterations reached")
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason=f"Reached max_iterations={max_iterations} without a required improvement.",
                iteration=iteration,
                accept=False,
            )

        remaining = [g for g in GENERATOR_CYCLE if g not in tried_generators]
        if current_plan.generator in remaining:
            remaining = [g for g in remaining if g != current_plan.generator]
        if not remaining:
            logger.info("No unused generators remain")
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason="All generator families have been attempted.",
                iteration=iteration,
                accept=False,
            )

        nxt = remaining[0]
        primary_delta = None
        if benchmark is not None:
            primary_delta = benchmark.improvement.get("primary_value")
        worsened = primary_delta is not None and primary_delta < 0
        next_samples = self._next_sample_count(
            current_plan,
            original_rows=current_plan.original_rows or current_plan.num_samples,
            improved_signal=worsened,
        )
        next_epochs = current_plan.epochs
        if nxt in {"ctgan", "tvae"}:
            next_epochs = max(current_plan.epochs or 10, 15)

        reason = (
            f"{current_plan.generator} did not improve downstream utility "
            f"(validation overall={validation.overall_score:.3f}). "
            f"Trying {nxt} with {next_samples} samples."
        )
        logger.info("Selecting next generator: %s", nxt)
        return OptimizationDecision(
            continue_loop=True,
            next_generator=nxt,
            num_samples=next_samples,
            epochs=next_epochs,
            reason=reason,
            iteration=iteration,
            accept=False,
        )

    def decide_unlabeled(
        self,
        iteration: int,
        max_iterations: int,
        current_plan: GenerationPlan,
        validation: ValidationResult,
        utility: UnsupervisedUtilityResult,
        tried_generators: list[str],
    ) -> OptimizationDecision:
        if validation.passed and utility.passed:
            logger.info("Synthetic-data quality improved")
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason=(
                    "Synthetic-data quality and unsupervised/statistical utility "
                    "passed configured thresholds."
                ),
                iteration=iteration,
                accept=True,
            )

        if iteration >= max_iterations:
            logger.info("Maximum iterations reached")
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason=f"Reached max_iterations={max_iterations} without passing quality thresholds.",
                iteration=iteration,
                accept=False,
            )

        remaining = [g for g in GENERATOR_CYCLE if g not in tried_generators]
        if not remaining:
            logger.info("No unused generators remain")
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason="All generator families have been attempted.",
                iteration=iteration,
                accept=False,
            )

        nxt = remaining[0]
        worsened = utility.overall_score < validation.overall_score
        next_samples = self._next_sample_count(
            current_plan,
            original_rows=current_plan.original_rows or current_plan.num_samples,
            improved_signal=worsened,
        )
        next_epochs = current_plan.epochs
        if nxt in {"ctgan", "tvae"}:
            next_epochs = max(current_plan.epochs or 10, 15)

        reason = (
            f"{current_plan.generator} did not pass unlabeled quality thresholds "
            f"(validation={validation.overall_score:.3f}, utility={utility.overall_score:.3f}). "
            f"Trying {nxt} with {next_samples} samples."
        )
        logger.info("Selecting next unlabeled generator: %s", nxt)
        return OptimizationDecision(
            continue_loop=True,
            next_generator=nxt,
            num_samples=next_samples,
            epochs=next_epochs,
            reason=reason,
            iteration=iteration,
            accept=False,
        )
