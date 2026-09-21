"""Choose the next generator if augmentation did not improve utility."""

from __future__ import annotations

from app.schemas.schemas import (
    BenchmarkResult,
    GenerationPlan,
    OptimizationDecision,
    ValidationResult,
)
from app.utils.logging_config import get_logger

logger = get_logger("Optimizer")

GENERATOR_CYCLE = ("ctgan", "tvae", "gaussian_copula")


class OptimizationAgent:
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
        next_samples = min(int(current_plan.num_samples * 1.5), current_plan.num_samples + 500)
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
