"""Choose the next bounded generator and sample-count candidate."""

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

class OptimizationAgent:
    @staticmethod
    def _cap_sample_count(requested: int, original_rows: int) -> int:
        settings = get_settings()
        maximum = max(1, int(original_rows * settings.max_generation_ratio_per_iteration))
        return min(max(1, requested), maximum)

    def _initial_sample_count(self, current_plan: GenerationPlan) -> int:
        configured = current_plan.sample_count_details.get("initial_sample_count")
        if isinstance(configured, int) and configured > 0:
            return configured
        return current_plan.samples_to_generate or current_plan.num_samples

    def _next_candidate(
        self,
        current_plan: GenerationPlan,
        tried_generators: list[str],
    ) -> tuple[str, int] | None:
        original_rows = current_plan.original_rows or current_plan.num_samples
        initial = self._initial_sample_count(current_plan)
        ranking = current_plan.sample_count_details.get("candidate_ranking", [current_plan.generator])
        if not isinstance(ranking, list) or not all(isinstance(name, str) for name in ranking):
            ranking = [current_plan.generator]
        attempted_generators = set(tried_generators)
        for generator in ranking:
            if generator not in attempted_generators:
                return generator, self._cap_sample_count(initial, original_rows)
        return None

    @staticmethod
    def _next_epochs(current_plan: GenerationPlan, next_generator: str) -> int | None:
        if next_generator in {"ctgan", "tvae"}:
            return max(current_plan.epochs or 10, 15)
        return None

    def decide(
        self,
        iteration: int,
        max_iterations: int,
        current_plan: GenerationPlan,
        benchmark: BenchmarkResult | None,
        validation: ValidationResult,
        tried_generators: list[str],
        hard_gate_failures: list[str] | None = None,
    ) -> OptimizationDecision:
        failures = hard_gate_failures or []
        if benchmark and benchmark.improved and validation.passed and not failures:
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason=(f"{current_plan.generator} passed validation quality, utility, and hard gates; "
                        "stop the search before touching another model."),
                iteration=iteration,
                accept=True,
            )
        if iteration >= max_iterations:
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason=(
                    f"Reached max_iterations={max_iterations}; the pipeline will select the "
                    "best candidate that meets all acceptance thresholds."
                ),
                iteration=iteration,
                accept=False,
            )

        candidate = self._next_candidate(current_plan, tried_generators)
        if candidate is None:
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason="Exhausted the bounded generator and sample-count search.",
                iteration=iteration,
                accept=False,
            )

        next_generator, next_samples = candidate
        delta = benchmark.improvement.get("primary_value") if benchmark else None
        delta_text = f", delta={delta:+.4f}" if delta is not None else ""
        status = "met the score threshold" if benchmark and benchmark.improved else "did not meet the score threshold"
        reason = (
            f"{current_plan.generator} with {current_plan.num_samples} samples {status} "
            f"(validation={validation.overall_score:.3f}{delta_text}). "
            f"Continue the configured search with {next_generator} + {next_samples}."
        )
        logger.info(reason)
        return OptimizationDecision(
            continue_loop=True,
            next_generator=next_generator,
            num_samples=next_samples,
            epochs=self._next_epochs(current_plan, next_generator),
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
        hard_gate_failures: list[str] | None = None,
    ) -> OptimizationDecision:
        failures = hard_gate_failures or []
        if validation.passed and utility.passed and not failures:
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason=(f"{current_plan.generator} passed validation, unlabeled utility, and hard gates; "
                        "stop the search before touching another model."),
                iteration=iteration,
                accept=True,
            )
        if iteration >= max_iterations:
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason=(
                    f"Reached max_iterations={max_iterations}; the pipeline will select the "
                    "best candidate that passes the unlabeled quality thresholds."
                ),
                iteration=iteration,
                accept=False,
            )

        candidate = self._next_candidate(current_plan, tried_generators)
        if candidate is None:
            return OptimizationDecision(
                continue_loop=False,
                next_generator=None,
                reason="Exhausted the bounded generator and sample-count search.",
                iteration=iteration,
                accept=False,
            )

        next_generator, next_samples = candidate
        passed = validation.passed and utility.passed
        status = "passed" if passed else "did not pass"
        reason = (
            f"{current_plan.generator} with {current_plan.num_samples} samples {status} "
            f"unlabeled thresholds (validation={validation.overall_score:.3f}, "
            f"utility={utility.overall_score:.3f}). Continue the configured search with "
            f"{next_generator} + {next_samples}."
        )
        logger.info(reason)
        return OptimizationDecision(
            continue_loop=True,
            next_generator=next_generator,
            num_samples=next_samples,
            epochs=self._next_epochs(current_plan, next_generator),
            reason=reason,
            iteration=iteration,
            accept=False,
        )
