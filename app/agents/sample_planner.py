"""Adaptive synthetic sample-count planning.

This component only decides how many rows should be generated. It does not
train generators, validate synthetic data, or benchmark downstream models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.config import get_settings
from app.schemas.schemas import DatasetAnalysis
from app.utils.logging_config import get_logger

logger = get_logger("SamplePlanner")


@dataclass
class SampleCountDecision:
    generation_needed: bool
    generation_mode: str
    samples_to_generate: int
    reason: str
    original_rows: int
    target_class: str | None = None
    current_target_count: int | None = None
    desired_target_count: int | None = None
    augmentation_ratio: float | None = None
    max_allowed_samples: int | None = None
    details: dict[str, float | int | str | bool] = field(default_factory=dict)


class SamplePlanner:
    def plan(self, analysis: DatasetAnalysis) -> SampleCountDecision:
        if analysis.target_column and analysis.class_distribution:
            return self._plan_labeled(analysis)
        return self._plan_unlabeled(analysis)

    def cap_samples(self, requested: int, original_rows: int) -> tuple[int, int, bool]:
        settings = get_settings()
        max_allowed = max(1, int(original_rows * settings.max_generation_ratio_per_iteration))
        capped = min(max(requested, 0), max_allowed)
        if requested > max_allowed:
            logger.info(
                "Capped to %s due to per-iteration limit (calculated=%s)",
                capped,
                requested,
            )
        return capped, max_allowed, requested > max_allowed

    def _plan_labeled(self, analysis: DatasetAnalysis) -> SampleCountDecision:
        settings = get_settings()
        rows = analysis.rows
        counts = {
            str(label): int(round(share * rows))
            for label, share in (analysis.class_distribution or {}).items()
        }
        if not counts:
            return self._small_or_skip(analysis, labeled=True)

        majority_label, majority_count = max(counts.items(), key=lambda item: item[1])
        minority_label, minority_count = min(counts.items(), key=lambda item: item[1])
        issues = set(analysis.issues)
        target_ratio = None
        mode = "none"

        if "severe_class_imbalance" in issues:
            target_ratio = settings.minority_target_ratio
            mode = "minority_augmentation"
        elif "class_imbalance" in issues:
            target_ratio = settings.moderate_imbalance_target_ratio
            mode = "moderate_minority_augmentation"

        logger.info("Original rows: %s", rows)
        logger.info("Minority class: %s (%s rows)", minority_label, minority_count)

        if target_ratio is not None:
            current_ratio = minority_count / max(rows, 1)
            desired = max(minority_count, int(math.ceil(rows * target_ratio)))
            calculated = max(0, desired - minority_count)
            logger.info("Target minority count: %s", desired)
            logger.info("Required synthetic rows: %s", calculated)
            final, max_allowed, capped = self.cap_samples(calculated, rows)
            logger.info("Final planned rows: %s", final)
            return SampleCountDecision(
                generation_needed=final > 0,
                generation_mode=mode,
                samples_to_generate=final,
                reason=(
                    f"Increase minority representation toward {target_ratio:.0%} "
                    "of the original dataset population."
                ),
                original_rows=rows,
                target_class=minority_label,
                current_target_count=minority_count,
                desired_target_count=desired,
                augmentation_ratio=final / max(rows, 1),
                max_allowed_samples=max_allowed,
                details={
                    "strategy": "minority_balance",
                    "majority_class": majority_label,
                    "majority_count": majority_count,
                    "current_minority_ratio": current_ratio,
                    "target_minority_ratio": target_ratio,
                    "current_minority_count": minority_count,
                    "desired_minority_count": desired,
                    "synthetic_rows_required": calculated,
                    "calculated_samples": calculated,
                    "capped": capped,
                    "reason": "Minority augmentation is only required below the conservative initial target.",
                },
            )

        return self._small_or_skip(analysis, labeled=True)

    def _plan_unlabeled(self, analysis: DatasetAnalysis) -> SampleCountDecision:
        return self._small_or_skip(analysis, labeled=False)

    def _small_or_skip(
        self,
        analysis: DatasetAnalysis,
        labeled: bool,
    ) -> SampleCountDecision:
        settings = get_settings()
        rows = analysis.rows
        issues = set(analysis.issues)
        if rows < settings.small_dataset_threshold:
            ratio = settings.small_dataset_augmentation_ratio
            mode = "small_dataset_expansion"
            reason = "Small dataset; using configured initial augmentation ratio."
        elif "low_diversity" in issues:
            ratio = settings.low_diversity_augmentation_ratio
            mode = "low_diversity_expansion"
            reason = "Low diversity detected; using configured diversity augmentation ratio."
        elif not issues or issues <= {"mixed_column_types"}:
            logger.info("Original rows: %s", rows)
            logger.info("Final planned rows: 0")
            return SampleCountDecision(
                generation_needed=False,
                generation_mode="none",
                samples_to_generate=0,
                reason=(
                    "No significant dataset issue detected and synthetic augmentation "
                    "is not currently justified."
                ),
                original_rows=rows,
                max_allowed_samples=int(rows * settings.max_generation_ratio_per_iteration),
            )
        else:
            ratio = settings.general_augmentation_ratio
            mode = "dataset_expansion" if not labeled else "general_labeled_expansion"
            reason = "Dataset issues detected; using modest configured augmentation ratio."

        calculated = int(round(rows * ratio))
        logger.info("Original rows: %s", rows)
        logger.info("Required synthetic rows: %s", calculated)
        final, max_allowed, capped = self.cap_samples(calculated, rows)
        logger.info("Final planned rows: %s", final)
        return SampleCountDecision(
            generation_needed=final > 0,
            generation_mode=mode,
            samples_to_generate=final,
            reason=reason,
            original_rows=rows,
            augmentation_ratio=ratio,
            max_allowed_samples=max_allowed,
            details={
                "ratio": ratio,
                "calculated_samples": calculated,
                "capped": capped,
                "labeled": labeled,
            },
        )
