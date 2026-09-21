"""Compare baseline vs augmented primary metrics."""

from __future__ import annotations

from app.schemas.schemas import ModelMetrics


def primary_improvement(baseline: ModelMetrics, augmented: ModelMetrics) -> float:
    if baseline.task_type == "regression" and baseline.primary_metric == "rmse":
        if not baseline.primary_value:
            return 0.0
        return (baseline.primary_value - augmented.primary_value) / abs(baseline.primary_value)
    denom = abs(baseline.primary_value) if baseline.primary_value else 1.0
    if denom == 0:
        return float(augmented.primary_value)
    return (augmented.primary_value - baseline.primary_value) / denom
