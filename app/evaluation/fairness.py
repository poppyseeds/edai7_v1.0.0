"""Simple group-wise metric comparison. Not a complete fairness audit."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from app.schemas.schemas import FairnessResult


def group_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    groups: pd.Series,
    task_type: str,
) -> FairnessResult:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups.astype(str)})
    metrics: dict[str, dict[str, float | None]] = {}
    for group, part in frame.groupby("group"):
        if task_type != "classification" or part["y_true"].nunique() < 1:
            metrics[str(group)] = {"n": float(len(part))}
            continue
        metrics[str(group)] = {
            "n": float(len(part)),
            "f1": _safe_f1(part["y_true"], part["y_pred"]),
            "precision": _safe_prf(precision_score, part),
            "recall": _safe_prf(recall_score, part),
        }
    f1s = [m["f1"] for m in metrics.values() if m.get("f1") is not None]
    max_gap = float(max(f1s) - min(f1s)) if len(f1s) >= 2 else None
    return FairnessResult(
        sensitive_column=str(groups.name or "group"),
        group_metrics=metrics,
        max_gap=max_gap,
    )


def _safe_f1(y_true: Any, y_pred: Any) -> float | None:
    try:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        return None


def _safe_prf(fn, part: pd.DataFrame) -> float | None:
    try:
        return float(fn(part["y_true"], part["y_pred"], average="macro", zero_division=0))
    except Exception:
        return None
