"""Statistical fidelity between original and synthetic tables.

Fidelity score:
    For each numerical column, Kolmogorov–Smirnov statistic D is in [0, 1]
    (0 = identical CDFs). Column score = 1 - D.
    For each categorical column, total-variation distance
    TV = 0.5 * sum(|p - q|) in [0, 1]. Column score = 1 - TV.
    Reported fidelity is the unweighted mean of column scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from app.utils.data_utils import infer_column_types


def _safe_series(s: pd.Series) -> pd.Series:
    return s.dropna()


def fidelity_score(original: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[float, dict[str, float]]:
    numerical, categorical = infer_column_types(original)
    scores: dict[str, float] = {}
    for col in numerical:
        if col not in synthetic.columns:
            continue
        a = pd.to_numeric(_safe_series(original[col]), errors="coerce").dropna()
        b = pd.to_numeric(_safe_series(synthetic[col]), errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        stat = float(ks_2samp(a, b, method="auto").statistic)
        scores[col] = float(np.clip(1.0 - stat, 0.0, 1.0))
    for col in categorical:
        if col not in synthetic.columns:
            continue
        p = original[col].astype(str).value_counts(normalize=True)
        q = synthetic[col].astype(str).value_counts(normalize=True)
        keys = set(p.index) | set(q.index)
        tv = 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
        scores[col] = float(np.clip(1.0 - tv, 0.0, 1.0))
    if not scores:
        return 0.0, {}
    return float(np.mean(list(scores.values()))), scores


def distribution_score(original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """1 / (1 + mean normalized Wasserstein distance) for numerical columns.

    Categorical columns reuse 1 - TV so mixed tables still get a score.
    """
    numerical, categorical = infer_column_types(original)
    parts: list[float] = []
    for col in numerical:
        if col not in synthetic.columns:
            continue
        a = pd.to_numeric(_safe_series(original[col]), errors="coerce").dropna()
        b = pd.to_numeric(_safe_series(synthetic[col]), errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        scale = float(a.std(ddof=0) or 1.0)
        w = wasserstein_distance(a, b) / scale
        parts.append(float(1.0 / (1.0 + w)))
    for col in categorical:
        if col not in synthetic.columns:
            continue
        p = original[col].astype(str).value_counts(normalize=True)
        q = synthetic[col].astype(str).value_counts(normalize=True)
        keys = set(p.index) | set(q.index)
        tv = 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
        parts.append(float(np.clip(1.0 - tv, 0.0, 1.0)))
    return float(np.mean(parts)) if parts else 0.0


def correlation_score(original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """1 - mean absolute difference of Pearson correlation matrices (numerical)."""
    numerical, _ = infer_column_types(original)
    cols = [c for c in numerical if c in synthetic.columns]
    if len(cols) < 2:
        return 1.0
    corr_a = original[cols].corr(numeric_only=True).fillna(0.0)
    corr_b = synthetic[cols].corr(numeric_only=True).fillna(0.0)
    corr_b = corr_b.reindex(index=corr_a.index, columns=corr_a.columns).fillna(0.0)
    diff = np.abs(corr_a.to_numpy() - corr_b.to_numpy())
    # Use upper triangle excluding diagonal.
    n = diff.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    mean_diff = float(diff[iu].mean()) if iu[0].size else 0.0
    return float(np.clip(1.0 - mean_diff, 0.0, 1.0))
