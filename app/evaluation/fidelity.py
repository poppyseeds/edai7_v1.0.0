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
from scipy.spatial.distance import jensenshannon
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


def distribution_details(original: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    """Return stronger per-column distribution diagnostics without changing scores."""

    numerical, categorical = infer_column_types(original)
    details: dict[str, dict[str, float | str]] = {}
    for col in numerical:
        if col not in synthetic.columns:
            details[col] = {"status": "missing_in_synthetic"}
            continue
        real = pd.to_numeric(_safe_series(original[col]), errors="coerce").dropna()
        fake = pd.to_numeric(_safe_series(synthetic[col]), errors="coerce").dropna()
        if len(real) < 2 or len(fake) < 2:
            details[col] = {"status": "not_applicable"}
            continue
        scale = float(real.std(ddof=0) or 1.0)
        quantiles = np.linspace(0.1, 0.9, 9)
        quantile_error = float(np.mean(np.abs(np.quantile(real, quantiles) - np.quantile(fake, quantiles))) / scale)
        low = min(float(real.min()), float(fake.min()))
        high = max(float(real.max()), float(fake.max()))
        if high <= low:
            overlap = 1.0
        else:
            real_hist, edges = np.histogram(real, bins=10, range=(low, high), density=True)
            fake_hist, _ = np.histogram(fake, bins=edges, density=True)
            overlap = float(np.minimum(real_hist, fake_hist).sum() * np.diff(edges).mean())
        details[col] = {
            "status": "computed",
            "ks_statistic": round(float(ks_2samp(real, fake, method="auto").statistic), 6),
            "normalized_wasserstein": round(float(wasserstein_distance(real, fake) / scale), 6),
            "quantile_error": round(quantile_error, 6),
            "histogram_overlap": round(float(np.clip(overlap, 0.0, 1.0)), 6),
        }
    for col in categorical:
        if col not in synthetic.columns:
            details[col] = {"status": "missing_in_synthetic"}
            continue
        p = original[col].dropna().astype(str).value_counts(normalize=True)
        q = synthetic[col].dropna().astype(str).value_counts(normalize=True)
        if p.empty or q.empty:
            details[col] = {"status": "not_applicable"}
            continue
        keys = sorted(set(p.index) | set(q.index))
        real_probs = np.array([p.get(key, 0.0) for key in keys], dtype=float)
        fake_probs = np.array([q.get(key, 0.0) for key in keys], dtype=float)
        tv = 0.5 * float(np.abs(real_probs - fake_probs).sum())
        js = float(jensenshannon(real_probs, fake_probs, base=2.0) ** 2)
        details[col] = {
            "status": "computed",
            "total_variation_distance": round(tv, 6),
            "jensen_shannon_divergence": round(js, 6),
            "category_coverage": round(float(sum(fake_probs > 0) / max(sum(real_probs > 0), 1)), 6),
        }
    return details


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
