"""Basic privacy-risk indicators — not differential privacy.

Scores:
    exact_duplicate_rate: fraction of synthetic rows that appear in original
    high_similarity_rate: fraction of sampled synthetic rows whose nearest
        original neighbor (min-max scaled numerical + overlap of categoricals)
        is closer than 0.05
    privacy_score: 1 - 0.7 * duplicate_rate - 0.3 * high_similarity_rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.data_utils import infer_column_types


def basic_privacy_indicators(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    sample_size: int = 80,
    random_state: int = 42,
) -> tuple[float, dict[str, float]]:
    orig_t = original.astype(str).apply(lambda r: tuple(r.tolist()), axis=1)
    syn_t = synthetic.astype(str).apply(lambda r: tuple(r.tolist()), axis=1)
    orig_set = set(orig_t.tolist())
    exact_duplicate_rate = float(sum(1 for t in syn_t if t in orig_set) / max(len(synthetic), 1))

    high_similarity_rate = _nearest_neighbor_rate(
        original, synthetic, sample_size=sample_size, random_state=random_state
    )
    privacy_score = float(
        np.clip(1.0 - 0.7 * exact_duplicate_rate - 0.3 * high_similarity_rate, 0.0, 1.0)
    )
    details = {
        "exact_duplicate_rate": exact_duplicate_rate,
        "high_similarity_rate": high_similarity_rate,
        "membership_risk_indicator": exact_duplicate_rate + 0.5 * high_similarity_rate,
    }
    return privacy_score, details


def _nearest_neighbor_rate(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    sample_size: int,
    random_state: int,
    threshold: float = 0.05,
) -> float:
    numerical, categorical = infer_column_types(original)
    rng = np.random.default_rng(random_state)
    n = min(sample_size, len(synthetic), len(original))
    if n == 0:
        return 0.0
    syn_idx = rng.choice(len(synthetic), size=n, replace=False)
    orig_idx = rng.choice(len(original), size=min(200, len(original)), replace=False)
    syn_sample = synthetic.iloc[syn_idx]
    orig_sample = original.iloc[orig_idx]

    mins = {}
    spans = {}
    for col in numerical:
        a = pd.to_numeric(original[col], errors="coerce")
        mins[col] = float(a.min()) if a.notna().any() else 0.0
        span = float(a.max() - a.min()) if a.notna().any() else 1.0
        spans[col] = span if span else 1.0

    close = 0
    for _, srow in syn_sample.iterrows():
        best = 1e9
        for _, orow in orig_sample.iterrows():
            dist = 0.0
            parts = 0
            for col in numerical:
                sv = pd.to_numeric(pd.Series([srow[col]]), errors="coerce").iloc[0]
                ov = pd.to_numeric(pd.Series([orow[col]]), errors="coerce").iloc[0]
                if pd.isna(sv) or pd.isna(ov):
                    continue
                dist += abs(float(sv) - float(ov)) / spans[col]
                parts += 1
            for col in categorical:
                dist += 0.0 if str(srow[col]) == str(orow[col]) else 1.0
                parts += 1
            if parts:
                best = min(best, dist / parts)
        if best < threshold:
            close += 1
    return float(close / n)
