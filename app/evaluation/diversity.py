"""Diversity of synthetic rows relative to the original table.

Diversity score is the mean of:
    unique_ratio: unique synthetic rows / n
    category_coverage: mean fraction of original categories retained
    range_coverage: mean fraction of original numerical range covered
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.data_utils import infer_column_types


def diversity_score(original: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[float, dict[str, float]]:
    if synthetic.empty:
        return 0.0, {"unique_ratio": 0.0, "category_coverage": 0.0, "range_coverage": 0.0}

    unique_ratio = float(synthetic.drop_duplicates().shape[0] / max(len(synthetic), 1))
    numerical, categorical = infer_column_types(original)

    coverages: list[float] = []
    for col in categorical:
        if col not in synthetic.columns:
            continue
        orig_vals = set(original[col].dropna().astype(str).unique())
        syn_vals = set(synthetic[col].dropna().astype(str).unique())
        if orig_vals:
            coverages.append(len(orig_vals & syn_vals) / len(orig_vals))
    category_coverage = float(np.mean(coverages)) if coverages else 1.0

    ranges: list[float] = []
    for col in numerical:
        if col not in synthetic.columns:
            continue
        a = pd.to_numeric(original[col], errors="coerce").dropna()
        b = pd.to_numeric(synthetic[col], errors="coerce").dropna()
        span = float(a.max() - a.min()) if len(a) else 0.0
        if span <= 0:
            continue
        syn_span = float(min(a.max(), b.max()) - max(a.min(), b.min())) if len(b) else 0.0
        ranges.append(float(np.clip(syn_span / span, 0.0, 1.0)))
    range_coverage = float(np.mean(ranges)) if ranges else 1.0

    details = {
        "unique_ratio": unique_ratio,
        "category_coverage": category_coverage,
        "range_coverage": range_coverage,
    }
    return float(np.mean(list(details.values()))), details
