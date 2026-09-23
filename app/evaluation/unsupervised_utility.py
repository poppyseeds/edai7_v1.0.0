"""Unsupervised/statistical utility for datasets without a target column.

This is not a supervised ML utility benchmark. It summarizes whether synthetic
data preserves broad distribution, correlation, diversity, and numerical
structure from the original table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.evaluation.diversity import diversity_score
from app.evaluation.fidelity import correlation_score, distribution_score
from app.schemas.schemas import UnsupervisedUtilityResult
from app.utils.data_utils import infer_column_types

METRIC_NOTES = {
    "distribution_score": "Existing distribution similarity metric across numerical and categorical columns.",
    "correlation_score": "Existing numerical correlation preservation score.",
    "diversity_score": "Existing unique-row, category coverage, and numerical range coverage score.",
    "structural_score": "PCA explained-variance similarity for numerical columns when at least two numerical columns exist.",
    "overall_score": "Mean of distribution, correlation, diversity, and structural scores.",
}


def evaluate_unsupervised_utility(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> UnsupervisedUtilityResult:
    dist = distribution_score(original, synthetic)
    corr = correlation_score(original, synthetic)
    div, div_details = diversity_score(original, synthetic)
    structural, structural_details = _pca_structure_score(original, synthetic)
    overall = float(np.mean([dist, corr, div, structural]))
    threshold = get_settings().validation_pass_threshold
    return UnsupervisedUtilityResult(
        distribution_score=round(float(dist), 4),
        correlation_score=round(float(corr), 4),
        diversity_score=round(float(div), 4),
        structural_score=round(float(structural), 4),
        overall_score=round(overall, 4),
        passed=overall >= threshold,
        details={
            "diversity": div_details,
            "structural": structural_details,
        },
        metric_notes=METRIC_NOTES,
    )


def _pca_structure_score(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> tuple[float, dict[str, float | str]]:
    numerical, _ = infer_column_types(original)
    cols = [c for c in numerical if c in synthetic.columns]
    if len(cols) < 2:
        return 1.0, {"status": "not_applicable", "reason": "Fewer than two numerical columns."}

    orig = original[cols].apply(pd.to_numeric, errors="coerce").dropna()
    syn = synthetic[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(orig) < 5 or len(syn) < 5:
        return 1.0, {"status": "not_applicable", "reason": "Not enough complete numerical rows."}

    n_components = min(3, len(cols), len(orig), len(syn))
    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig)
    syn_scaled = scaler.transform(syn)
    pca_orig = PCA(n_components=n_components, random_state=0).fit(orig_scaled)
    pca_syn = PCA(n_components=n_components, random_state=0).fit(syn_scaled)
    diff = np.abs(pca_orig.explained_variance_ratio_ - pca_syn.explained_variance_ratio_)
    mean_diff = float(diff.mean()) if len(diff) else 0.0
    score = float(np.clip(1.0 - mean_diff, 0.0, 1.0))
    return score, {
        "status": "computed",
        "components": float(n_components),
        "mean_explained_variance_difference": mean_diff,
    }
