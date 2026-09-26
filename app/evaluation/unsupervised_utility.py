"""Unsupervised/statistical utility for datasets without a target column.

There is no target column in this mode, so accuracy and F1 would be misleading.
Instead, this evaluates whether a K-nearest-neighbors (KNN) classifier can tell
original and synthetic rows apart, alongside distribution and structure checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from app.config import get_settings
from app.evaluation.diversity import diversity_score
from app.evaluation.fidelity import correlation_score, distribution_score
from app.schemas.schemas import UnsupervisedUtilityResult
from app.utils.data_utils import infer_column_types

METRIC_NOTES = {
    "distribution_score": "How closely individual column values match (distribution similarity). Higher is better.",
    "correlation_score": "How closely relationships between number columns match (correlation preservation). Higher is better.",
    "diversity_score": "Whether the generated table has enough varied rows and categories. Higher is better.",
    "structural_score": "Whether broad numerical patterns match, measured with PCA (a way to summarize data shape). Higher is better.",
    "knn_similarity_score": "KNN similarity: a K-nearest-neighbors classifier tries to separate original from synthetic rows. Higher means it could not easily tell them apart.",
    "knn_discriminator_auc": "KNN discriminator AUC: 0.50 means the classifier is guessing; values closer to 1.00 mean it can tell the two datasets apart.",
    "overall_score": "Average of the available unlabeled similarity checks. This is a data-quality comparison, not prediction accuracy.",
}


def evaluate_unsupervised_utility(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    random_state: int | None = None,
) -> UnsupervisedUtilityResult:
    dist = distribution_score(original, synthetic)
    corr = correlation_score(original, synthetic)
    div, div_details = diversity_score(original, synthetic)
    structural, structural_details = _pca_structure_score(original, synthetic)
    knn_similarity, knn_auc, knn_details = _knn_similarity_score(
        original, synthetic, random_state=random_state
    )
    scores = [dist, corr, div, structural]
    if knn_similarity is not None:
        scores.append(knn_similarity)
    overall = float(np.mean(scores))
    threshold = get_settings().validation_pass_threshold
    return UnsupervisedUtilityResult(
        distribution_score=round(float(dist), 4),
        correlation_score=round(float(corr), 4),
        diversity_score=round(float(div), 4),
        structural_score=round(float(structural), 4),
        knn_similarity_score=(round(knn_similarity, 4) if knn_similarity is not None else None),
        knn_discriminator_auc=(round(knn_auc, 4) if knn_auc is not None else None),
        overall_score=round(overall, 4),
        passed=overall >= threshold,
        details={
            "diversity": div_details,
            "structural": structural_details,
            "knn": knn_details,
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


def _knn_similarity_score(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    random_state: int | None,
) -> tuple[float | None, float | None, dict[str, float | int | str]]:
    """Measure whether KNN can distinguish an original row from a synthetic one."""

    columns = [column for column in original.columns if column in synthetic.columns]
    sample_size = min(len(original), len(synthetic))
    if sample_size < 4 or not columns:
        return None, None, {
            "status": "not_applicable",
            "reason": "At least four rows from both datasets and one shared column are required.",
        }

    seed = get_settings().random_state if random_state is None else random_state
    original_sample = original[columns].sample(n=sample_size, random_state=seed).copy()
    synthetic_sample = synthetic[columns].sample(n=sample_size, random_state=seed).copy()
    combined = pd.concat([original_sample, synthetic_sample], ignore_index=True)
    labels = np.array([0] * sample_size + [1] * sample_size)
    numerical, categorical = infer_column_types(original_sample)
    transformers = []
    if numerical:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        return None, None, {"status": "not_applicable", "reason": "No usable shared columns."}

    try:
        train_x, test_x, train_y, test_y = train_test_split(
            combined,
            labels,
            test_size=0.5,
            random_state=seed,
            stratify=labels,
        )
        model = Pipeline(
            [
                ("preprocess", ColumnTransformer(transformers=transformers)),
                ("knn", KNeighborsClassifier(n_neighbors=min(5, len(train_x)))),
            ]
        )
        model.fit(train_x, train_y)
        auc = float(roc_auc_score(test_y, model.predict_proba(test_x)[:, 1]))
    except Exception as exc:
        return None, None, {"status": "not_applicable", "reason": str(exc)}

    similarity = float(np.clip(1.0 - (2.0 * abs(auc - 0.5)), 0.0, 1.0))
    return similarity, auc, {
        "status": "computed",
        "sample_size_per_dataset": sample_size,
        "test_rows": int(len(test_x)),
        "n_neighbors": min(5, len(train_x)),
    }
