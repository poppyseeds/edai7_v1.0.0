"""CPU-bounded multivariate, dependency, and discriminator diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import get_settings
from app.utils.data_utils import infer_column_types


def dependency_similarity(original: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Compare numeric, categorical, and mixed relationships where supported."""

    numerical, categorical = infer_column_types(original)
    numerical = [column for column in numerical if column in synthetic.columns]
    categorical = [column for column in categorical if column in synthetic.columns]
    parts: dict[str, float] = {}
    pearson = _matrix_similarity(original, synthetic, numerical, method="pearson")
    if pearson is not None:
        parts["pearson_similarity"] = pearson
    spearman = _matrix_similarity(original, synthetic, numerical, method="spearman")
    if spearman is not None:
        parts["spearman_similarity"] = spearman
    categorical_score = _categorical_dependency_similarity(original, synthetic, categorical)
    if categorical_score is not None:
        parts["categorical_dependency_similarity"] = categorical_score
    mixed_score = _mixed_dependency_similarity(original, synthetic, numerical, categorical)
    if mixed_score is not None:
        parts["mixed_dependency_similarity"] = mixed_score
    return {
        "status": "computed" if parts else "not_applicable",
        **{key: round(value, 6) for key, value in parts.items()},
        "overall_dependency_score": round(float(np.mean(list(parts.values()))), 6) if parts else None,
    }


def mmd_similarity(
    original: pd.DataFrame, synthetic: pd.DataFrame, random_state: int = 42
) -> dict[str, Any]:
    """RBF-kernel MMD with deterministic sampling to cap quadratic work."""

    settings = get_settings()
    if not settings.enable_mmd:
        return {"status": "disabled", "similarity": None}
    real, fake = _balanced_samples(original, synthetic, settings.evaluation_sample_cap, random_state)
    if len(real) < 3:
        return {"status": "not_applicable", "similarity": None, "reason": "Too few aligned rows."}
    try:
        transformed = _preprocessor(real).fit_transform(pd.concat([real, fake], ignore_index=True))
        matrix = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        real_x, fake_x = matrix[: len(real)], matrix[len(real) :]
        combined = np.vstack([real_x, fake_x])
        distances = np.linalg.norm(combined[:, None, :] - combined[None, :, :], axis=2)
        positive = distances[distances > 0]
        gamma = 1.0 / max(float(np.median(positive)) ** 2, 1e-12) if positive.size else 1.0
        mmd2 = float(
            rbf_kernel(real_x, real_x, gamma=gamma).mean()
            + rbf_kernel(fake_x, fake_x, gamma=gamma).mean()
            - 2.0 * rbf_kernel(real_x, fake_x, gamma=gamma).mean()
        )
        mmd2 = max(mmd2, 0.0)
        return {
            "status": "computed",
            "mmd_squared": round(mmd2, 6),
            "similarity": round(float(1.0 / (1.0 + mmd2)), 6),
            "sample_size_per_dataset": len(real),
        }
    except Exception as exc:
        return {"status": "not_applicable", "similarity": None, "reason": str(exc)}


def discriminator_ensemble(
    original: pd.DataFrame, synthetic: pd.DataFrame, random_state: int = 42
) -> dict[str, Any]:
    """Use multiple classifiers to see whether real and synthetic rows separate."""

    settings = get_settings()
    if not settings.enable_discriminator_ensemble:
        return {"status": "disabled", "models": {}}
    real, fake = _balanced_samples(original, synthetic, settings.evaluation_sample_cap, random_state)
    if len(real) < 4:
        return {"status": "not_applicable", "models": {}, "reason": "Too few aligned rows."}
    data = pd.concat([real, fake], ignore_index=True)
    labels = np.array([0] * len(real) + [1] * len(fake))
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.5, random_state=random_state, stratify=labels
    )
    try:
        preprocessor = _preprocessor(data)
        train_matrix = preprocessor.fit_transform(train_x)
        test_matrix = preprocessor.transform(test_x)
        train_matrix = train_matrix.toarray() if hasattr(train_matrix, "toarray") else np.asarray(train_matrix)
        test_matrix = test_matrix.toarray() if hasattr(test_matrix, "toarray") else np.asarray(test_matrix)
    except Exception as exc:
        return {"status": "not_applicable", "models": {}, "reason": str(exc)}
    models = {
        "logistic_regression": LogisticRegression(max_iter=300, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
    }
    results: dict[str, dict[str, float | str]] = {}
    similarities: list[float] = []
    aucs: list[float] = []
    for name, model in models.items():
        try:
            model.fit(train_matrix, train_y)
            auc = float(roc_auc_score(test_y, model.predict_proba(test_matrix)[:, 1]))
            similarity = float(np.clip(1.0 - 2.0 * abs(auc - 0.5), 0.0, 1.0))
            results[name] = {"status": "computed", "auc": round(auc, 6), "similarity": round(similarity, 6)}
            aucs.append(auc)
            similarities.append(similarity)
        except Exception as exc:
            results[name] = {"status": "not_applicable", "reason": str(exc)}
    return {
        "status": "computed" if similarities else "not_applicable",
        "models": results,
        "mean_discriminator_auc": round(float(np.mean(aucs)), 6) if aucs else None,
        "worst_discriminator_auc": round(float(max(aucs)), 6) if aucs else None,
        "mean_discriminator_similarity": round(float(np.mean(similarities)), 6) if similarities else None,
        "worst_discriminator_similarity": round(float(min(similarities)), 6) if similarities else None,
    }


def _balanced_samples(
    original: pd.DataFrame, synthetic: pd.DataFrame, cap: int, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [column for column in original.columns if column in synthetic.columns]
    count = min(len(original), len(synthetic), cap)
    if not columns or count == 0:
        return pd.DataFrame(), pd.DataFrame()
    return (
        original[columns].sample(n=count, random_state=random_state).reset_index(drop=True),
        synthetic[columns].sample(n=count, random_state=random_state).reset_index(drop=True),
    )


def _preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numerical, categorical = infer_column_types(df)
    transformers = []
    if numerical:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numerical))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    if not transformers:
        raise ValueError("No usable shared columns.")
    return ColumnTransformer(transformers=transformers)


def _matrix_similarity(original: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], method: str) -> float | None:
    if len(columns) < 2:
        return None
    first = original[columns].corr(method=method).fillna(0.0).to_numpy()
    second = synthetic[columns].corr(method=method).fillna(0.0).to_numpy()
    indices = np.triu_indices(len(columns), k=1)
    return float(np.clip(1.0 - np.abs(first[indices] - second[indices]).mean(), 0.0, 1.0))


def _cramers_v(first: pd.Series, second: pd.Series) -> float:
    table = pd.crosstab(first.astype(str), second.astype(str))
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    chi2 = ((observed - expected) ** 2 / np.maximum(expected, 1e-12)).sum()
    return float(np.sqrt((chi2 / observed.sum()) / min(table.shape[0] - 1, table.shape[1] - 1)))


def _categorical_dependency_similarity(original: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]) -> float | None:
    if len(columns) < 2:
        return None
    differences = []
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            differences.append(abs(_cramers_v(original[first], original[second]) - _cramers_v(synthetic[first], synthetic[second])))
    return float(np.clip(1.0 - np.mean(differences), 0.0, 1.0)) if differences else None


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    frame = pd.DataFrame({"category": categories.astype(str), "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if frame.empty or frame["category"].nunique() < 2:
        return 0.0
    total_mean = frame["value"].mean()
    numerator = sum(len(group) * (group["value"].mean() - total_mean) ** 2 for _, group in frame.groupby("category"))
    denominator = ((frame["value"] - total_mean) ** 2).sum()
    return float(np.sqrt(numerator / denominator)) if denominator else 0.0


def _mixed_dependency_similarity(original: pd.DataFrame, synthetic: pd.DataFrame, numerical: list[str], categorical: list[str]) -> float | None:
    if not numerical or not categorical:
        return None
    differences = [
        abs(_correlation_ratio(original[category], original[number]) - _correlation_ratio(synthetic[category], synthetic[number]))
        for number in numerical
        for category in categorical
    ]
    return float(np.clip(1.0 - np.mean(differences), 0.0, 1.0)) if differences else None
