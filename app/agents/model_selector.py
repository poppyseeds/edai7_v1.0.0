"""Explainable dataset-aware ranking of compatible tabular generators."""
from __future__ import annotations

from typing import Any

from app.generators.registry import get_available_generators
from app.schemas.schemas import DatasetAnalysis


BASE_SCORES = {
    "gaussian_copula": 45.0,
    "ctgan": 40.0,
    "tvae": 35.0,
    "tabsyn": 30.0,
    "bootstrap": 5.0,
}


def rank_tabular_generators(
    analysis: DatasetAnalysis,
    generation_mode: str,
    device: str = "cpu",
    include_experimental: bool = False,
    preferred_generator: str | None = None,
) -> list[dict[str, Any]]:
    """Rank only installed, compatible models using observed dataset properties."""
    statuses = get_available_generators(include_experimental)
    conditional = generation_mode in {"minority_augmentation", "moderate_minority_augmentation"}
    eligible = []
    for name, status in statuses.items():
        capabilities = status["capabilities"]
        if not status["available"] or "tabular" not in capabilities["modalities"]:
            continue
        if conditional and not capabilities["supports_conditional_sampling"]:
            continue
        if device == "cuda" and not capabilities["supports_gpu"]:
            continue
        eligible.append(name)

    if preferred_generator and preferred_generator not in eligible:
        return []

    issues = set(analysis.issues)
    numerical = len(analysis.numerical_columns)
    categorical = len(analysis.categorical_columns)
    rows = analysis.rows
    ranked = []
    for name in eligible:
        score = BASE_SCORES.get(name, 20.0)
        reasons: list[str] = []

        def adjust(points: float, reason: str) -> None:
            nonlocal score
            score += points
            if points:
                reasons.append(f"{reason} ({points:+g})")

        if rows < 1_000:
            adjust({"gaussian_copula": 35, "bootstrap": 15, "ctgan": -15,
                    "tvae": -15, "tabsyn": -35}.get(name, 0), "small dataset")
        elif rows >= 5_000:
            adjust({"ctgan": 18, "tvae": 14, "tabsyn": 25,
                    "gaussian_copula": -5}.get(name, 0), "large dataset")

        if "severe_class_imbalance" in issues:
            adjust({"ctgan": 45, "tvae": 15, "gaussian_copula": -5,
                    "bootstrap": -10}.get(name, 0), "severe class imbalance")
        elif "class_imbalance" in issues:
            adjust({"ctgan": 25, "tvae": 12, "gaussian_copula": 3}.get(name, 0),
                   "class imbalance")

        if numerical and categorical:
            adjust({"ctgan": 25, "tvae": 15, "tabsyn": 20,
                    "gaussian_copula": 3}.get(name, 0), "mixed column types")
        elif numerical > categorical:
            adjust({"gaussian_copula": 25, "tabsyn": 15, "tvae": 3}.get(name, 0),
                   "numerical-heavy table")
        elif categorical > numerical:
            adjust({"tvae": 35, "ctgan": 20, "tabsyn": 12,
                    "gaussian_copula": -5}.get(name, 0), "categorical-heavy table")

        if "high_cardinality_categorical" in issues:
            adjust({"ctgan": 15, "tabsyn": 10, "tvae": 5,
                    "gaussian_copula": -10}.get(name, 0), "high-cardinality categories")
        if "skewed_numerical_distributions" in issues:
            adjust({"ctgan": 10, "tvae": 8, "tabsyn": 10,
                    "gaussian_copula": -5}.get(name, 0), "skewed numerical features")
        if "low_diversity" in issues:
            adjust({"gaussian_copula": 10, "bootstrap": 5}.get(name, 0),
                   "low diversity")

        if device == "cuda":
            adjust({"tabsyn": 15, "ctgan": 5, "tvae": 5}.get(name, 0), "CUDA available")
        else:
            adjust({"tabsyn": -30, "ctgan": -5, "tvae": -5,
                    "gaussian_copula": 10, "bootstrap": 8}.get(name, 0), "CPU execution cost")

        ranked.append({"generator": name, "score": round(score, 3), "reasons": reasons})

    ranked.sort(key=lambda item: (-item["score"], item["generator"]))
    if preferred_generator:
        preferred = next(item for item in ranked if item["generator"] == preferred_generator)
        preferred["reasons"] = ["explicit user preference", *preferred["reasons"]]
        ranked = [preferred, *(item for item in ranked if item["generator"] != preferred_generator)]
    return ranked
