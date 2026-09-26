"""Compare synthetic data to the original table with documented statistical metrics."""

from __future__ import annotations

import pandas as pd

from app.config import get_settings
from app.evaluation.diversity import diversity_score
from app.evaluation.fidelity import (
    correlation_score,
    distribution_details,
    distribution_score,
    fidelity_score,
)
from app.evaluation.privacy import basic_privacy_indicators
from app.evaluation.constraints import discover_constraints, validate_constraints
from app.evaluation.advanced_similarity import (
    dependency_similarity,
    discriminator_ensemble,
    mmd_similarity,
)
from app.schemas.schemas import ValidationResult
from app.utils.logging_config import get_logger

logger = get_logger("Validator")

METRIC_NOTES = {
    "fidelity_score": "Mean of (1 - KS) for numerical columns and (1 - total variation) for categorical columns.",
    "distribution_score": "Mean of 1/(1+normalized Wasserstein) for numerical columns and (1 - TV) for categoricals.",
    "correlation_score": "1 - mean absolute difference of Pearson correlation matrices (numerical columns).",
    "diversity_score": "Mean of unique-row ratio, category coverage, and numerical range coverage.",
    "privacy_score": "Basic privacy risk indicator: 1 - 0.7*exact_duplicate_rate - 0.3*high_similarity_rate. Not DP.",
}


class ValidationAgent:
    def validate(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        random_state: int = 42,
    ) -> ValidationResult:
        if synthetic.empty:
            raise ValueError("Synthetic dataset is empty.")
        fid, fid_details = fidelity_score(original, synthetic)
        dist = distribution_score(original, synthetic)
        dist_details = distribution_details(original, synthetic)
        corr = correlation_score(original, synthetic)
        div, div_details = diversity_score(original, synthetic)
        priv, priv_details = basic_privacy_indicators(
            original, synthetic, random_state=random_state
        )
        constraints = discover_constraints(original)
        constraint_details = validate_constraints(original, synthetic, constraints)
        dependency_details = dependency_similarity(original, synthetic)
        mmd_details = mmd_similarity(original, synthetic, random_state=random_state)
        discriminator_details = discriminator_ensemble(original, synthetic, random_state=random_state)
        overall = float((fid + dist + corr + div + priv) / 5.0)
        threshold = get_settings().validation_pass_threshold
        passed = overall >= threshold
        logger.info("Fidelity score: %.3f", fid)
        logger.info("Overall validation score: %.3f (passed=%s)", overall, passed)
        return ValidationResult(
            fidelity_score=round(fid, 4),
            distribution_score=round(dist, 4),
            correlation_score=round(corr, 4),
            diversity_score=round(div, 4),
            privacy_score=round(priv, 4),
            overall_score=round(overall, 4),
            passed=passed,
            constraint_score=constraint_details["overall_validity_score"],
            details={
                "column_fidelity": fid_details,
                "distribution": dist_details,
                "diversity": div_details,
                "privacy": priv_details,
                "constraints": constraint_details,
                "dependencies": dependency_details,
                "multivariate": mmd_details,
                "discriminators": discriminator_details,
            },
            metric_notes=METRIC_NOTES,
        )
