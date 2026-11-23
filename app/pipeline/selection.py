"""Hard-gated, deterministic Pareto selection for candidate evaluations."""

from __future__ import annotations

from typing import Literal

from app.config import get_settings
from app.schemas.schemas import CandidateEvaluation


def apply_hard_gates(candidate: CandidateEvaluation) -> CandidateEvaluation:
    """Reject unsafe or invalid candidates before any utility ranking."""

    settings = get_settings()
    failed: list[str] = []
    if candidate.privacy_score is None or candidate.privacy_score < settings.min_privacy_score:
        failed.append("privacy_score_below_threshold")
    if candidate.exact_match_rate is not None and candidate.exact_match_rate > settings.max_exact_match_rate:
        failed.append("exact_match_rate_above_threshold")
    if candidate.validity_score is None or candidate.validity_score < settings.min_validity_score:
        failed.append("validity_score_below_threshold")
    if candidate.distribution_score is None or candidate.distribution_score < settings.min_distribution_score:
        failed.append("distribution_score_below_threshold")
    candidate.failed_gates = failed
    candidate.passed_hard_gates = not failed
    candidate.failure_reason = ", ".join(failed) if failed else None
    return candidate


def select_candidate(
    candidates: list[CandidateEvaluation], mode: Literal["labeled", "unlabeled"]
) -> CandidateEvaluation | None:
    """Choose a deterministic winner from the non-dominated valid candidates."""

    eligible = [apply_hard_gates(candidate) for candidate in candidates]
    valid = [candidate for candidate in eligible if candidate.passed_hard_gates]
    if not valid:
        return None
    frontier = [candidate for candidate in valid if not any(_dominates(other, candidate, mode) for other in valid if other is not candidate)]
    for candidate in frontier:
        candidate.pareto_optimal = True
    winner = max(frontier, key=lambda candidate: (_priority_score(candidate, mode), candidate.candidate_id))
    winner.selected = True
    return winner


def _dominates(left: CandidateEvaluation, right: CandidateEvaluation, mode: str) -> bool:
    left_values = _objectives(left, mode)
    right_values = _objectives(right, mode)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(a > b for a, b in zip(left_values, right_values))


def _objectives(candidate: CandidateEvaluation, mode: str) -> tuple[float, ...]:
    utility = candidate.downstream_cv.median_primary_value if candidate.downstream_cv else 0.0
    common = (
        candidate.dependency_score or 0.0,
        candidate.distribution_score or 0.0,
        candidate.multivariate_score or 0.0,
        candidate.diversity_score or 0.0,
        candidate.discriminator_score or 0.0,
        candidate.privacy_score or 0.0,
        candidate.validity_score or 0.0,
    )
    return (utility, *common) if mode == "labeled" else common


def _priority_score(candidate: CandidateEvaluation, mode: str) -> float:
    utility = candidate.downstream_cv.median_primary_value if candidate.downstream_cv else 0.0
    if mode == "labeled":
        return (
            0.45 * utility
            + 0.15 * (candidate.dependency_score or 0.0)
            + 0.15 * (candidate.distribution_score or 0.0)
            + 0.10 * (candidate.multivariate_score or 0.0)
            + 0.10 * (candidate.diversity_score or 0.0)
            + 0.05 * (candidate.discriminator_score or 0.0)
        )
    return (
        0.30 * (candidate.discriminator_score or 0.0)
        + 0.25 * (candidate.dependency_score or 0.0)
        + 0.20 * (candidate.distribution_score or 0.0)
        + 0.15 * (candidate.multivariate_score or 0.0)
        + 0.10 * (candidate.diversity_score or 0.0)
    )
