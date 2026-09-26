"""Aggregate-only final report payloads and deterministic fallback text."""

from __future__ import annotations

from typing import Any

from app.schemas.schemas import PipelineResult


def build_final_report_payload(result: PipelineResult) -> dict[str, Any]:
    selected = next((item for item in result.iterations if item.iteration == result.best_iteration), None)
    candidates = []
    for item in result.iterations:
        benchmark = item.benchmark
        candidates.append({
            "generator": item.plan.generator,
            "family": _family(item.plan.generator),
            "status": "selected" if item.iteration == result.best_iteration else item.decision,
            "distribution_score": item.validation.distribution_score,
            "privacy_score": item.validation.privacy_score,
            "validity_score": item.validation.constraint_score,
            "utility_improvement": benchmark.absolute_improvement if benchmark else None,
            "failure_reasons": _failure_reasons(item),
        })
    return {
        "dataset": {
            "rows": result.dataset_analysis.rows,
            "columns": result.dataset_analysis.columns,
            "task": result.dataset_analysis.task_type,
            "target": result.target_column,
            "issues": result.dataset_analysis.issues,
        },
        "objective": {"evaluation_mode": result.evaluation_mode, "final_decision": result.final_decision},
        "candidates": candidates,
        "selected_candidate": candidates[result.best_iteration - 1] if selected and result.best_iteration else None,
        "baseline": result.baseline.model_dump() if result.baseline else None,
        "final_evaluation": result.final_evaluation,
    }


def deterministic_final_report(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    selected = payload.get("selected_candidate")
    candidates = payload.get("candidates", [])
    diagnosis = ", ".join(dataset.get("issues") or []) or "no major structural issues"
    winner = (
        f"{selected['generator']} ({selected['family']})" if selected else "No candidate passed the configured selection gates"
    )
    rejected = [item for item in candidates if item.get("failure_reasons")]
    rejection_text = (
        "; ".join(f"{item['generator']}: {', '.join(item['failure_reasons'])}" for item in rejected)
        or "No explicit hard-gate rejection reasons were recorded."
    )
    return (
        "Executive Summary\n"
        f"The pipeline analyzed {dataset['rows']} rows and {dataset['columns']} columns. Selected result: {winner}.\n\n"
        "Dataset Diagnosis\n"
        f"Observed issues: {diagnosis}. These findings guide generation but do not prove that synthetic data solves them.\n\n"
        "Generator Tournament\n"
        f"Evaluated candidates: {', '.join(item['generator'] for item in candidates) or 'none'}.\n"
        f"Rejected or limited candidates: {rejection_text}\n\n"
        "Privacy and Reliability\n"
        "Privacy and validity are empirical checks, not formal anonymity or differential-privacy guarantees. "
        "The untouched final test set remains separate from generator fitting and selection.\n\n"
        "Recommended Next Actions\n"
        "Review feature-level validation details, inspect any rejected candidate reasons, and obtain domain review before operational use."
    )


def _family(generator: str) -> str:
    return {"bootstrap": "resampling", "gaussian_copula": "statistical", "ctgan": "GAN", "tvae": "VAE"}.get(generator, "optional")


def _failure_reasons(item) -> list[str]:
    privacy = item.validation.details.get("privacy", {})
    reasons = []
    if item.validation.constraint_score is not None and item.validation.constraint_score < 0.95:
        reasons.append("validity_score_below_threshold")
    if item.validation.privacy_score < 0.70:
        reasons.append("privacy_score_below_threshold")
    if privacy.get("exact_duplicate_rate", 0) > 0.05:
        reasons.append("exact_match_rate_above_threshold")
    return reasons
