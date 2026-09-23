"""Pydantic models shared across agents, API, and the dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetAnalysis(BaseModel):
    rows: int
    columns: int
    column_names: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    missing_percentage: float
    missing_by_column: dict[str, float]
    duplicate_percentage: float
    duplicate_rows: int
    target_column: str | None = None
    target_detected: bool = False
    target_detection_reason: str | None = None
    task_type: Literal["classification", "regression"] | None = None
    class_distribution: dict[str, float] | None = None
    unique_value_stats: dict[str, int]
    numerical_stats: dict[str, dict[str, float]]
    categorical_distributions: dict[str, dict[str, float]]
    low_cardinality_columns: list[str]
    correlations: dict[str, dict[str, float]] = Field(default_factory=dict)
    issues: list[str]
    issue_details: dict[str, str] = Field(default_factory=dict)


class GenerationPlan(BaseModel):
    generator: Literal["ctgan", "tvae", "gaussian_copula"]
    num_samples: int
    reason: str
    generation_needed: bool = True
    generation_mode: str = "dataset_expansion"
    target_column: str | None = None
    target_class: str | int | float | None = None
    target_strategy: str = "full_distribution"
    original_rows: int | None = None
    current_target_count: int | None = None
    desired_target_count: int | None = None
    samples_to_generate: int | None = None
    augmentation_ratio: float | None = None
    max_allowed_samples: int | None = None
    sample_count_reason: str | None = None
    sample_count_details: dict[str, Any] = Field(default_factory=dict)
    epochs: int | None = None
    random_state: int = 42
    preserve_columns: list[str] = Field(default_factory=list)
    llm_explanation: str | None = None


class ValidationResult(BaseModel):
    fidelity_score: float
    distribution_score: float
    correlation_score: float
    diversity_score: float
    privacy_score: float
    overall_score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    metric_notes: dict[str, str] = Field(default_factory=dict)


class UnsupervisedUtilityResult(BaseModel):
    label: str = "Unsupervised / Statistical Utility"
    distribution_score: float
    correlation_score: float
    diversity_score: float
    structural_score: float
    overall_score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    metric_notes: dict[str, str] = Field(default_factory=dict)


class ModelMetrics(BaseModel):
    task_type: str
    model_name: str
    n_train: int
    n_test: int
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    primary_metric: str
    primary_value: float


class BenchmarkResult(BaseModel):
    baseline: ModelMetrics
    augmented: ModelMetrics
    improvement: dict[str, float | None]
    improved: bool
    split_seed: int
    notes: str = ""


class OptimizationDecision(BaseModel):
    continue_loop: bool = Field(serialization_alias="continue")
    next_generator: str | None = None
    num_samples: int | None = None
    epochs: int | None = None
    reason: str
    iteration: int
    accept: bool = False

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class AgentLog(BaseModel):
    agent: str
    message: str
    timestamp: str


class IterationRecord(BaseModel):
    iteration: int
    plan: GenerationPlan
    validation: ValidationResult
    benchmark: BenchmarkResult | None = None
    unsupervised_utility: UnsupervisedUtilityResult | None = None
    samples_requested: int
    samples_generated: int
    cumulative_synthetic_rows: int
    decision: str = "retry"
    synthetic_rows: int
    synthetic_path: str | None = None


class ProvenanceFingerprint(BaseModel):
    label: str = "Prototype provenance fingerprint"
    dataset_hash: str
    generator: str
    random_seed: int
    timestamp: str
    num_rows: int
    columns: list[str]


class FairnessResult(BaseModel):
    sensitive_column: str
    group_metrics: dict[str, dict[str, float | None]]
    max_gap: float | None = None
    notes: str = "Simple group comparison only - not a complete fairness audit."


class PipelineResult(BaseModel):
    run_id: str
    timestamp: str
    evaluation_mode: Literal["labeled", "unlabeled"]
    target_column: str | None = None
    dataset_filename: str | None = None
    dataset_analysis: DatasetAnalysis
    baseline: ModelMetrics | None = None
    iterations: list[IterationRecord] = Field(default_factory=list)
    optimization_history: list[OptimizationDecision] = Field(default_factory=list)
    agent_logs: list[AgentLog] = Field(default_factory=list)
    llm_summaries: dict[str, str] = Field(default_factory=dict)
    fairness: FairnessResult | None = None
    provenance: ProvenanceFingerprint | None = None
    final_evaluation: dict[str, Any] = Field(default_factory=dict)
    final_decision: str
    improved: bool
    final_dataset_path: str | None = None
    best_iteration: int | None = None


class PipelineConfigModel(BaseModel):
    max_iterations: int = 3
    random_state: int = 42
    synthetic_ratio: float = 0.3
    ctgan_epochs: int = 10
    tvae_epochs: int = 10
    min_improvement: float = 0.005
    test_size: float = 0.25
    enable_llm: bool = True
    auto_detect_target: bool = False
    preferred_generator: str | None = None
    sensitive_column: str | None = None


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
