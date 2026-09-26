import pandas as pd

from app.agents.analyzer import DatasetAnalyzer
from app.agents.optimizer import OptimizationAgent
from app.agents.planner import GenerationPlanner
from app.agents.sample_planner import SamplePlanner
from app.schemas.schemas import (
    BenchmarkResult,
    GenerationPlan,
    ModelMetrics,
    OptimizationDecision,
    ValidationResult,
)


def _labeled_counts(zeros: int, ones: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": list(range(zeros + ones)),
            "target": [0] * zeros + [1] * ones,
        }
    )


def _validation(score: float = 0.5) -> ValidationResult:
    return ValidationResult(
        fidelity_score=score,
        distribution_score=score,
        correlation_score=score,
        diversity_score=score,
        privacy_score=score,
        overall_score=score,
        passed=False,
    )


def test_severe_imbalance_uses_conservative_minority_population_target() -> None:
    df = _labeled_counts(450, 50)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    decision = SamplePlanner().plan(analysis)
    assert decision.generation_mode == "minority_augmentation"
    assert decision.current_target_count == 50
    assert decision.desired_target_count == 100
    assert decision.samples_to_generate == 50
    assert decision.details["strategy"] == "minority_balance"
    assert decision.details["target_minority_ratio"] == 0.20


def test_minority_already_at_target_is_not_forced_to_augment() -> None:
    df = _labeled_counts(800, 200)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    decision = SamplePlanner().plan(analysis)
    assert decision.generation_mode in {
        "minority_augmentation",
        "moderate_minority_augmentation",
    }
    assert decision.current_target_count == 200
    assert decision.desired_target_count == 200
    assert decision.samples_to_generate == 0


def test_extreme_imbalance_is_capped(monkeypatch) -> None:
    monkeypatch.setenv("MINORITY_TARGET_RATIO", "0.75")
    df = _labeled_counts(99000, 1000)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    decision = SamplePlanner().plan(analysis)
    assert decision.desired_target_count == 75000
    assert decision.details["calculated_samples"] == 74000
    assert decision.samples_to_generate == 50000
    assert decision.details["capped"] is True


def test_unlabeled_small_dataset_uses_half_augmentation() -> None:
    df = pd.DataFrame({"x": range(500), "y": [i * 2 for i in range(500)]})
    analysis = DatasetAnalyzer().analyze(df)
    decision = SamplePlanner().plan(analysis)
    assert decision.generation_mode == "small_dataset_expansion"
    assert decision.samples_to_generate == 250


def test_unlabeled_low_diversity_uses_thirty_percent() -> None:
    df = pd.DataFrame(
        {
            "segment": ["A"] * 5000 + ["B"] * 5000,
            "city": ["Pune"] * 10000,
            "spending_band": ["low"] * 9000 + ["high"] * 1000,
        }
    )
    analysis = DatasetAnalyzer().analyze(df)
    decision = SamplePlanner().plan(analysis)
    assert "low_diversity" in analysis.issues
    assert decision.generation_mode == "low_diversity_expansion"
    assert decision.samples_to_generate == 3000


def test_healthy_large_dataset_can_skip_generation() -> None:
    df = pd.DataFrame(
        {
            "x": range(2000),
            "y": [i * 3 for i in range(2000)],
            "z": [(i % 17) + i * 0.01 for i in range(2000)],
        }
    )
    analysis = DatasetAnalyzer().analyze(df)
    decision = SamplePlanner().plan(analysis)
    assert decision.generation_needed is False
    assert decision.samples_to_generate == 0


def test_optimizer_retries_same_generator_with_different_sample_count() -> None:
    plan = GenerationPlan(
        generator="ctgan",
        num_samples=2000,
        reason="test",
        original_rows=10000,
    )
    decision = OptimizationAgent().decide(
        iteration=1,
        max_iterations=3,
        current_plan=plan,
        benchmark=None,
        validation=_validation(0.4),
        tried_generators=["ctgan"],
    )
    assert isinstance(decision, OptimizationDecision)
    assert decision.continue_loop is True
    assert decision.next_generator == "ctgan"
    assert decision.num_samples != 2000


def test_optimizer_switches_generator_after_same_family_retry() -> None:
    plan = GenerationPlan(
        generator="ctgan",
        num_samples=25,
        reason="test",
        original_rows=500,
        target_column="target",
        target_class=1,
        generation_mode="minority_augmentation",
        sample_count_details={
            "initial_sample_count": 50,
            "candidate_history": [
                {"generator": "ctgan", "num_samples": 50},
                {"generator": "ctgan", "num_samples": 25},
            ],
        },
    )
    decision = OptimizationAgent().decide(
        iteration=2,
        max_iterations=3,
        current_plan=plan,
        benchmark=None,
        validation=_validation(0.4),
        tried_generators=["ctgan"],
    )
    assert decision.next_generator == "tvae"
    assert decision.num_samples == 50
    assert plan.target_column == "target"
    assert plan.target_class == 1


def test_optimizer_continues_search_after_a_passing_benchmark() -> None:
    metrics = ModelMetrics(
        task_type="classification",
        model_name="random_forest",
        n_train=50,
        n_test=20,
        primary_metric="f1",
        primary_value=0.70,
        f1=0.70,
    )
    benchmark = BenchmarkResult(
        baseline=metrics.model_copy(update={"primary_value": 0.60, "f1": 0.60}),
        augmented=metrics,
        improvement={"primary_value": 0.10},
        improved=True,
        split_seed=42,
    )
    decision = OptimizationAgent().decide(
        iteration=1,
        max_iterations=3,
        current_plan=GenerationPlan(
            generator="gaussian_copula", num_samples=50, reason="test", original_rows=100
        ),
        benchmark=benchmark,
        validation=_validation(0.8),
        tried_generators=["gaussian_copula"],
    )

    assert decision.continue_loop is True
    assert decision.accept is False


def test_generation_planner_records_sample_count_details() -> None:
    df = _labeled_counts(9000, 1000)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    plan = GenerationPlanner().plan(analysis, preferred_generator="gaussian_copula")
    assert plan.samples_to_generate == 1000
    assert plan.current_target_count == 1000
    assert plan.desired_target_count == 2000
    assert plan.generation_mode == "minority_augmentation"


def test_generation_planner_applies_synthetic_ratio_to_non_minority_expansion() -> None:
    df = pd.DataFrame({"feature": range(200), "other": range(200)})
    analysis = DatasetAnalyzer().analyze(df)
    plan = GenerationPlanner().plan(analysis, synthetic_ratio=0.30)
    assert plan.num_samples == 60
    assert plan.sample_count_details["synthetic_ratio_applied"] is True
