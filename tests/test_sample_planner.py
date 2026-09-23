import pandas as pd

from app.agents.analyzer import DatasetAnalyzer
from app.agents.optimizer import OptimizationAgent
from app.agents.planner import GenerationPlanner
from app.agents.sample_planner import SamplePlanner
from app.schemas.schemas import GenerationPlan, OptimizationDecision, ValidationResult


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


def test_severe_imbalance_plans_half_majority_minority_target() -> None:
    df = _labeled_counts(9000, 1000)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    decision = SamplePlanner().plan(analysis)
    assert decision.generation_mode == "minority_augmentation"
    assert decision.current_target_count == 1000
    assert decision.desired_target_count == 4500
    assert decision.samples_to_generate == 3500


def test_moderate_imbalance_does_not_full_balance() -> None:
    df = _labeled_counts(7000, 3000)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    decision = SamplePlanner().plan(analysis)
    assert decision.generation_mode == "moderate_minority_augmentation"
    assert decision.desired_target_count == 2800
    assert decision.samples_to_generate == 0


def test_extreme_imbalance_is_capped(monkeypatch) -> None:
    monkeypatch.setenv("MINORITY_TARGET_RATIO", "1.0")
    df = _labeled_counts(99000, 1000)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    decision = SamplePlanner().plan(analysis)
    assert decision.desired_target_count == 99000
    assert decision.details["calculated_samples"] == 98000
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


def test_optimizer_changes_sample_count_after_poor_attempt() -> None:
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
    assert decision.next_generator != "ctgan"
    assert decision.num_samples != 2000


def test_generation_planner_records_sample_count_details() -> None:
    df = _labeled_counts(9000, 1000)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    plan = GenerationPlanner().plan(analysis, preferred_generator="gaussian_copula")
    assert plan.samples_to_generate == 3500
    assert plan.current_target_count == 1000
    assert plan.desired_target_count == 4500
    assert plan.generation_mode == "minority_augmentation"
