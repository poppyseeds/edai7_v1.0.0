import json

import pandas as pd
import pytest

from app.agents.benchmark import BenchmarkAgent
from app.agents.generator import GeneratorAgent
from app.agents.planner import GenerationPlanner
from app.agents.validator import ValidationAgent
from app.pipeline.orchestrator import PipelineConfig, run_pipeline
from app.schemas.schemas import BenchmarkResult, ModelMetrics, ValidationResult
from app.utils.exceptions import BenchmarkError, GeneratorError


def table():
    return pd.DataFrame({"row_marker": range(80), "value": [i * 2.5 for i in range(80)], "target": [0, 1] * 40})


def test_selection_never_sees_final_test_and_planner_sees_training_only(monkeypatch, tmp_path):
    seen, fit_rows, planned_rows = [], [], []
    original_planner = GenerationPlanner.plan

    def planner(self, analysis, **kwargs):
        planned_rows.append(analysis.rows)
        return original_planner(self, analysis, **kwargs)

    def generate(self, data, plan, output_path=None):
        fit_rows.append(set(data.row_marker))
        return data.sample(plan.num_samples, replace=True, random_state=1).reset_index(drop=True)

    def validate(self, original, synthetic, **kwargs):
        return ValidationResult(fidelity_score=1, distribution_score=1, correlation_score=1,
            diversity_score=1, privacy_score=1, constraint_score=1, overall_score=1, passed=True,
            details={"privacy": {"exact_duplicate_rate": 0}})

    def compare(self, train, test, synthetic, target_column, task_type, evaluation_role, **kwargs):
        seen.append((evaluation_role, set(test.row_marker)))
        value = (0.6 if len(seen) == 1 else 0.9) if evaluation_role == "validation" else 0.1
        baseline = ModelMetrics(task_type="classification", model_name="test", n_train=len(train),
            n_test=len(test), primary_metric="f1", primary_value=0.5, f1=0.5)
        return BenchmarkResult(baseline=baseline, augmented=baseline.model_copy(update={"primary_value": value}),
            improvement={"primary_value": value - 0.5}, improved=value > 0.5, split_seed=42,
            evaluation_role=evaluation_role)

    monkeypatch.setattr(GenerationPlanner, "plan", planner)
    monkeypatch.setattr(GeneratorAgent, "generate", generate)
    monkeypatch.setattr(ValidationAgent, "validate", validate)
    monkeypatch.setattr(BenchmarkAgent, "compare", compare)
    output = run_pipeline(table(), "target", PipelineConfig(max_iterations=2,
        preferred_generator="bootstrap", output_dir=tmp_path))
    splits = output.result.split_manifest
    train, valid, test = (set(splits[key]) for key in ("train", "validation", "test"))
    assert not train & valid and not train & test and not valid & test
    assert train | valid | test == set(range(80))
    assert fit_rows == [train]  # passing validation stops before warming another model
    assert planned_rows == [len(train)]
    assert seen == [("validation", valid), ("final_test", test)]
    assert output.result.best_iteration == 1  # final-test score cannot restart or change selection
    assert output.result.final_benchmark.evaluation_role == "final_test"
    assert output.result.improved is False  # final test did not confirm validation gain
    assert all(item.benchmark.evaluation_role == "validation" for item in output.result.iterations)
    assert (tmp_path / f"{output.result.run_id}_splits.json").exists()


def test_failure_is_persisted_without_success_or_fake_dataset(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise GeneratorError("fixture failure")

    monkeypatch.setenv("GEMINI_API_KEY", "never-persist-this-test-secret")
    monkeypatch.setattr(GeneratorAgent, "generate", fail)
    with pytest.raises(GeneratorError) as error:
        run_pipeline(table(), "target", PipelineConfig(max_iterations=1,
            preferred_generator="bootstrap", output_dir=tmp_path))
    manifest_path = tmp_path / f"{error.value.run_id}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "GeneratorError"
    assert "never-persist-this-test-secret" not in manifest_path.read_text()
    assert not list(tmp_path.glob("*_result.json"))
    assert not list(tmp_path.glob("*.csv"))


def test_rare_classes_do_not_silently_change_split_protocol():
    data = table()
    data["target"] = [0] * 79 + [1]
    with pytest.raises(BenchmarkError, match="Cannot create"):
        BenchmarkAgent().research_split(data, "target", "classification", 0.25, 0.25, 42)


@pytest.mark.parametrize("kwargs", [{"max_iterations": 0}, {"max_iterations": 51},
    {"test_size": 1}, {"validation_size": 0}, {"random_state": -1}, {"synthetic_ratio": -1}])
def test_invalid_pipeline_configuration(kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)


def test_skipped_run_has_report_and_completed_manifest(tmp_path):
    data = pd.DataFrame({"x": range(2000), "y": [i * 0.37 for i in range(2000)]})
    output = run_pipeline(data, config=PipelineConfig(output_dir=tmp_path))
    assert output.result.final_decision == "skipped_generation_not_needed"
    assert output.result.final_report
    assert json.loads((tmp_path / f"{output.result.run_id}_manifest.json").read_text())["status"] == "completed"


def test_explicit_environment_ratio_still_overrides_adaptive_count(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULT_SYNTHETIC_RATIO", "0.1")
    output = run_pipeline(table().drop(columns="target"), config=PipelineConfig(
        max_iterations=1, preferred_generator="bootstrap", output_dir=tmp_path))
    assert output.result.iterations[0].samples_requested == 8
