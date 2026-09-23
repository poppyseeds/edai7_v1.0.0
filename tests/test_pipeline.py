import pandas as pd
import pytest

from app.pipeline.orchestrator import PipelineConfig, run_pipeline
from app.utils.data_utils import create_sample_churn_dataset, create_unlabeled_customer_dataset


def test_pipeline_runs_and_respects_max_iterations(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    create_sample_churn_dataset(path, n_rows=80, seed=1)
    df = pd.read_csv(path)
    output = run_pipeline(
        df,
        target_column="target",
        config=PipelineConfig(
            max_iterations=2,
            enable_llm=False,
            preferred_generator="gaussian_copula",
            random_state=0,
            ctgan_epochs=1,
            tvae_epochs=1,
            min_improvement=10.0,
            output_dir=tmp_path / "generated",
            dataset_filename="sample.csv",
        ),
    )
    result = output.result
    assert result.evaluation_mode == "labeled"
    assert result.target_column == "target"
    assert result.baseline is not None
    assert result.dataset_analysis.rows == 80
    assert len(result.iterations) <= 2
    assert len(result.iterations) >= 1
    assert result.final_decision in {
        "completed_improved",
        "completed_no_improvement",
        "failed",
    }
    assert output.synthetic_df is not None
    assert list(output.synthetic_df.columns) == list(df.columns)


def test_unlabeled_pipeline_runs_without_target(tmp_path) -> None:
    path = tmp_path / "unlabeled.csv"
    create_unlabeled_customer_dataset(path, n_rows=80, seed=2)
    df = pd.read_csv(path)
    output = run_pipeline(
        df,
        target_column=None,
        config=PipelineConfig(
            max_iterations=1,
            enable_llm=False,
            preferred_generator="gaussian_copula",
            random_state=0,
            output_dir=tmp_path / "generated_unlabeled",
            dataset_filename="unlabeled.csv",
        ),
    )
    result = output.result
    assert result.evaluation_mode == "unlabeled"
    assert result.target_column is None
    assert result.baseline is None
    assert result.iterations[0].benchmark is None
    assert result.iterations[0].unsupervised_utility is not None
    assert output.synthetic_df is not None
    assert list(output.synthetic_df.columns) == list(df.columns)


def test_explicit_non_target_named_label_runs_labeled(tmp_path) -> None:
    path = tmp_path / "churn.csv"
    create_sample_churn_dataset(path, n_rows=80, seed=3)
    df = pd.read_csv(path).rename(columns={"target": "churn"})
    output = run_pipeline(
        df,
        target_column="churn",
        config=PipelineConfig(
            max_iterations=1,
            enable_llm=False,
            preferred_generator="gaussian_copula",
            random_state=0,
            output_dir=tmp_path / "generated_churn",
            dataset_filename="churn.csv",
        ),
    )
    assert output.result.evaluation_mode == "labeled"
    assert output.result.target_column == "churn"
    assert output.result.baseline is not None


def test_auto_detection_reports_detected_target(tmp_path) -> None:
    path = tmp_path / "auto.csv"
    create_sample_churn_dataset(path, n_rows=80, seed=4)
    df = pd.read_csv(path).rename(columns={"target": "churn"})
    output = run_pipeline(
        df,
        target_column=None,
        config=PipelineConfig(
            max_iterations=1,
            enable_llm=False,
            auto_detect_target=True,
            preferred_generator="gaussian_copula",
            random_state=0,
            output_dir=tmp_path / "generated_auto",
            dataset_filename="auto.csv",
        ),
    )
    assert output.result.evaluation_mode == "labeled"
    assert output.result.dataset_analysis.target_detected is True
    assert output.result.dataset_analysis.target_detection_reason is not None


def test_invalid_target_is_clear_error() -> None:
    df = pd.DataFrame({"age": [20] * 30, "income": range(30), "spending": range(100, 130)})
    with pytest.raises(ValueError, match="not in the dataset"):
        run_pipeline(
            df,
            target_column="missing_target",
            config=PipelineConfig(max_iterations=1, enable_llm=False),
        )


def test_unlabeled_numerical_only_pipeline_succeeds(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "age": list(range(30, 110)),
            "income": [40000 + i * 700 for i in range(80)],
            "spending": [8000 + i * 120 for i in range(80)],
        }
    )
    output = run_pipeline(
        df,
        target_column=None,
        config=PipelineConfig(
            max_iterations=1,
            enable_llm=False,
            preferred_generator="gaussian_copula",
            output_dir=tmp_path / "generated_num_only",
        ),
    )
    assert output.result.evaluation_mode == "unlabeled"
    assert output.synthetic_df is not None
