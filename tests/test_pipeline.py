import pandas as pd

from app.pipeline.orchestrator import PipelineConfig, run_pipeline
from app.utils.data_utils import create_sample_churn_dataset


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
