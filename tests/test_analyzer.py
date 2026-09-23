import pandas as pd
import pytest

from app.agents.analyzer import DatasetAnalyzer
from app.utils.data_utils import create_sample_churn_dataset


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [21, 22, None, 24, 25, 26, 27, 28, 29, 30, 31, 21],
            "city": ["A", "A", "B", "B", "A", "C", "C", "B", "A", "A", "B", "A"],
            "income": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        }
    )


def test_detects_missing_values(messy_df: pd.DataFrame) -> None:
    analysis = DatasetAnalyzer().analyze(messy_df, target_column="target")
    assert analysis.missing_percentage > 0
    assert analysis.missing_by_column["age"] > 0


def test_detects_duplicates(messy_df: pd.DataFrame) -> None:
    analysis = DatasetAnalyzer().analyze(messy_df, target_column="target")
    assert analysis.duplicate_rows >= 1
    assert analysis.duplicate_percentage > 0


def test_detects_categorical_columns(messy_df: pd.DataFrame) -> None:
    analysis = DatasetAnalyzer().analyze(messy_df, target_column="target")
    assert "city" in analysis.categorical_columns


def test_detects_imbalance(messy_df: pd.DataFrame) -> None:
    analysis = DatasetAnalyzer().analyze(messy_df, target_column="target")
    assert analysis.class_distribution is not None
    assert "severe_class_imbalance" in analysis.issues or "class_imbalance" in analysis.issues


def test_no_target_by_default_does_not_use_last_column() -> None:
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            "income": [30000 + i * 1000 for i in range(12)],
            "spending": [5000 + i * 100 for i in range(12)],
        }
    )
    analysis = DatasetAnalyzer().analyze(df)
    assert analysis.target_column is None
    assert analysis.target_detected is False
    assert analysis.task_type is None


def test_auto_detects_named_target(messy_df: pd.DataFrame) -> None:
    df = messy_df.rename(columns={"target": "churn"})
    analysis = DatasetAnalyzer().analyze(df, auto_detect_target=True)
    assert analysis.target_column == "churn"
    assert analysis.target_detected is True
    assert analysis.target_detection_reason is not None


def test_invalid_explicit_target_raises(messy_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not in the dataset"):
        DatasetAnalyzer().analyze(messy_df, target_column="does_not_exist")


def test_sample_dataset_is_imbalanced(tmp_path) -> None:
    path = tmp_path / "churn.csv"
    create_sample_churn_dataset(path, n_rows=200, seed=0)
    df = pd.read_csv(path)
    analysis = DatasetAnalyzer().analyze(df, target_column="target")
    pos = analysis.class_distribution["1"]
    assert 0.05 <= pos <= 0.20
