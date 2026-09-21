import pandas as pd
from sklearn.datasets import make_classification

from app.evaluation.diversity import diversity_score
from app.evaluation.fidelity import fidelity_score
from app.models.benchmark_model import DownstreamModel


def test_fidelity_identical_tables() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": ["x", "y", "x", "y", "x", "y"]})
    score, details = fidelity_score(df, df.copy())
    assert score > 0.99
    assert details


def test_diversity_unique_rows() -> None:
    orig = pd.DataFrame({"a": [1, 2, 3, 4], "k": ["a", "b", "a", "b"]})
    syn = pd.DataFrame({"a": [1, 2, 3, 4], "k": ["a", "b", "a", "b"]})
    score, details = diversity_score(orig, syn)
    assert 0 <= score <= 1
    assert details["unique_ratio"] == 1.0


def test_benchmark_metrics_classification() -> None:
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(y.astype(str), name="target")
    model = DownstreamModel("classification", random_state=0)
    model.fit(df.iloc[:60], y.iloc[:60])
    metrics = model.evaluate(df.iloc[60:], y.iloc[60:], n_train=60)
    assert metrics.f1 is not None
    assert 0 <= metrics.accuracy <= 1
