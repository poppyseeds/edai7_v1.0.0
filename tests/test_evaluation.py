import pandas as pd
from sklearn.datasets import make_classification

from app.evaluation.diversity import diversity_score
from app.evaluation.fidelity import distribution_details, fidelity_score
from app.evaluation.unsupervised_utility import evaluate_unsupervised_utility
from app.evaluation.advanced_similarity import (
    dependency_similarity,
    discriminator_ensemble,
    mmd_similarity,
)
from app.models.benchmark_model import DownstreamModel
from app.agents.benchmark import BenchmarkAgent
from app.agents.llm_reasoner import LLMReasoner


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


def test_unlabeled_utility_includes_knn_similarity_benchmark() -> None:
    original = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "spend": [100 + index * 5 for index in range(20)],
            "segment": ["A" if index % 2 else "B" for index in range(20)],
        }
    )
    utility = evaluate_unsupervised_utility(original, original.copy(), random_state=0)

    assert utility.knn_similarity_score is not None
    assert utility.knn_discriminator_auc is not None
    assert 0 <= utility.knn_similarity_score <= 1
    assert 0 <= utility.knn_discriminator_auc <= 1


def test_distribution_details_distinguish_matching_and_shifted_columns() -> None:
    real = pd.DataFrame({"amount": list(range(20)), "segment": ["A", "B"] * 10})
    matching = distribution_details(real, real.copy())
    shifted = distribution_details(
        real, pd.DataFrame({"amount": list(range(100, 120)), "segment": ["C"] * 20})
    )

    assert matching["amount"]["ks_statistic"] == 0.0
    assert matching["segment"]["jensen_shannon_divergence"] == 0.0
    assert shifted["amount"]["normalized_wasserstein"] > 1.0
    assert shifted["segment"]["total_variation_distance"] == 1.0


def test_advanced_similarity_prefers_matching_data() -> None:
    real = pd.DataFrame({"x": list(range(20)), "y": [index * 2 for index in range(20)], "group": ["A", "B"] * 10})
    shifted = pd.DataFrame({"x": list(range(100, 120)), "y": list(range(200, 220)), "group": ["C"] * 20})

    matching_mmd = mmd_similarity(real, real.copy(), random_state=0)
    shifted_mmd = mmd_similarity(real, shifted, random_state=0)
    dependency = dependency_similarity(real, real.copy())
    discriminators = discriminator_ensemble(real, real.copy(), random_state=0)

    assert matching_mmd["similarity"] > shifted_mmd["similarity"]
    assert dependency["overall_dependency_score"] is not None
    assert discriminators["mean_discriminator_similarity"] is not None


def test_model_suite_cross_validation_reports_all_cpu_models() -> None:
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=3, n_redundant=0, random_state=1
    )
    df = pd.DataFrame(X, columns=[f"f{index}" for index in range(5)])
    df["target"] = y.astype(str)

    result = BenchmarkAgent().evaluate_model_suite_cv(
        df, "target", "classification", random_state=1, folds=3
    )

    assert set(result.models) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert result.median_primary_value is not None
    assert all(len(item.fold_primary_values) == 3 for item in result.models.values())


def test_gemini_final_report_uses_fallback_when_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reasoner = LLMReasoner()
    fallback = "Deterministic report"

    assert reasoner.explain_final_report({"dataset": {"rows": 20}}, fallback) == fallback
