import json
import numpy as np
import pandas as pd
import pytest

from app.research.config import ExperimentConfig
from app.research.store import ExperimentStore
from app.research.engine import execute_experiment
from app.research.evaluation import sequence_metrics, distance_privacy, forecasting_utility
from app.research.ingestion import read_table
from tests.test_timeseries import fixture, schema


def run(frame, config, tmp_path):
    store = ExperimentStore(tmp_path)
    identifier = store.create(config.model_dump(), "test-hash", "test-fixture")
    frame.to_csv(store.directory(identifier) / "input.csv", index=False)
    return execute_experiment(identifier, store), store


def test_temporal_metrics_detect_destroyed_order_with_same_marginal():
    original = np.tile(np.sin(np.arange(32)/3)[None, :, None], (12, 1, 1))
    same = sequence_metrics(original, original)
    shuffled = original[:, np.random.default_rng(4).permutation(32)]
    changed = sequence_metrics(original, shuffled)
    assert same["per_feature"][0]["ks_statistic"] == 0
    assert same["per_feature"][0]["acf_error"] == 0
    assert changed["per_feature"][0]["ks_statistic"] == 0
    assert changed["per_feature"][0]["acf_error"] > 0.1
    assert same["cross_correlation_error"] is None


def test_constant_temporal_metrics_are_not_fabricated():
    metrics = sequence_metrics(np.ones((4, 16, 2)), np.ones((4, 16, 2)))
    assert metrics["per_feature"][0]["acf_error"] is None
    assert metrics["per_feature"][0]["spectral_js_divergence"] is None
    assert metrics["cross_correlation_error"] is None


def test_privacy_records_copying_and_holdout_attack():
    rng = np.random.default_rng(2)
    training = rng.normal(size=(20, 8, 2)).astype("float32")
    heldout = rng.normal(size=(20, 8, 2)).astype("float32")
    privacy = distance_privacy(training, heldout, training.copy())
    assert privacy["exact_training_match_rate"] == 1
    assert privacy["near_training_match_rate"] == 1
    assert privacy["distance_attack_auc"] > 0.9


def test_tabular_tstr_and_persisted_exports(tmp_path):
    frame = pd.DataFrame({"value": np.arange(80)*0.3, "group": ["a", "b"]*40, "target": [0, 1]*40})
    config = ExperimentConfig(target="target", sensitive_column="group", generator="bootstrap", num_samples=40)
    result, store = run(frame, config, tmp_path)
    assert result["status"] == "completed", result["failure"]
    assert result["metrics"]["utility"]["synthetic_only"]["n_train"] == 40
    assert result["metrics"]["utility"]["evaluation_role"] == "final_test"
    assert (store.directory(result["id"]) / "report.json").exists()
    assert ExperimentStore(tmp_path).get(result["id"])["status"] == "completed"


def test_temporal_forecasting_and_copy_baseline_not_approved(tmp_path):
    config = ExperimentConfig(modality="time_series", generator="window_bootstrap", num_samples=8,
                              sequence_schema=schema(stride=8), forecast_target="measurement")
    result, _ = run(fixture(), config, tmp_path)
    assert result["status"] == "completed", result["failure"]
    assert result["metrics"]["artifact_eligible"] is False
    assert result["metrics"]["utility"]["metrics"]["synthetic_only"]["mae"] >= 0
    assert result["metrics"]["privacy"]["exact_training_match_rate"] == 1


def test_failed_experiments_and_interrupted_jobs_remain_visible(tmp_path):
    result, store = run(pd.DataFrame({"x": range(40), "y": range(40)}),
                        ExperimentConfig(generator="not_installed"), tmp_path)
    assert result["status"] == "failed" and result["failure"]["type"]
    queued = store.create({}, "hash", "queued")
    store.recover_interrupted()
    assert store.get(queued)["status"] == "failed"
    assert len(store.list()) == 2


def test_csv_and_parquet_ingestion_roundtrip():
    from io import BytesIO
    frame = fixture()
    assert len(read_table(frame.to_csv(index=False).encode(), "fixture.csv")) == len(frame)
    pytest.importorskip("pyarrow")
    buffer = BytesIO(); frame.to_parquet(buffer)
    assert len(read_table(buffer.getvalue(), "fixture.parquet")) == len(frame)
