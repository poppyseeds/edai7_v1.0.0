import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.timeseries.data import TimeSeriesSchema, prepare_sequences, profile_sequences, sequences_to_frame
from app.timeseries.generators import SequenceGenerator, SequenceConfig, TimeGANGenerator
from app.research.sources import ROOT, source_status
from app.utils.data_utils import create_sample_timeseries_dataset


def fixture():
    return pd.DataFrame({"timestamp": list(pd.date_range("2024-01-01", periods=120, freq="h")) * 2,
        "entity": ["a"] * 120 + ["b"] * 120, "measurement": list(np.arange(120.0)) * 2,
        "context": ["sensor"] * 240})


def schema(**kwargs):
    return TimeSeriesSchema(timestamp="timestamp", entity="entity", features=["measurement"],
                            context=["context"], sequence_length=8, **kwargs)


def test_windows_respect_entities_splits_and_training_scaling():
    data = fixture().sample(frac=1, random_state=1).reset_index(drop=True)
    prepared = prepare_sequences(data, schema())
    rows = {role: set(row for window in records for row in window["rows"])
            for role, records in prepared.provenance.items()}
    assert not rows["train"] & rows["test"] and not rows["train"] & rows["validation"]
    assert not rows["validation"] & rows["test"]
    for role, records in prepared.provenance.items():
        for record in records:
            assert data.iloc[record["rows"]].entity.nunique() == 1
            assert pd.to_datetime(record["timestamps"]).is_monotonic_increasing
    assert prepared.windows["test"].max() > 1  # scaler was not fitted on future data
    exported = sequences_to_frame(prepared.windows["train"][:2], prepared.metadata)
    assert len(exported) == 16 and exported.entity.nunique() == 2
    assert list(exported)[-1] == "measurement"


def test_irregularity_detected_and_never_silently_resampled():
    data = fixture()
    data.loc[5, "timestamp"] += pd.Timedelta(minutes=10)
    assert profile_sequences(data, schema())["irregular"]
    with pytest.raises(ValueError, match="Irregular"):
        prepare_sequences(data, schema())


def test_explicit_missing_policy_and_train_only_median():
    data = fixture()
    data.loc[80, "measurement"] = np.nan
    with pytest.raises(ValueError, match="Missing measurements"):
        prepare_sequences(data, schema())
    prepared = prepare_sequences(data, schema(missing_policy="training_median"))
    assert all(np.isfinite(values).all() for values in prepared.windows.values())
    assert prepared.metadata["scaler"]["training_median"] == [35.5]


def test_duplicate_timestamps_are_rejected():
    data = fixture()
    data.loc[1, "timestamp"] = data.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_sequences(data, schema())


def test_bundled_timeseries_sample_matches_documented_schema(tmp_path):
    path = create_sample_timeseries_dataset(tmp_path / "sensor.csv")
    data = pd.read_csv(path)
    documented = TimeSeriesSchema(
        timestamp="timestamp", entity="machine_id",
        features=["temperature_c", "vibration_mm_s", "pressure_kpa"],
        context=["machine_type"], sequence_length=12, stride=6,
    )
    prepared = prepare_sequences(data, documented)
    assert prepared.metadata["profile"]["irregular"] is False
    assert prepared.windows["train"].shape[2] == 3
    assert all(len(prepared.windows[split]) > 0 for split in ("train", "validation", "test"))


@pytest.mark.skipif(source_status("diffusion_ts") is not None or not (ROOT / ".research/env").exists(),
                    reason="Install pinned Diffusion-TS and optional research environment")
def test_actual_diffusion_ts_train_reload_sample(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    prepared = prepare_sequences(fixture(), schema(stride=8))
    config = SequenceConfig(epochs=1, batch_size=4, hidden_dim=8, layers=1, heads=2,
                            timesteps=4, sampling_steps=2)
    model = SequenceGenerator().fit(prepared.windows["train"], config, prepared.metadata)
    restored = SequenceGenerator.load(model.get_metadata()["checkpoint_dir"])
    values = restored.sample(3)
    assert values.shape == (3, 8, 1) and np.isfinite(values).all()
    assert model.get_metadata()["training_seconds"] > 0


def test_actual_timegan_three_phase_training_reload_sample(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    prepared = prepare_sequences(fixture(), schema(stride=8))
    model = TimeGANGenerator().fit(prepared.windows["train"], SequenceConfig(
        epochs=1, batch_size=4, hidden_dim=8, layers=1, heads=2))
    restored = TimeGANGenerator.load(model.get_metadata()["checkpoint_dir"])
    values = restored.sample(3)
    assert values.shape == (3, 8, 1) and np.isfinite(values).all()
    assert set(model.get_metadata()["losses"]) == {"reconstruction", "supervised", "generator", "discriminator", "embedding_joint"}
