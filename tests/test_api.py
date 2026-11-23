from io import StringIO

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.utils.data_utils import create_sample_timeseries_dataset


def test_plan_accepts_uploaded_csv_with_missing_cells() -> None:
    df = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "spend": [None if index == 3 else index * 10 for index in range(20)],
            "segment": ["A" if index % 2 else "B" for index in range(20)],
            "target": [index % 2 for index in range(20)],
        }
    )
    csv = StringIO()
    df.to_csv(csv, index=False)

    response = TestClient(app).post(
        "/plan",
        files={"file": ("device_upload.csv", csv.getvalue(), "text/csv")},
        data={"target_column": "target"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["preview"][3]["spend"] is None


def _csv():
    return pd.DataFrame({"value": range(80), "group": ["a", "b"] * 40,
                         "target": [0, 1] * 40}).to_csv(index=False).encode()


def test_sample_catalog_includes_temporal_dataset():
    client = TestClient(app)
    response = client.get("/samples")
    assert response.status_code == 200
    temporal = next(item for item in response.json()["samples"] if item["id"] == "time_series")
    assert temporal["filename"] == "time_series_sensor.csv"
    download = client.get("/samples/time_series")
    assert download.status_code == 200
    sample = pd.read_csv(StringIO(download.text))
    assert len(sample) == 240
    assert {"timestamp", "machine_id", "temperature_c", "vibration_mm_s", "pressure_kpa"} <= set(sample)


def test_time_series_profile_and_unified_background_run(monkeypatch, tmp_path):
    import json
    import time
    from app.research.store import ExperimentStore

    path = create_sample_timeseries_dataset(tmp_path / "temporal.csv")
    content = path.read_bytes()
    monkeypatch.setattr("app.api.main.EXPERIMENT_STORE", ExperimentStore(tmp_path / "experiments"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    fields = {
        "timestamp_column": "timestamp",
        "entity_column": "machine_id",
        "feature_columns": json.dumps(["temperature_c", "vibration_mm_s", "pressure_kpa"]),
        "context_columns": json.dumps(["machine_type"]),
        "sequence_length": "12",
        "stride": "6",
        "missing_policy": "reject",
    }
    client = TestClient(app)
    profile = client.post("/time-series/profile", files={"file": ("temporal.csv", content, "text/csv")}, data=fields)
    assert profile.status_code == 200, profile.text
    assert profile.json()["window_shapes"]["train"] == [22, 12, 3]

    response = client.post("/pipeline-runs", files={"file": ("temporal.csv", content, "text/csv")},
        data={**fields, "modality": "time_series", "preferred_generator": "window_bootstrap",
              "num_sequences": "4", "temporal_epochs": "1", "temporal_batch_size": "4"})
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["modality"] == "time_series"
    deadline = time.monotonic() + 15
    while job["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.05)
        job = client.get(f"/pipeline-runs/{job['job_id']}").json()
    assert job["status"] == "completed", job
    assert job["temporal_result"]["generator"] == "window_bootstrap"
    assert job["temporal_result"]["split_counts"] == {"train": 22, "validation": 6, "test": 6}
    download = client.get(f"/experiments/{job['job_id']}/synthetic-data")
    assert download.status_code == 200 and "synthetic_sequence_id" not in download.text


def test_research_model_inventory_skips_slow_hardware_probe_by_default(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("hardware probe must be explicit")
    monkeypatch.setattr("app.research.planner.worker_hardware", forbidden)
    response = TestClient(app).get("/research/models")
    assert response.status_code == 200
    assert response.json()["worker_hardware"]["status"] == "not_probed"


def test_upload_storage_rejects_paths_and_avoids_collisions(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
    client = TestClient(app)
    for filename in ("../escape.csv", "sub/file.csv"):
        response = client.post("/plan", files={"file": (filename, _csv(), "text/csv")})
        assert response.status_code == 400
    # The multipart parser may strip Windows directory prefixes before our code
    # receives them. Exercise the remaining ingestion boundary directly as well.
    from io import BytesIO
    import pytest
    from fastapi import HTTPException, UploadFile
    from app.api.main import _read_upload
    with pytest.raises(HTTPException) as error:
        _read_upload(UploadFile(filename="C:\\escape.csv", file=BytesIO(_csv())))
    assert error.value.status_code == 400
    for _ in range(2):
        assert client.post("/plan", files={"file": ("same.csv", _csv(), "text/csv")}).status_code == 200
    files = list(tmp_path.iterdir())
    assert len(files) == 2 and all(path.suffix == ".csv" for path in files)
    assert not (tmp_path / "same.csv").exists()


def test_read_limit_is_applied_before_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    response = TestClient(app).post("/plan", files={"file": ("large.csv", b"x" * (1024 * 1024 + 1), "text/csv")})
    assert response.status_code == 400
    assert not list(tmp_path.iterdir())


def test_api_rejects_invalid_configuration_and_unavailable_model(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
    client = TestClient(app)
    for endpoint, fields in [("/run", {"max_iterations": "0"}),
                             ("/run", {"device": "invented"}),
                             ("/run", {"test_size": "1"}),
                             ("/generate", {"num_samples": "0"})]:
        response = client.post(endpoint, files={"file": ("data.csv", _csv(), "text/csv")}, data=fields)
        assert response.status_code == 422
    for name in ("tabddpm", "made_up"):
        response = client.post("/plan", files={"file": ("data.csv", _csv(), "text/csv")}, data={"generator": name})
        assert response.status_code == 400


def test_background_run_final_test_and_manifest_without_external_llm(monkeypatch, tmp_path):
    import time
    from app.api.main import RUNS

    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("GENERATED_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-sent")
    def forbidden_llm():
        raise AssertionError("LLM must be opt-in")
    monkeypatch.setattr("app.pipeline.orchestrator.LLMReasoner", forbidden_llm)
    client = TestClient(app)
    response = client.post("/pipeline-runs", files={"file": ("data.csv", _csv(), "text/csv")},
        data={"target_column": "target", "preferred_generator": "gaussian_copula", "max_iterations": "1"})
    assert response.status_code == 200
    job = response.json()
    deadline = time.monotonic() + 20
    while job["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.05)
        job = client.get(f"/pipeline-runs/{job['job_id']}").json()
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["iterations"][0]["benchmark"]["evaluation_role"] == "validation"
    assert result["final_benchmark"]["evaluation_role"] == "final_test"
    assert result["iterations"][0]["generator_metadata"]["generator"] == "gaussian_copula"
    RUNS.pop(result["run_id"], None)
    assert client.get(f"/runs/{result['run_id']}").status_code == 200
    assert client.get(f"/runs/{result['run_id']}/synthetic-data").status_code == 200
    manifest = client.get(f"/runs/{result['run_id']}/manifest")
    assert manifest.status_code == 200 and manifest.json()["status"] == "completed"
    assert "must-not-be-sent" not in manifest.text
