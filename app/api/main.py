"""REST API for analysis, generation, and the full autonomous pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder

from app import __version__
from app.agents.analyzer import DatasetAnalyzer
from app.agents.generator import GeneratorAgent
from app.generators.registry import get_available_generators
from app.generators.base import GeneratorConfig
from app.agents.planner import GenerationPlanner
from app.config import get_settings
from app.pipeline.orchestrator import PipelineConfig, run_pipeline
from app.utils.data_utils import (
    create_sample_churn_dataset,
    create_sample_timeseries_dataset,
    create_unlabeled_customer_dataset,
    load_csv_bytes,
    save_csv,
)
from app.utils.exceptions import InvalidDatasetError, SyntheticAIError
from app.utils.logging_config import get_logger, setup_logging
from app.research.store import ExperimentStore
from app.research.planner import model_inventory
from app.research.config import ExperimentConfig
from app.research.engine import submit_experiment
from app.timeseries.data import TimeSeriesSchema, prepare_sequences
from app.timeseries.generators import SequenceConfig, SEQUENCE_GENERATORS

setup_logging(get_settings().log_level)
logger = get_logger("API")

app = FastAPI(
    title="Synthetic-AI Multi-Agent Platform",
    version=__version__,
    description="Autonomous tabular synthetic-data generation, validation, and utility loop.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS: dict[str, dict] = {}
PIPELINE_JOBS: dict[str, dict] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_STORE = ExperimentStore(Path(get_settings().generated_dir) / "experiments")


def _target_options(target_column: str | None) -> tuple[str | None, bool]:
    if target_column in {None, "", "None"}:
        return None, False
    if target_column in {"__auto__", "Auto Detect", "auto"}:
        return None, True
    return target_column, False


def _read_upload(file: UploadFile, min_rows: int | None = None):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    if any(char in file.filename for char in ('/', '\\', ':')) or any(ord(char) < 32 for char in file.filename):
        raise HTTPException(status_code=400, detail="Upload filenames must be plain basenames.")
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024
    content = file.file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=400, detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.")
    try:
        df = load_csv_bytes(content, file.filename)
        if min_rows is not None:
            from app.utils.data_utils import validate_dataframe

            df = validate_dataframe(df, min_rows=min_rows)
    except InvalidDatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    settings = get_settings()
    upload_root = Path(settings.uploads_dir).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    dest = upload_root / f"{uuid4().hex}.csv"
    if dest.resolve().parent != upload_root:
        raise HTTPException(status_code=400, detail="Invalid upload destination.")
    with dest.open("xb") as output:
        output.write(content)
    return df, file.filename


def _preview_records(df, rows: int = 20) -> list[dict[str, Any]]:
    """Serialize previews safely when uploaded CSVs contain missing values."""

    preview = df.head(rows).astype(object).where(pd.notna(df.head(rows)), None)
    return jsonable_encoder(preview.to_dict(orient="records"))


def _column_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return [item.strip() for item in parsed if item.strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


def _time_series_schema(
    timestamp_column: str | None,
    feature_columns: str | None,
    entity_column: str | None,
    context_columns: str | None,
    sequence_length: int,
    stride: int,
    missing_policy: str,
) -> TimeSeriesSchema:
    if not timestamp_column:
        raise ValueError("timestamp_column is required for time-series data.")
    features = _column_list(feature_columns)
    if not features:
        raise ValueError("Select at least one numerical feature column.")
    return TimeSeriesSchema(
        timestamp=timestamp_column,
        entity=entity_column or None,
        features=features,
        context=_column_list(context_columns),
        sequence_length=sequence_length,
        stride=stride,
        missing_policy=missing_policy,
    )


def _temporal_job_payload(identifier: str, filename: str | None = None) -> dict:
    record = EXPERIMENT_STORE.get(identifier)
    payload = {
        "job_id": identifier,
        "run_id": identifier,
        "modality": "time_series",
        "status": record["status"],
        "stage": record.get("stage"),
        "events": record.get("events", []),
        "filename": filename or record.get("dataset_identifier", "time_series.csv"),
    }
    if record["status"] == "completed":
        payload["temporal_result"] = record.get("metrics")
        payload["synthetic_preview"] = (record.get("metrics") or {}).get("synthetic_sequence_preview", [])
    if record["status"] == "failed":
        failure = record.get("failure") or {}
        payload["error"] = failure.get("message", "Time-series experiment failed.")
    return payload


@app.get("/")
def root() -> dict:
    return {
        "name": "Synthetic-AI Multi-Agent Platform",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/generators")
def generators() -> dict:
    """Expose installed and optional generator availability for web clients."""

    return {"generators": get_available_generators(get_settings().enable_experimental_generators)}


@app.get("/research/models")
def research_models(probe_hardware: bool = False) -> dict:
    """Return research-model availability and execution constraints for the dashboard."""
    return model_inventory(probe_hardware=probe_hardware)


@app.get("/experiments")
def experiments(limit: int = 100) -> dict:
    return {"experiments": EXPERIMENT_STORE.list(limit)}


@app.get("/experiments/{experiment_id}")
def experiment(experiment_id: str) -> dict:
    try:
        return EXPERIMENT_STORE.get(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc


@app.get("/samples")
def list_samples() -> dict:
    return {
        "samples": [
            {"id": "labeled", "filename": "labeled_customer.csv", "label": "Labeled customer sample"},
            {"id": "unlabeled", "filename": "unlabeled_customer.csv", "label": "Unlabeled customer sample"},
            {"id": "time_series", "filename": "time_series_sensor.csv", "label": "Multivariate sensor time-series sample"},
        ]
    }


@app.get("/samples/{sample_id}")
def download_sample(sample_id: str):
    sample_paths = {
        "labeled": PROJECT_ROOT / "datasets" / "sample" / "labeled_customer.csv",
        "unlabeled": PROJECT_ROOT / "datasets" / "sample" / "unlabeled_customer.csv",
        "time_series": PROJECT_ROOT / "datasets" / "sample" / "time_series_sensor.csv",
    }
    path = sample_paths.get(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    if not path.exists():
        if sample_id == "labeled":
            create_sample_churn_dataset(path)
        elif sample_id == "time_series":
            create_sample_timeseries_dataset(path)
        else:
            create_unlabeled_customer_dataset(path)
    return FileResponse(path, filename=path.name, media_type="text/csv")


@app.post("/time-series/profile")
async def profile_time_series(
    file: UploadFile = File(...),
    timestamp_column: str = Form(...),
    feature_columns: str = Form(...),
    entity_column: str | None = Form(None),
    context_columns: str | None = Form(None),
    sequence_length: int = Form(12, ge=8, le=4096),
    stride: int = Form(1, ge=1),
    missing_policy: Literal["reject", "forward_fill", "training_median"] = Form("reject"),
):
    frame, filename = _read_upload(file)
    try:
        schema = _time_series_schema(timestamp_column, feature_columns, entity_column,
                                     context_columns, sequence_length, stride, missing_policy)
        prepared = prepare_sequences(frame, schema)
    except (ValueError, SyntheticAIError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "filename": filename,
        "columns": list(frame.columns),
        "schema": schema.model_dump(),
        "profile": prepared.metadata["profile"],
        "window_shapes": {name: list(values.shape) for name, values in prepared.windows.items()},
        "preview": _preview_records(frame),
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    auto_detect_target: bool = Form(False),
):
    df, _ = _read_upload(file)
    explicit_target, auto = _target_options(target_column)
    try:
        result = DatasetAnalyzer().analyze(
            df,
            target_column=explicit_target,
            auto_detect_target=auto or auto_detect_target,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "analysis": result.model_dump(),
        "filename": file.filename,
        "preview": _preview_records(df),
    }


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    generator: str | None = Form(None),
    num_samples: int | None = Form(None, ge=1),
    auto_detect_target: bool = Form(False),
    device: Literal["cpu", "cuda", "auto"] = Form("cpu"),
    batch_size: int | None = Form(None, ge=1),
    epochs: int | None = Form(None, ge=1),
    random_state: int = Form(42, ge=0, le=4294967295),
):
    df, filename = _read_upload(file)
    analyzer = DatasetAnalyzer()
    explicit_target, auto = _target_options(target_column)
    try:
        analysis = analyzer.analyze(
            df,
            target_column=explicit_target,
            auto_detect_target=auto or auto_detect_target,
        )
        plan = GenerationPlanner().plan(analysis, preferred_generator=generator, random_state=random_state,
                generator_config=GeneratorConfig(random_state=random_state, device=device, batch_size=batch_size, epochs=epochs))
    except (SyntheticAIError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if num_samples is not None:
        plan.num_samples = num_samples
        plan.samples_to_generate = num_samples
        plan.generation_needed = True
    settings = get_settings()
    out = Path(settings.generated_dir) / f"manual_{uuid4().hex}.csv"
    try:
        synthetic = GeneratorAgent().generate(df, plan, output_path=out)
    except SyntheticAIError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "plan": plan.model_dump(),
        "rows": len(synthetic),
        "path": str(out),
        "preview": _preview_records(synthetic, 10),
    }


@app.post("/plan")
async def plan_generation(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    generator: str | None = Form(None),
    auto_detect_target: bool = Form(False),
    device: Literal["cpu", "cuda", "auto"] = Form("cpu"),
    batch_size: int | None = Form(None, ge=1),
    epochs: int | None = Form(None, ge=1),
    random_state: int = Form(42, ge=0, le=4294967295),
):
    """Return the real analyzer and planner output without generating data."""

    df, filename = _read_upload(file)
    explicit_target, auto = _target_options(target_column)
    try:
        analysis = DatasetAnalyzer().analyze(
            df,
            target_column=explicit_target,
            auto_detect_target=auto or auto_detect_target,
        )
        plan = GenerationPlanner().plan(analysis, preferred_generator=generator, random_state=random_state,
            generator_config=GeneratorConfig(random_state=random_state, device=device, batch_size=batch_size, epochs=epochs))
    except (SyntheticAIError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Dataset analysis or planning failed for %s", filename)
        raise HTTPException(status_code=400, detail=f"Dataset analysis failed: {exc}") from exc
    return {
        "filename": filename,
        "analysis": analysis.model_dump(),
        "plan": plan.model_dump(),
        "preview": _preview_records(df),
    }


@app.post("/run")
async def run(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    max_iterations: int = Form(5, ge=1, le=50),
    sensitive_column: str | None = Form(None),
    preferred_generator: str | None = Form(None),
    enable_llm: bool = Form(False),
    auto_detect_target: bool = Form(False),
    device: Literal["cpu", "cuda", "auto"] = Form("cpu"),
    batch_size: int | None = Form(None, ge=1),
    epochs: int | None = Form(None, ge=1),
    random_state: int = Form(42, ge=0, le=4294967295),
    test_size: float = Form(0.25, gt=0, lt=1),
    validation_size: float = Form(0.25, gt=0, lt=1),
):
    df, filename = _read_upload(file)
    explicit_target, auto = _target_options(target_column)
    try:
        output = run_pipeline(
            df,
            target_column=explicit_target,
            config=PipelineConfig(
                max_iterations=max_iterations,
                enable_llm=enable_llm,
                auto_detect_target=auto or auto_detect_target,
                preferred_generator=preferred_generator,
                sensitive_column=sensitive_column or None,
                dataset_filename=filename,
                random_state=random_state, test_size=test_size, validation_size=validation_size,
                generator_config=GeneratorConfig(random_state=random_state, device=device, batch_size=batch_size, epochs=epochs),
            ),
        )
    except (SyntheticAIError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    RUNS[output.result.run_id] = {
        "result": output.result,
        "synthetic_df": output.synthetic_df,
        "original_df": output.original_df,
    }
    payload = output.result.model_dump()
    if output.synthetic_df is not None:
        payload["synthetic_preview"] = _preview_records(output.synthetic_df, 15)
    return payload


@app.post("/pipeline-runs")
async def start_pipeline_run(
    file: UploadFile = File(...),
    modality: Literal["tabular", "time_series"] = Form("tabular"),
    target_column: str | None = Form(None),
    max_iterations: int = Form(5, ge=1, le=50),
    sensitive_column: str | None = Form(None),
    preferred_generator: str | None = Form(None),
    enable_llm: bool = Form(False),
    auto_detect_target: bool = Form(False),
    device: Literal["cpu", "cuda", "auto"] = Form("cpu"),
    batch_size: int | None = Form(None, ge=1),
    epochs: int | None = Form(None, ge=1),
    random_state: int = Form(42, ge=0, le=4294967295),
    test_size: float = Form(0.25, gt=0, lt=1),
    validation_size: float = Form(0.25, gt=0, lt=1),
    timestamp_column: str | None = Form(None),
    entity_column: str | None = Form(None),
    feature_columns: str | None = Form(None),
    context_columns: str | None = Form(None),
    sequence_length: int = Form(12, ge=8, le=4096),
    stride: int = Form(1, ge=1),
    missing_policy: Literal["reject", "forward_fill", "training_median"] = Form("reject"),
    num_sequences: int = Form(10, ge=2, le=100000),
    temporal_epochs: int = Form(5, ge=1),
    temporal_batch_size: int = Form(16, ge=1),
    hidden_dim: int = Form(32, ge=4),
    layers: int = Form(2, ge=1),
    heads: int = Form(4, ge=1),
    timesteps: int = Form(50, ge=2),
    sampling_steps: int = Form(10, ge=1),
    forecast_target: str | None = Form(None),
    forecast_horizon: int = Form(1, ge=1),
):
    """Start a tabular or time-series pipeline behind one polling contract."""

    df, filename = _read_upload(file)
    if modality == "time_series":
        try:
            schema = _time_series_schema(timestamp_column, feature_columns, entity_column,
                                         context_columns, sequence_length, stride, missing_policy)
            # Validate before queueing so schema errors return synchronously.
            prepare_sequences(df, schema)
            generator_name = preferred_generator or "diffusion_ts"
            if generator_name not in SEQUENCE_GENERATORS:
                raise ValueError(f"Unknown time-series generator '{generator_name}'.")
            inventory = model_inventory()
            status = inventory["models"].get(generator_name, {})
            if not status.get("available"):
                raise ValueError(f"Time-series generator '{generator_name}' is unavailable: {status.get('reason')}.")
            sequence_config = SequenceConfig(
                random_state=random_state,
                device=device,
                epochs=epochs or temporal_epochs,
                batch_size=batch_size or temporal_batch_size,
                hidden_dim=hidden_dim,
                layers=layers,
                heads=heads,
                timesteps=timesteps,
                sampling_steps=sampling_steps,
            )
            config = ExperimentConfig(
                modality="time_series",
                generator=generator_name,
                num_samples=num_sequences,
                seed=random_state,
                sequence_schema=schema,
                sequence_config=sequence_config,
                forecast_target=forecast_target or None,
                forecast_horizon=forecast_horizon,
                selection="explicit",
            )
            record = submit_experiment(df, config, EXPERIMENT_STORE, dataset_identifier=filename)
        except (ValueError, SyntheticAIError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job_id = record["id"]
        PIPELINE_JOBS[job_id] = {
            "job_id": job_id,
            "experiment_id": job_id,
            "modality": "time_series",
            "status": record["status"],
            "filename": filename,
        }
        return _temporal_job_payload(job_id, filename)

    explicit_target, auto = _target_options(target_column)
    job_id = uuid4().hex
    PIPELINE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": filename,
    }

    def execute() -> None:
        PIPELINE_JOBS[job_id]["status"] = "running"
        try:
            output = run_pipeline(
                df,
                target_column=explicit_target,
                config=PipelineConfig(
                    max_iterations=max_iterations,
                    enable_llm=enable_llm,
                    auto_detect_target=auto or auto_detect_target,
                    preferred_generator=preferred_generator,
                    sensitive_column=sensitive_column or None,
                    dataset_filename=filename,
                    random_state=random_state, test_size=test_size, validation_size=validation_size,
                    generator_config=GeneratorConfig(random_state=random_state, device=device, batch_size=batch_size, epochs=epochs),
                ),
            )
            RUNS[output.result.run_id] = {
                "result": output.result,
                "synthetic_df": output.synthetic_df,
                "original_df": output.original_df,
            }
            PIPELINE_JOBS[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "filename": filename,
                "run_id": output.result.run_id,
                "result": output.result.model_dump(),
                "synthetic_preview": (
                    _preview_records(output.synthetic_df)
                    if output.synthetic_df is not None
                    else []
                ),
            }
        except (SyntheticAIError, ValueError) as exc:
            PIPELINE_JOBS[job_id].update({"status": "failed", "error": str(exc), "run_id": getattr(exc, "run_id", None)})
        except Exception as exc:  # pragma: no cover - defensive job boundary
            PIPELINE_JOBS[job_id].update({"status": "failed", "error": str(exc), "run_id": getattr(exc, "run_id", None)})

    Thread(target=execute, daemon=True).start()
    return dict(PIPELINE_JOBS[job_id])


@app.get("/pipeline-runs/{job_id}")
def get_pipeline_run(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if job is not None and job.get("modality") == "time_series":
        return _temporal_job_payload(job_id, job.get("filename"))
    if job is None:
        try:
            record = EXPERIMENT_STORE.get(job_id)
            if record.get("config", {}).get("modality") == "time_series":
                return _temporal_job_payload(job_id)
        except KeyError:
            pass
        raise HTTPException(status_code=404, detail="Pipeline run not found.")
    return job


@app.get("/experiments/{experiment_id}/synthetic-data")
def download_experiment_synthetic(experiment_id: str):
    try:
        record = EXPERIMENT_STORE.get(experiment_id)
        directory = EXPERIMENT_STORE.directory(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    if record["status"] != "completed" or not (directory / "synthetic.csv").exists():
        raise HTTPException(status_code=404, detail="Synthetic time-series data is not available.")
    return FileResponse(directory / "synthetic.csv",
                        filename=f"{experiment_id}_synthetic_time_series.csv", media_type="text/csv")


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id in RUNS:
        return RUNS[run_id]["result"].model_dump()
    settings = get_settings()
    path = Path(settings.generated_dir) / f"{run_id}_result.json"
    if path.exists():
        return JSONResponse(content=__import__("json").loads(path.read_text(encoding="utf-8")))
    raise HTTPException(status_code=404, detail="Run not found.")


@app.get("/runs/{run_id}/synthetic-data")
def download_synthetic(run_id: str):
    result = None
    if run_id in RUNS:
        result = RUNS[run_id]["result"]
        df = RUNS[run_id]["synthetic_df"]
        if df is not None:
            settings = get_settings()
            path = Path(settings.generated_dir) / f"{run_id}_download.csv"
            save_csv(df, path)
            return FileResponse(path, filename=f"{run_id}_synthetic.csv", media_type="text/csv")
    settings = get_settings()
    stored = Path(settings.generated_dir) / f"{run_id}_result.json"
    if stored.exists():
        import json

        data = json.loads(stored.read_text(encoding="utf-8"))
        csv_path = data.get("final_dataset_path")
        if csv_path and Path(csv_path).exists():
            return FileResponse(
                csv_path, filename=f"{run_id}_synthetic.csv", media_type="text/csv"
            )
    raise HTTPException(status_code=404, detail="Synthetic dataset not found.")


@app.get("/runs/{run_id}/manifest")
def get_run_manifest(run_id: str):
    import re
    import json
    if not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=404, detail="Run manifest not found.")
    path = Path(get_settings().generated_dir) / f"{run_id}_manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run manifest not found.")
    return json.loads(path.read_text(encoding="utf-8"))
