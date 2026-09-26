"""REST API for analysis, generation, and the full autonomous pipeline."""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder

from app import __version__
from app.agents.analyzer import DatasetAnalyzer
from app.agents.generator import GeneratorAgent
from app.agents.planner import GenerationPlanner
from app.config import get_settings
from app.pipeline.orchestrator import PipelineConfig, run_pipeline
from app.utils.data_utils import (
    create_sample_churn_dataset,
    create_unlabeled_customer_dataset,
    load_csv_bytes,
    save_csv,
)
from app.utils.exceptions import InvalidDatasetError, SyntheticAIError
from app.utils.logging_config import get_logger, setup_logging

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


def _target_options(target_column: str | None) -> tuple[str | None, bool]:
    if target_column in {None, "", "None"}:
        return None, False
    if target_column in {"__auto__", "Auto Detect", "auto"}:
        return None, True
    return target_column, False


def _read_upload(file: UploadFile, min_rows: int | None = None):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    content = file.file.read()
    try:
        df = load_csv_bytes(content, file.filename)
        if min_rows is not None:
            from app.utils.data_utils import validate_dataframe

            df = validate_dataframe(df, min_rows=min_rows)
    except InvalidDatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    settings = get_settings()
    dest = Path(settings.uploads_dir) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return df, file.filename


def _preview_records(df, rows: int = 20) -> list[dict[str, Any]]:
    """Serialize previews safely when uploaded CSVs contain missing values."""

    preview = df.head(rows).astype(object).where(pd.notna(df.head(rows)), None)
    return jsonable_encoder(preview.to_dict(orient="records"))


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


@app.get("/samples")
def list_samples() -> dict:
    return {
        "samples": [
            {"id": "labeled", "filename": "labeled_customer.csv", "label": "Labeled customer sample"},
            {"id": "unlabeled", "filename": "unlabeled_customer.csv", "label": "Unlabeled customer sample"},
        ]
    }


@app.get("/samples/{sample_id}")
def download_sample(sample_id: str):
    sample_paths = {
        "labeled": PROJECT_ROOT / "datasets" / "sample" / "labeled_customer.csv",
        "unlabeled": PROJECT_ROOT / "datasets" / "sample" / "unlabeled_customer.csv",
    }
    path = sample_paths.get(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    if not path.exists():
        if sample_id == "labeled":
            create_sample_churn_dataset(path)
        else:
            create_unlabeled_customer_dataset(path)
    return FileResponse(path, filename=path.name, media_type="text/csv")


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
    num_samples: int | None = Form(None),
    auto_detect_target: bool = Form(False),
):
    df, filename = _read_upload(file)
    analyzer = DatasetAnalyzer()
    explicit_target, auto = _target_options(target_column)
    analysis = analyzer.analyze(
        df,
        target_column=explicit_target,
        auto_detect_target=auto or auto_detect_target,
    )
    plan = GenerationPlanner().plan(analysis, preferred_generator=generator)
    if num_samples:
        plan.num_samples = num_samples
    settings = get_settings()
    out = Path(settings.generated_dir) / f"manual_{filename}"
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
        plan = GenerationPlanner().plan(analysis, preferred_generator=generator)
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
    max_iterations: int = Form(5),
    sensitive_column: str | None = Form(None),
    preferred_generator: str | None = Form(None),
    enable_llm: bool = Form(True),
    auto_detect_target: bool = Form(False),
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
    target_column: str | None = Form(None),
    max_iterations: int = Form(5),
    sensitive_column: str | None = Form(None),
    preferred_generator: str | None = Form(None),
    enable_llm: bool = Form(True),
    auto_detect_target: bool = Form(False),
):
    """Start the existing synchronous pipeline in a background thread for web clients."""

    df, filename = _read_upload(file)
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
            PIPELINE_JOBS[job_id].update({"status": "failed", "error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive job boundary
            PIPELINE_JOBS[job_id].update({"status": "failed", "error": str(exc)})

    Thread(target=execute, daemon=True).start()
    return PIPELINE_JOBS[job_id]


@app.get("/pipeline-runs/{job_id}")
def get_pipeline_run(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found.")
    return job


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
