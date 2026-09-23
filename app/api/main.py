"""REST API for analysis, generation, and the full autonomous pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.agents.analyzer import DatasetAnalyzer
from app.agents.generator import GeneratorAgent
from app.agents.planner import GenerationPlanner
from app.config import get_settings
from app.pipeline.orchestrator import PipelineConfig, run_pipeline
from app.utils.data_utils import load_csv_bytes, save_csv
from app.utils.exceptions import InvalidDatasetError, SyntheticAIError
from app.utils.logging_config import setup_logging

setup_logging(get_settings().log_level)

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
    return result.model_dump()


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
        "preview": synthetic.head(10).to_dict(orient="records"),
    }


@app.post("/run")
async def run(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    max_iterations: int = Form(3),
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
        payload["synthetic_preview"] = output.synthetic_df.head(15).to_dict(orient="records")
    return payload


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
