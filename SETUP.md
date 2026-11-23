# Setup and execution

Updated through Milestones 3–9; audit date: 2026-10-31. Run commands from the repository root unless a directory change is shown. These instructions preserve the current application environment and keep research dependencies isolated.

## Time-series research environment

On a new machine, keep advanced-model dependencies outside the application virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\setup_research.py diffusion_ts --create-env
.\.venv\Scripts\python.exe scripts\setup_research.py timegan_torch --create-env
.\.venv\Scripts\python.exe scripts\setup_research.py all --check
```

The setup command creates `.research/env` with `--system-site-packages`, installs the minimal overlay from `requirements-research.txt`, clones only requested pinned sources, verifies Git revisions, and checks that Torch/einops are visible. Install an appropriate CPU or CUDA PyTorch build in the base Python first; the setup script deliberately does not choose a hardware-specific Torch package for you.

After starting FastAPI and React, open **Time Series** in the dashboard. Load the sensor sample or upload a CSV, assign timestamp/entity/features/context, validate its chronological windows, then start Diffusion-TS, `timegan_torch`, or the window-bootstrap diagnostic baseline.

## Existing environment

The audited `.venv` uses Python 3.12.4 on Windows. Installed versions: FastAPI 0.141.1, Pydantic 2.13.5, pandas 2.3.3, NumPy 2.5.3, SciPy 1.18.1, sklearn 1.9.1, SDV 1.38.3, PyTorch 2.14.0. Node is 22.17.1. This is an observed snapshot, not a lockfile or guaranteed supported matrix.

Use the existing environment for development:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -ra
.\.venv\Scripts\python.exe run.py --reload
```

API documentation: `http://127.0.0.1:8000/docs`. Existing sample data lives in `datasets/sample`. `python run.py --prepare-sample` regenerates those files; it is unnecessary if they already exist.

For a new baseline environment, Python 3.12 matches the Dockerfile:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency ranges are unpinned. A fresh install is not guaranteed to match this audit. Create a reviewed lock/snapshot before research runs; do not upgrade the current environment to accommodate a model's historical requirements.

## React dashboard

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open `http://localhost:5173`. The development server forwards `/api` to `http://127.0.0.1:8000`. On this machine `npm.ps1` is blocked by PowerShell execution policy; `npm.cmd` works without changing system policy.

`npm.cmd run build` performs TypeScript checking and creates `frontend/dist`. Production hosting must supply an `/api` reverse proxy or set `VITE_API_URL` at build time; Vite's development proxy is not included in the generated static assets. Browser interaction and production hosting were not verified in Milestone 1.

## Streamlit alternative

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/dashboard.py
```

Open `http://localhost:8501`. Streamlit calls the Python pipeline directly and does not require API execution for that path.

## Configuration and local data

Defaults live in `app/config.py`. Environment variables override `.env` and defaults. Results, uploads, and model paths default to `generated`, `datasets/uploads`, and `models` relative to the working directory. Keep uploads/results out of Git. Current adapters do not persist models despite the model-directory setting.

Core processing needs no external credentials. Keep `GEMINI_API_KEY` empty and send `enable_llm=false` for local-only API runs. React, Streamlit, API and pipeline explanation defaults are off; enable them explicitly when desired. With a configured key, analysis/category summaries and measured results can be sent to Gemini. React includes an explicit opt-in checkbox and transfer disclosure.

Milestone 2 removed the credential-like value from `.env.example`; the template key is now empty. If the historical value was real, its owner still needs to revoke/rotate it; repository edits cannot revoke a credential or remove it from Git history. Docker Compose loads the sanitized `.env.example`.

The README, example environment and code default now agree on a minority ratio of 0.20 of the original training population. `CTGAN_EPOCHS` and `TVAE_EPOCHS` also vary by entry/config path. Record resolved values per experiment rather than assuming an example file is authoritative runtime configuration.

## CUDA and RTX 4060

The driver detects an NVIDIA RTX 4060 Laptop GPU with 8188 MiB and driver 616.92. The current PyTorch build returns `torch.version.cuda is None` and `torch.cuda.is_available() == False`. Seeing the GPU in `nvidia-smi` does not mean this Python environment can use it.

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CUDA-enabled model environment is still to be established and tested. Choose a supported PyTorch/CUDA build for the selected adapter and verify a real allocation/training operation there. Do not replace the working CPU environment blindly. No GPU training or peak VRAM measurements were performed in this milestone.

Expose device, batch size, model size, sequence length, and sampling settings in future typed configs. Execute GPU runs sequentially; save checkpoints and fail explicitly on OOM. Do not silently change precision, batch size, or model settings.

## Optional model environments

TabSyn, Diffusion-TS and the separately named `timegan_torch` baseline now have isolated worker checks. `scripts/setup_research.py` provisions pinned source trees and `.research/env` contains only the minimal research overlay. TabDDPM remains unavailable; do not install legacy dependency groups into `.venv`.

The implemented arrangement uses isolated environments with pinned source commits, local subprocess adapters, explicit interpreter selection, and per-run input/config/checkpoint/output directories. Tiny integration checks are software evidence only, not full-size quality claims.

Parquet ingestion, MLflow tracking, and text-model dependencies are also planned. Select minimal optional packages when those paths are implemented.

## Docker

`docker compose up --build` builds the Python API and Streamlit services, exposing ports 8000/8501. It does not build React, configure GPU passthrough, or isolate advanced research models. Docker execution was not tested in this audit. Review the example environment configuration first as described above.

## Generator lifecycle and API configuration

```python
from app.generators.base import GeneratorConfig, SamplingConfig
from app.generators.registry import create_generator

model = create_generator("gaussian_copula")
model.fit(training_frame, GeneratorConfig(random_state=42, device="cpu"))
synthetic = model.sample(100, SamplingConfig())
metadata = model.get_metadata()
```

Only tabular DataFrames are supported by this lifecycle today. Existing
`generate(training_frame, num_samples=100)` remains valid. Fit a model only on
training data. Bootstrap is a named baseline; no adapter failure substitutes it.

For CTGAN/TVAE, `GeneratorConfig(epochs=1, batch_size=20, device="cpu")` is a small
software smoke configuration, not a research training schedule. CTGAN requires
batch sizes divisible by 10. Gaussian Copula/bootstrap reject neural training
options. `device="cuda"` requires a compatible CUDA environment; `auto` explicitly
allows CPU selection when supported. No automatic OOM parameter changes occur.

`POST /plan`, `/generate`, `/run`, and `/pipeline-runs` accept device, batch size,
epochs and seed fields. Run routes also accept `test_size` and `validation_size`;
the latter is a fraction of the remaining development rows, not the full input.
Default labeled splits are 56.25% train, 18.75% validation and 25% final test.
The run recomputes planning on training rows; `/plan` is a full-input preview.

Per-iteration `benchmark` is a validation comparison. `final_benchmark` is the
selected candidate's final-test comparison. `artifact_eligible=false` means a
retained CSV is an audit artifact that failed quality gates. It is not approved
by a privacy guarantee even when true.

Execution manifests and labeled split manifests are stored next to result JSONs.
Use `GET /runs/{run_id}/manifest` to retrieve a known execution record, including
failed runs. These files survive normal exceptions; abruptly terminated jobs
may remain marked running. Job dictionaries and threads are still not durable.

Leave `DEFAULT_SYNTHETIC_RATIO` unset to preserve adaptive counts. Setting it explicitly still supplies a fixed-count override; a direct `PipelineConfig.synthetic_ratio` takes precedence.
