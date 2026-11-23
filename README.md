# Synthetic-AI - Autonomous Multi-Agent Synthetic Data Platform

College prototype for adaptive synthetic data generation, validation, and continuous dataset improvement on tabular CSV data.

The platform now supports two modes:

1. Labeled Mode: a user-selected or conservatively auto-detected target column is available.
2. Unlabeled Mode: no target column is selected, so the system generates and validates synthetic data without supervised ML metrics.

Synthetic data generation, validation, privacy checks, diversity checks, distribution comparison, and correlation comparison work in both modes. Labeled mode uses supervised prediction benchmarking; unlabeled mode uses a KNN similarity benchmark plus statistical checks.

## Quick Start

```powershell
.\.venv\Scripts\activate
python run.py --prepare-sample
```

Run the API:

```powershell
python run.py --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Run the dashboard in a second terminal:

```powershell
.\.venv\Scripts\activate
streamlit run frontend/dashboard.py
```

Optional Gemini explanations: copy `.env.example` to `.env` and set a valid `GEMINI_API_KEY`. Gemini explains measured results in plain language; core statistical and ML metrics never come from the LLM. Explanations are off by default. Enabling them with a configured key sends dataset summaries and measured results to Google.

## Sample Data

`python run.py --prepare-sample` creates:

- `datasets/sample/labeled_customer.csv`
- `datasets/sample/unlabeled_customer.csv`
- `datasets/sample/customer_churn.csv` for backward compatibility

In the dashboard, choose:

- `Target Column = target` for labeled mode
- `Target Column = None` for unlabeled mode
- `Target Column = Auto Detect` to conservatively detect names such as `target`, `label`, `class`, `churn`, or `outcome`

The system no longer assumes the final CSV column is the target.

## Labeled Mode

Use this when the CSV has a real target column.

The pipeline:

1. Resolves the target and creates disjoint original train/validation/final-test splits.
2. Profiles training rows and plans the generator/sample count using training information only.
3. Fits generator metadata/preprocessing on training rows and generates synthetic rows.
4. Validates generated data against real training data.
5. Compares models trained on original versus original-plus-synthetic training rows on validation rows.
6. Evaluates the configured generator/sample-count search budget and selects by validation utility and quality gates.
7. Evaluates only the selected candidate on the untouched final-test rows; no further search follows.
8. Reports improvement only when quality gates pass and both validation and final-test gains clear the configured thresholds.

Classification metrics include accuracy, precision, recall, F1, and ROC-AUC when available. Regression metrics include MAE, RMSE, and R2. By default, a labeled candidate needs at least `2%` relative and `+0.01` absolute primary-metric improvement before the pipeline calls it an improvement. Both final models use the same held-out original test rows. Per-iteration benchmarks are validation results; `final_benchmark` is the final-test comparison. Default fractions are 25% final test, then 25% of the remaining rows for validation (56.25% train / 18.75% validation / 25% test). Rare classes that cannot support this split produce a clear error.

## Unlabeled Mode

Use this when the CSV does not have a target column, or when you explicitly select `None`.

The pipeline:

1. Analyzes dataset quality without assigning a target.
2. Plans a general synthetic-data generation strategy.
3. Generates synthetic rows for the full table.
4. Validates fidelity, distribution similarity, correlation similarity, diversity, and basic privacy risk.
5. Runs a KNN similarity benchmark: KNN tries to distinguish original from generated rows. An AUC near `0.50` means it is guessing, which is desirable.
6. Computes overall unlabeled utility from the KNN result and deterministic statistical metrics.
7. Optimizes using synthetic-data quality and statistical utility.

No target-prediction F1, accuracy, precision, recall, MAE, RMSE, or R2 values are shown for unlabeled data. The displayed KNN AUC is a dataset-similarity comparison, not a prediction-quality claim.

## Adaptive Synthetic Sample Count

The platform does not blindly generate a fixed percentage for every dataset. `SamplePlanner` calculates an initial row count from dataset characteristics, then the optimizer can adjust the count after validation and utility feedback.

Prototype defaults are configurable in `app/config.py` or `.env`. The default five-iteration search has enough room to compare sample counts and all supported generator families; it takes longer than the previous three-iteration prototype run.

- `MINORITY_TARGET_RATIO=0.20`
- `MODERATE_IMBALANCE_TARGET_RATIO=0.40`
- `SMALL_DATASET_THRESHOLD=1000`
- `SMALL_DATASET_AUGMENTATION_RATIO=0.50`
- `LOW_DIVERSITY_AUGMENTATION_RATIO=0.30`
- `GENERAL_AUGMENTATION_RATIO=0.20`
- `MAX_GENERATION_RATIO_PER_ITERATION=0.50`
- `MAX_TOTAL_SYNTHETIC_RATIO=1.50`
- `SAMPLE_COUNT_INCREASE_FACTOR=1.50`
- `SAMPLE_COUNT_DECREASE_FACTOR=0.75`

Labeled severe imbalance formula:

```text
desired_minority_count = max(current_minority_count, ceil(original_training_rows * MINORITY_TARGET_RATIO))
requested_samples = desired_minority_count - current_minority_count
requested_samples = min(requested_samples, original_training_rows * MAX_GENERATION_RATIO_PER_ITERATION)
```

Moderate imbalance uses `MODERATE_IMBALANCE_TARGET_RATIO` instead of full class balancing. The default target is intentionally not 50/50.

Unlabeled formula:

- small dataset: `rows * SMALL_DATASET_AUGMENTATION_RATIO`
- low diversity: `rows * LOW_DIVERSITY_AUGMENTATION_RATIO`
- other meaningful data-quality issues: `rows * GENERAL_AUGMENTATION_RATIO`
- healthy large datasets may skip generation

Every iteration records requested rows, generated rows, cumulative synthetic rows, generator, validation results, and benchmark or unsupervised utility results.

## Architecture

- `DatasetAnalyzer`: deterministic column/type/quality analysis and optional conservative target detection
- `GenerationPlanner`: rule-based generator choice for labeled or unlabeled datasets
- `SamplePlanner`: adaptive initial synthetic row-count calculation
- `GeneratorAgent`: SDV CTGAN, TVAE, GaussianCopula, and an explicit bootstrap baseline; failed models never substitute another generator
- `ValidationAgent`: fidelity, distribution, correlation, diversity, and basic privacy indicators
- `BenchmarkAgent`: supervised original vs augmented downstream ML utility for labeled mode
- `evaluate_unsupervised_utility`: KNN similarity benchmark and statistical utility for unlabeled mode
- `OptimizationAgent`: separate labeled and unlabeled optimization decisions

Privacy, fairness, and provenance are prototype-level:

- Privacy: duplicate and nearest-neighbor similarity indicators, not differential privacy
- Fairness: supervised group comparison only when a target and sensitive column exist
- Provenance: metadata fingerprint, not an invisible cryptographic watermark

## API

- `GET /`
- `GET /health`
- `POST /analyze`
- `POST /generate`
- `POST /run`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/synthetic-data`

For `/run`, `target_column` is optional:

- omit it or send `None` for unlabeled mode
- send a real column name for labeled mode
- send `__auto__` or enable `auto_detect_target=true` for conservative auto-detection

If a specified target column does not exist, the API returns HTTP 400.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover labeled mode, unlabeled mode, alternate target names, auto-detection, invalid target handling, numerical-only unlabeled data, mixed unlabeled data, and generator behavior without a target column.

## Docker

```powershell
docker compose up --build
```

API: `http://localhost:8000`

Dashboard: `http://localhost:8501`

## Milestones 3–9: research extensions

The completed follow-on milestones add optional isolated TabSyn, Diffusion-TS and `timegan_torch` workers; explicit temporal preprocessing; durable SQLite experiment records; validation-only adaptive candidate selection; research-model/experiment API visibility; and a deterministic local text-generation/consistency baseline. These are reproducible software baselines with tiny integration checks, not claims of paper reproduction, production privacy, or full-size model quality. See [MILESTONES.md](MILESTONES.md) for actual timings and boundaries.

### Dataset-aware generator selection

The tabular planner ranks only installed and compatible generators from the training-data profile: row count, numerical/categorical balance, class imbalance, cardinality, skew, diversity, conditional-generation needs, device and model cost. It trains the top-ranked model first. If validation quality, downstream utility and hard privacy/validity gates pass, the search stops immediately; otherwise it advances to the next model in that dataset-specific ranking. Final-test rows remain untouched until selection is frozen.

### Time-series sample

Download `time_series_sensor.csv` from the dashboard or `GET /samples/time_series`. Its schema is stored beside it in `datasets/sample/time_series_sensor.schema.json`. Verify timestamp/entity handling, chronological splits, training-only scaling and window shapes with:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_timeseries_sample.py
```

This sample uses `timestamp` as the hourly timestamp, `machine_id` as the entity, three numerical features, and `machine_type` as context. Open **Time Series** in the dashboard to load it, validate window shapes, select Diffusion-TS or `timegan_torch`, follow durable job progress, inspect temporal/privacy metrics, and download generated sequences. Both modalities dispatch through `POST /pipeline-runs`; temporal jobs use the SQLite experiment store behind the same polling route.

## Milestone 2: generator contracts and baseline integrity

Adapters now support `fit(data, GeneratorConfig)`, `sample(count, SamplingConfig)`,
`get_metadata()`, `validate_input()`, and `get_capabilities()`. Existing
`generate(df, num_samples, ...)` calls still work. This is a tabular contract;
future temporal adapters will use a separate sequence representation.

The API accepts optional `device` (`cpu` by default, `cuda`, or `auto`),
`batch_size`, `epochs`, and `random_state` on planning/generation/run routes.
Training parameters must be supported by the selected model. CTGAN batch sizes
must be multiples of 10. Explicit CUDA requests fail if CUDA is unavailable.
`test_size` and `validation_size` are configurable on run routes. Sample counts
are adaptive unless `PipelineConfig.synthetic_ratio` or `DEFAULT_SYNTHETIC_RATIO` explicitly overrides them.

Per-run configuration/environment/failure records are stored as
`generated/{run_id}_manifest.json`; labeled split row positions are stored as
`{run_id}_splits.json`. `GET /runs/{run_id}/manifest` retrieves an execution
record, including a failed run's record when its ID is known. This is a local
tracking foundation, not a durable job queue or full experiment database.

The React dashboard runs with `cd frontend` then `npm.cmd ci` and `npm.cmd run dev`.
The API must run separately on port 8000. See [SETUP.md](SETUP.md) for details.

Current architecture, model feasibility, experimental protocol and verified
milestone status are in [ARCHITECTURE.md](ARCHITECTURE.md),
[MODEL_COMPARISON.md](MODEL_COMPARISON.md), [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md),
and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

The complete final-year/research documentation is available in
[PROJECT_REPORT.md](PROJECT_REPORT.md). It consolidates the architecture,
algorithms, data and model workflows, agent design, testing evidence,
performance, security, deployment strategy, results, and roadmap.
