# Synthetic-AI - Autonomous Multi-Agent Synthetic Data Platform

College prototype for adaptive synthetic data generation, validation, and continuous dataset improvement on tabular CSV data.

The platform now supports two modes:

1. Labeled Mode: a user-selected or conservatively auto-detected target column is available.
2. Unlabeled Mode: no target column is selected, so the system generates and validates synthetic data without supervised ML metrics.

Synthetic data generation, validation, privacy checks, diversity checks, distribution comparison, and correlation comparison work in both modes. Supervised benchmarking runs only in labeled mode.

## Quick Start

```powershell
cd C:\Users\Ketan\Desktop\edai7_v1.0
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
cd C:\Users\Ketan\Desktop\edai7_v1.0
.\.venv\Scripts\activate
streamlit run frontend/dashboard.py
```

Optional Gemini explanations: copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Core statistical and ML metrics never come from the LLM.

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

1. Analyzes dataset quality and target distribution.
2. Plans a generator strategy.
3. Generates synthetic rows.
4. Validates synthetic data.
5. Trains a baseline model on original training data.
6. Trains an augmented model on original plus synthetic training data.
7. Evaluates both on the same held-out original test set.
8. Optimizes using supervised utility, such as F1 or R2.

Classification metrics include accuracy, precision, recall, F1, and ROC-AUC when available. Regression metrics include MAE, RMSE, and R2.

## Unlabeled Mode

Use this when the CSV does not have a target column, or when you explicitly select `None`.

The pipeline:

1. Analyzes dataset quality without assigning a target.
2. Plans a general synthetic-data generation strategy.
3. Generates synthetic rows for the full table.
4. Validates fidelity, distribution similarity, correlation similarity, diversity, and basic privacy risk.
5. Computes "Unsupervised / Statistical Utility" using deterministic statistical metrics.
6. Optimizes using synthetic-data quality and statistical utility.

No F1, accuracy, precision, recall, ROC-AUC, MAE, RMSE, or R2 values are shown for unlabeled data.

## Adaptive Synthetic Sample Count

The platform does not blindly generate a fixed percentage for every dataset. `SamplePlanner` calculates an initial row count from dataset characteristics, then the optimizer can adjust the count after validation and utility feedback.

Prototype defaults are configurable in `app/config.py` or `.env`:

- `MINORITY_TARGET_RATIO=0.50`
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
desired_minority_count = ceil(majority_count * MINORITY_TARGET_RATIO)
requested_samples = desired_minority_count - current_minority_count
requested_samples = min(requested_samples, original_rows * MAX_GENERATION_RATIO_PER_ITERATION)
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
- `GeneratorAgent`: SDV CTGAN, TVAE, GaussianCopula, with bootstrap fallback
- `ValidationAgent`: fidelity, distribution, correlation, diversity, and basic privacy indicators
- `BenchmarkAgent`: supervised original vs augmented downstream ML utility for labeled mode only
- `evaluate_unsupervised_utility`: statistical utility for unlabeled mode
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
