"""CSV loading, validation, and sample-dataset helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import get_settings
from app.utils.exceptions import InvalidDatasetError


CATEGORICAL_NAME_HINTS = {
    "city",
    "type",
    "status",
    "gender",
    "employment",
    "employment_type",
    "category",
    "label",
    "target",
    "class",
    "churn",
}


def load_csv(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise InvalidDatasetError(f"Could not read CSV: {exc}") from exc
    return validate_dataframe(df)


def load_csv_bytes(content: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise InvalidDatasetError("Only .csv files are supported.")
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise InvalidDatasetError(
            f"File exceeds the {settings.max_upload_mb} MB upload limit."
        )
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise InvalidDatasetError(f"Could not parse CSV: {exc}") from exc
    return validate_dataframe(df)


def validate_dataframe(df: pd.DataFrame, min_rows: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        raise InvalidDatasetError("The dataset is empty.")
    if df.shape[1] < 2:
        raise InvalidDatasetError("The dataset must have at least two columns.")
    threshold = get_settings().min_rows if min_rows is None else min_rows
    if len(df) < threshold:
        raise InvalidDatasetError(
            f"The dataset has {len(df)} rows; at least {threshold} are required."
        )
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise InvalidDatasetError("Duplicate column names are not supported.")
    return df


def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        series = df[col]
        name = col.lower()
        nunique = series.nunique(dropna=True)
        if name in CATEGORICAL_NAME_HINTS or nunique <= 12:
            if not pd.api.types.is_numeric_dtype(series) or nunique <= 12:
                if nunique <= max(12, int(0.05 * len(df))) and not (
                    pd.api.types.is_float_dtype(series) and nunique > 20
                ):
                    if nunique <= 12 or not pd.api.types.is_numeric_dtype(series):
                        categorical.append(col)
                        continue
        if pd.api.types.is_numeric_dtype(series) and nunique > 12:
            numerical.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            categorical.append(col)
        else:
            categorical.append(col)
    if not numerical and not categorical:
        categorical = list(df.columns)
    return numerical, categorical


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def create_sample_churn_dataset(
    path: str | Path,
    n_rows: int = 500,
    seed: int = 42,
) -> Path:
    """Create an imbalanced customer-churn style CSV for demos and tests."""
    rng = np.random.default_rng(seed)
    employment = rng.choice(
        ["salaried", "self-employed", "unemployed"],
        size=n_rows,
        p=[0.62, 0.28, 0.10],
    )
    city = rng.choice(["Mumbai", "Pune", "Delhi", "Bengaluru"], size=n_rows)
    age = rng.integers(22, 70, size=n_rows)
    income = rng.normal(72000, 22000, size=n_rows).clip(18000, 180000).round(0)
    credit_score = rng.normal(680, 70, size=n_rows).clip(300, 850).round(0)
    loan_amount = rng.normal(18000, 9000, size=n_rows).clip(1000, 80000).round(0)
    existing_loans = rng.integers(0, 6, size=n_rows)
    tenure_months = rng.integers(1, 72, size=n_rows)

    # Rank-based target assignment keeps the target imbalanced while making the
    # minority class learnable enough for the augmentation demo.
    risk_score = (
        1.9 * (employment == "unemployed")
        + 0.9 * (employment == "self-employed")
        + 0.000055 * loan_amount
        - 0.012 * credit_score
        + 0.42 * existing_loans
        - 0.000018 * income
        - 0.018 * tenure_months
        + rng.normal(0, 0.35, size=n_rows)
    )
    desired_pos = max(1, int(0.10 * n_rows))
    target = np.zeros(n_rows, dtype=int)
    positive_idx = np.argsort(risk_score)[-desired_pos:]
    target[positive_idx] = 1

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "employment_type": employment,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "existing_loans": existing_loans,
            "city": city,
            "tenure_months": tenure_months,
            "target": target,
        }
    )
    return save_csv(df, path)


def create_unlabeled_customer_dataset(
    path: str | Path,
    n_rows: int = 500,
    seed: int = 123,
) -> Path:
    """Create an unlabeled customer-style CSV with no target column."""
    rng = np.random.default_rng(seed)
    city = rng.choice(["Mumbai", "Pune", "Delhi", "Bengaluru"], size=n_rows)
    age = rng.integers(22, 70, size=n_rows)
    income = rng.normal(76000, 24000, size=n_rows).clip(18000, 200000).round(0)
    credit_score = rng.normal(685, 75, size=n_rows).clip(300, 850).round(0)
    spending = (
        0.18 * income
        + rng.normal(8000, 5000, size=n_rows)
        + (city == "Mumbai") * 3500
        - (credit_score < 600) * 2500
    ).clip(1000, 90000).round(0)
    visits_per_month = rng.poisson(4, size=n_rows).clip(0, 20)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "city": city,
            "credit_score": credit_score,
            "spending": spending,
            "visits_per_month": visits_per_month,
        }
    )
    return save_csv(df, path)
