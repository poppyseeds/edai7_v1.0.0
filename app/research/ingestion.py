"""Bounded local CSV/Parquet ingestion shared by research modalities."""
from io import BytesIO
import pandas as pd


def read_table(content: bytes, filename: str, max_bytes: int = 20 * 1024 * 1024) -> pd.DataFrame:
    if len(content) > max_bytes:
        raise ValueError("Dataset exceeds the configured upload limit.")
    if filename.lower().endswith(".csv"):
        frame = pd.read_csv(BytesIO(content))
    elif filename.lower().endswith(".parquet"):
        try:
            frame = pd.read_parquet(BytesIO(content))
        except ImportError as exc:
            raise ValueError("Parquet ingestion needs optional pyarrow or fastparquet.") from exc
    else:
        raise ValueError("Research datasets must use CSV or Parquet.")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Dataset is empty or contains duplicate columns.")
    return frame
