"""Shared SDV fitting helpers."""

from __future__ import annotations

import pandas as pd

from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")


def build_metadata(df: pd.DataFrame):
    try:
        from sdv.metadata import SingleTableMetadata
    except Exception as exc:  # pragma: no cover
        raise GeneratorError(f"SDV is not available: {exc}") from exc
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    return metadata


def sample_sdv(synthesizer, num_samples: int) -> pd.DataFrame:
    try:
        return synthesizer.sample(num_rows=num_samples)
    except Exception as exc:
        raise GeneratorError(f"SDV sampling failed: {exc}") from exc
