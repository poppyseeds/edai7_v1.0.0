"""Empirical bootstrap with light numerical noise. Used if SDV is unavailable."""

from __future__ import annotations

import pandas as pd
import numpy as np

from app.generators.base import BaseSyntheticGenerator
from app.utils.data_utils import infer_column_types
from app.utils.logging_config import get_logger

logger = get_logger("Generator")


class BootstrapGenerator(BaseSyntheticGenerator):
    name = "bootstrap"

    def generate(
        self,
        df: pd.DataFrame,
        num_samples: int,
        random_state: int = 42,
        epochs: int | None = None,
    ) -> pd.DataFrame:
        logger.info("Training bootstrap fallback generator")
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, len(df), size=num_samples)
        out = df.iloc[idx].reset_index(drop=True).copy()
        numerical, _ = infer_column_types(df)
        for col in numerical:
            series = pd.to_numeric(out[col], errors="coerce")
            std = float(pd.to_numeric(df[col], errors="coerce").std(ddof=0) or 0.0)
            noise = rng.normal(0.0, 0.05 * std if std else 0.0, size=num_samples)
            out[col] = series + noise
        logger.info("Generated %s rows via bootstrap", num_samples)
        return out
