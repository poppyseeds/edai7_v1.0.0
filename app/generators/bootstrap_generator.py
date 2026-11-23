"""Empirical bootstrap fallback generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.generators.base import BaseSyntheticGenerator, GeneratorConfig, SamplingConfig
from app.generators.runtime import resolve_device
from app.utils.data_utils import infer_column_types
from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")


class BootstrapGenerator(BaseSyntheticGenerator):
    name = "bootstrap"

    def fit(self, data: pd.DataFrame, config: GeneratorConfig | None = None):
        self._data = None
        self.validate_input(data)
        cfg = config or GeneratorConfig()
        if cfg.advanced is not None:
            raise GeneratorError("Bootstrap does not accept advanced diffusion configuration.")
        resolve_device(cfg.device, supports_gpu=False)
        if cfg.epochs is not None or cfg.batch_size is not None:
            raise GeneratorError("Bootstrap does not support epochs or batch_size.")
        self._config = cfg.model_copy(update={"device": "cpu"})
        self._rng = np.random.default_rng(cfg.random_state)
        self._data = data.copy()
        return self

    def sample(self, num_samples: int, config: SamplingConfig | None = None) -> pd.DataFrame:
        cfg = config or SamplingConfig()
        self._validate_sampling(num_samples, cfg)
        if getattr(self, "_data", None) is None:
            raise GeneratorError("Bootstrap must be fitted before sampling.")
        return self._sample_from_frame(self._data, num_samples, self._rng,
                                       condition_column=cfg.condition_column, condition_value=cfg.condition_value)

    def get_metadata(self) -> dict:
        fitted = getattr(self, "_data", None) is not None
        return {"generator": self.name, "modality": "tabular", "fitted": fitted,
                "config": self._config.model_dump() if fitted else None,
                "columns": list(self._data.columns) if fitted else []}

    def _sample_from_frame(
        self,
        df: pd.DataFrame,
        num_samples: int,
        random_state: int = 42,
        epochs: int | None = None,
        condition_column: str | None = None,
        condition_value: object | None = None,
    ) -> pd.DataFrame:

        if df.empty:
            raise GeneratorError(
                "Cannot bootstrap an empty dataframe."
            )

        if num_samples <= 0:
            raise GeneratorError(
                "num_samples must be greater than zero."
            )

        rng = np.random.default_rng(random_state)

        source = df.copy()

        # If conditional generation is requested, filter the source
        # rather than changing the target after generation.
        if condition_column is not None:
            if condition_column not in source.columns:
                raise GeneratorError(
                    f"Condition column '{condition_column}' not found."
                )

            mask = (
                source[condition_column].astype(str)
                == str(condition_value)
            )

            conditioned = source.loc[mask]

            if not conditioned.empty:
                source = conditioned.reset_index(drop=True)
            else:
                raise GeneratorError(
                    f"No rows available for condition "
                    f"{condition_column}={condition_value!r}."
                )

        indices = rng.integers(
            0,
            len(source),
            size=num_samples,
        )

        out = source.iloc[indices].reset_index(drop=True).copy()

        numerical, _ = infer_column_types(df)

        for column in numerical:
            if column not in out.columns:
                continue

            original = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            values = pd.to_numeric(
                out[column],
                errors="coerce",
            )

            std = float(original.std(ddof=0) or 0.0)

            if std > 0:
                noise = rng.normal(
                    0.0,
                    0.01 * std,
                    size=num_samples,
                )
                values = values + noise

            # Prevent obvious out-of-range values.
            minimum = original.min()
            maximum = original.max()

            if pd.notna(minimum) and pd.notna(maximum):
                values = values.clip(
                    lower=minimum,
                    upper=maximum,
                )

            # Preserve integer columns.
            if pd.api.types.is_integer_dtype(df[column]):
                values = np.rint(values)

            out[column] = values

        # Restore obvious boolean columns.
        for column in df.columns:
            if pd.api.types.is_bool_dtype(df[column]):
                out[column] = out[column].astype(bool)

        # Restore integer columns where possible.
        for column in df.columns:
            if pd.api.types.is_integer_dtype(df[column]):
                try:
                    out[column] = (
                        pd.to_numeric(out[column], errors="coerce")
                        .round()
                        .astype(df[column].dtype)
                    )
                except (TypeError, ValueError):
                    pass

        out = out[df.columns]

        logger.info("Bootstrap baseline generated %d rows", len(out))

        return out.reset_index(drop=True)
