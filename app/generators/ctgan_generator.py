from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from app.generators.base import BaseSyntheticGenerator
from app.generators.sdv_common import build_metadata, sample_sdv
from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")

# CTGAN benefits from more training on small tables, while very large prototype
# datasets remain practical with fewer epochs. Explicit user configuration wins.
DEFAULT_EPOCHS = 300
SMALL_DATASET_EPOCHS = 500
LARGE_DATASET_EPOCHS = 200
SMALL_DATASET_ROWS = 500
LARGE_DATASET_ROWS = 10_000


class CTGANGenerator(BaseSyntheticGenerator):
    name = "ctgan"

    def generate(
        self,
        df: pd.DataFrame,
        num_samples: int,
        random_state: int = 42,
        epochs: int | None = None,
    ) -> pd.DataFrame:
        self._validate_input(df, num_samples, epochs, random_state)
        resolved_epochs = self._resolve_epochs(len(df), epochs)
        logger.info(
            "Training CTGAN: rows=%s, columns=%s, missing_values=%s, epochs=%s, random_state=%s",
            len(df),
            len(df.columns),
            int(df.isna().sum().sum()),
            resolved_epochs,
            random_state,
        )
        try:
            from sdv.single_table import CTGANSynthesizer
        except Exception as exc:
            raise GeneratorError(f"CTGAN unavailable: {exc}") from exc

        metadata = build_metadata(df)
        with self._seed_rngs(random_state):
            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=resolved_epochs,
                verbose=False,
            )
            self._set_random_state(synthesizer, random_state)
            try:
                synthesizer.fit(df)
            except Exception as exc:
                logger.exception("CTGAN training failed")
                raise GeneratorError(f"CTGAN training failed: {exc}") from exc

            # SDV exposes this public hook for reproducible sampling. Reapply it
            # after fitting because fitting may advance the synthesizer RNG state.
            self._set_random_state(synthesizer, random_state)
            synthetic = sample_sdv(synthesizer, num_samples)

        synthetic = self._validate_output(df, synthetic, num_samples)
        logger.info(
            "Generated CTGAN rows=%s, columns=%s, missing_values=%s",
            len(synthetic),
            len(synthetic.columns),
            int(synthetic.isna().sum().sum()),
        )
        return synthetic

    @staticmethod
    def _resolve_epochs(row_count: int, configured_epochs: int | None) -> int:
        if configured_epochs is not None:
            return configured_epochs
        if row_count < SMALL_DATASET_ROWS:
            return SMALL_DATASET_EPOCHS
        if row_count >= LARGE_DATASET_ROWS:
            return LARGE_DATASET_EPOCHS
        return DEFAULT_EPOCHS

    @staticmethod
    def _validate_input(
        df: pd.DataFrame,
        num_samples: int,
        epochs: int | None,
        random_state: int,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise GeneratorError("CTGAN input must be a pandas dataframe.")
        if df.empty:
            raise GeneratorError("CTGAN input dataframe must contain at least one row.")
        if not len(df.columns):
            raise GeneratorError("CTGAN input dataframe must contain at least one column.")
        if not df.columns.is_unique:
            raise GeneratorError("CTGAN input dataframe must have unique column names.")
        if not all(isinstance(column, str) for column in df.columns):
            raise GeneratorError("CTGAN input dataframe column names must be strings.")
        if df.isna().all().any():
            columns = list(df.columns[df.isna().all()])
            raise GeneratorError(f"CTGAN input has all-missing columns: {columns}")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise GeneratorError("CTGAN num_samples must be a positive integer.")
        if epochs is not None and (
            isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0
        ):
            raise GeneratorError("CTGAN epochs must be a positive integer when provided.")
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise GeneratorError("CTGAN random_state must be an integer.")

    @staticmethod
    def _set_random_state(synthesizer: object, random_state: int) -> None:
        set_random_state = getattr(synthesizer, "set_random_state", None)
        if not callable(set_random_state):
            logger.warning(
                "Installed SDV CTGAN has no public set_random_state method; "
                "using process RNG seeds for best-effort reproducibility."
            )
            return
        try:
            set_random_state(random_state)
        except Exception as exc:
            raise GeneratorError(f"Could not configure CTGAN random_state: {exc}") from exc

    @classmethod
    def _validate_output(
        cls, original: pd.DataFrame, synthetic: pd.DataFrame, num_samples: int
    ) -> pd.DataFrame:
        if not isinstance(synthetic, pd.DataFrame):
            raise GeneratorError("CTGAN sampling did not return a pandas dataframe.")
        if len(synthetic) != num_samples:
            raise GeneratorError(
                f"CTGAN generated {len(synthetic)} rows; expected {num_samples}."
            )
        if list(synthetic.columns) != list(original.columns):
            raise GeneratorError("CTGAN output columns do not match the input columns.")
        if synthetic.columns.has_duplicates:
            raise GeneratorError("CTGAN output has duplicate column names.")

        output = synthetic.copy()
        for column in original.columns:
            output[column] = cls._restore_dtype(output[column], original[column])
            if output[column].isna().all():
                raise GeneratorError(f"CTGAN output column '{column}' contains only missing values.")

        for column in original.columns:
            if output[column].dtype != original[column].dtype:
                raise GeneratorError(
                    f"CTGAN output dtype for '{column}' is {output[column].dtype}; "
                    f"expected {original[column].dtype}."
                )
        return output.reset_index(drop=True)

    @staticmethod
    def _restore_dtype(values: pd.Series, original: pd.Series) -> pd.Series:
        dtype = original.dtype
        try:
            if is_categorical_dtype(dtype):
                categories = original.cat.categories
                unknown = values[values.notna() & ~values.isin(categories)]
                if not unknown.empty:
                    raise GeneratorError(
                        f"CTGAN output for '{original.name}' contains values outside its categories."
                    )
                return values.astype(dtype)
            if is_bool_dtype(dtype):
                valid = values.dropna().isin([True, False])
                if not valid.all():
                    raise GeneratorError(f"CTGAN output for '{original.name}' is not boolean.")
                return values.astype(dtype)
            if is_datetime64_any_dtype(dtype):
                return pd.to_datetime(values, errors="raise").astype(dtype)
            if is_integer_dtype(dtype):
                numeric = pd.to_numeric(values, errors="raise")
                non_null = numeric.dropna()
                if not np.isclose(non_null, np.round(non_null)).all():
                    raise GeneratorError(
                        f"CTGAN output for '{original.name}' cannot be represented as {dtype}."
                    )
                return numeric.round().astype(dtype)
            if is_numeric_dtype(dtype) or is_string_dtype(dtype):
                return values.astype(dtype)
            return values.astype(dtype)
        except GeneratorError:
            raise
        except (TypeError, ValueError) as exc:
            raise GeneratorError(
                f"CTGAN output dtype does not match input column '{original.name}': {exc}"
            ) from exc

    @staticmethod
    @contextmanager
    def _seed_rngs(random_state: int) -> Iterator[None]:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch = None
        torch_state = None
        cuda_state = None
        try:
            np.random.seed(random_state)
            random.seed(random_state)
            try:
                import torch as torch_module

                torch = torch_module
                torch_state = torch.get_rng_state()
                torch.manual_seed(random_state)
                if torch.cuda.is_available():
                    cuda_state = torch.cuda.get_rng_state_all()
                    torch.cuda.manual_seed_all(random_state)
            except ImportError:
                pass
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if torch is not None and torch_state is not None:
                torch.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
