"""Shared SDV fitting and sampling helpers."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")

ID_NAME_HINTS = ("id", "_id", "uuid", "guid", "index", "key")
BOOLEAN_STRINGS = {"true", "false", "yes", "no"}


def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a dataframe before passing it to SDV."""

    if not isinstance(df, pd.DataFrame):
        raise GeneratorError("Generator input must be a pandas DataFrame.")

    if df.empty:
        raise GeneratorError("Cannot train a generator on an empty dataframe.")

    if len(df.columns) == 0:
        raise GeneratorError("Dataset contains no columns.")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise GeneratorError(f"Dataset contains duplicate column names: {duplicated}")

    out = df.copy()

    # SDV works better with ordinary RangeIndex.
    out = out.reset_index(drop=True)

    # Object columns are explicitly converted to strings only when
    # they actually contain mixed Python types.
    for column in out.columns:
        series = out[column]

        if series.dtype == "object":
            non_null_types = {
                type(value)
                for value in series.dropna().head(1000)
            }

            if len(non_null_types) > 1:
                logger.warning(
                    "Column '%s' contains mixed Python types. Converting to string.",
                    column,
                )
                out[column] = series.astype("string")

    return out


def build_metadata(df: pd.DataFrame):
    """Build SDV metadata using the installed SDV version."""

    try:
        from sdv.metadata import SingleTableMetadata
    except Exception as exc:
        raise GeneratorError(f"SDV is not available: {exc}") from exc

    clean_df = validate_input_dataframe(df)

    metadata = SingleTableMetadata()

    try:
        metadata.detect_from_dataframe(data=clean_df)
    except Exception as exc:
        raise GeneratorError(
            f"SDV metadata detection failed: {exc}"
        ) from exc

    _remove_auto_primary_key(metadata)
    detected = metadata.to_dict().get("columns", {})
    for column in clean_df.columns:
        _update_column(
            metadata,
            column,
            _classify_column(clean_df[column], column, detected.get(column, {})),
        )

    # Validate metadata if supported by the installed SDV version.
    validate_method = getattr(metadata, "validate", None)

    if callable(validate_method):
        try:
            validate_method()
        except Exception as exc:
            raise GeneratorError(
                f"SDV metadata validation failed: {exc}"
            ) from exc

    validate_data = getattr(metadata, "validate_data", None)
    if callable(validate_data):
        try:
            validate_data(clean_df)
        except Exception as exc:
            raise GeneratorError(
                f"SDV metadata data validation failed: {exc}"
            ) from exc

    logger.info(
        "Built SDV metadata for %d columns and %d rows",
        len(clean_df.columns),
        len(clean_df),
    )

    return metadata


def _remove_auto_primary_key(metadata) -> None:
    meta = metadata.to_dict()
    if "primary_key" not in meta:
        return
    try:
        metadata.remove_primary_key()
    except Exception as exc:
        raise GeneratorError(f"Could not remove auto-detected primary key: {exc}") from exc


def _update_column(metadata, column: str, decision: dict) -> None:
    try:
        metadata.update_column(column, **decision["metadata"])
    except Exception as exc:
        raise GeneratorError(
            f"Could not update SDV metadata for column '{column}': {exc}"
        ) from exc


def _classify_column(series: pd.Series, column: str, detected: dict) -> dict:
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))

    if unique_count <= 1:
        return {"metadata": {"sdtype": "categorical"}}
    if is_datetime64_any_dtype(series) or detected.get("sdtype") == "datetime":
        metadata = {"sdtype": "datetime"}
        if detected.get("datetime_format"):
            metadata["datetime_format"] = detected["datetime_format"]
        return {"metadata": metadata}
    if is_bool_dtype(series) or _is_boolean_like(non_null):
        return {"metadata": {"sdtype": "boolean"}}
    if _is_id_like(series, column):
        return {"metadata": {"sdtype": "id"}}
    if is_numeric_dtype(series):
        if _is_integer_like(non_null) and unique_count <= 12:
            return {"metadata": {"sdtype": "categorical"}}
        return {"metadata": {"sdtype": "numerical"}}
    if is_categorical_dtype(series):
        return {"metadata": {"sdtype": "categorical"}}
    return {"metadata": {"sdtype": "categorical"}}


def _is_boolean_like(series: pd.Series) -> bool:
    if series.empty:
        return False
    values = {str(value).strip().lower() for value in series.unique()}
    return values <= BOOLEAN_STRINGS


def _is_id_like(series: pd.Series, column: str) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    name = column.lower()
    name_suggests_id = name in ID_NAME_HINTS or any(name.endswith(hint) for hint in ID_NAME_HINTS)
    return bool(name_suggests_id and non_null.nunique(dropna=True) / len(non_null) >= 0.95)


def _is_integer_like(series: pd.Series) -> bool:
    if series.empty or not is_numeric_dtype(series):
        return False
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return bool((numeric % 1 == 0).all())


def set_random_state(synthesizer, random_state: int) -> None:
    """
    Set SDV random state when supported by the installed version.

    SDV versions differ slightly, so this intentionally checks
    the available API instead of assuming a constructor argument.
    """

    method = getattr(synthesizer, "set_random_state", None)
    method_name = "set_random_state"
    if not callable(method):
        # SDV 1.38 exposes the supported seed hook with this compatibility name.
        method = getattr(synthesizer, "_set_random_state", None)
        method_name = "_set_random_state"

    if not callable(method):
        logger.warning("Installed SDV synthesizer does not expose a random-state API.")
        return

    try:
        method(random_state)
        logger.info("Configured SDV random state=%s via %s", random_state, method_name)
    except Exception as exc:
        raise GeneratorError(
            f"Could not configure SDV random state via {method_name}: {exc}"
        ) from exc


def fit_sdv(synthesizer, df: pd.DataFrame, random_state: int) -> None:
    """Fit an SDV synthesizer safely."""

    clean_df = validate_input_dataframe(df)

    try:
        synthesizer.fit(clean_df)
    except Exception as exc:
        raise GeneratorError(
            f"SDV synthesizer training failed: {exc}"
        ) from exc

    # SDV 1.38 creates the underlying model during fit, so its compatible
    # _set_random_state hook can only be called after training and before sample.
    set_random_state(synthesizer, random_state)


def sample_sdv(
    synthesizer,
    num_samples: int,
    condition_column: str | None = None,
    condition_value: object | None = None,
) -> pd.DataFrame:
    """
    Sample from an SDV synthesizer.

    Conditional generation uses SDV's native conditional sampling API.
    We never generate arbitrary rows and overwrite the target afterward.
    """

    if num_samples <= 0:
        raise GeneratorError("num_samples must be greater than zero.")

    try:
        if condition_column is None:
            synthetic = synthesizer.sample(num_rows=num_samples)

        else:
            if condition_value is None:
                raise GeneratorError(
                    "condition_value is required when condition_column is provided."
                )

            try:
                from sdv.sampling import Condition
            except Exception as exc:
                raise GeneratorError(
                    "This SDV installation does not expose "
                    "the Condition API required for conditional generation."
                ) from exc

            condition = Condition(
                num_rows=num_samples,
                column_values={
                    condition_column: condition_value,
                },
            )

            conditional_sampler = getattr(
                synthesizer,
                "sample_from_conditions",
                None,
            )

            if not callable(conditional_sampler):
                raise GeneratorError(
                    "Installed SDV synthesizer does not support "
                    "sample_from_conditions()."
                )

            synthetic = conditional_sampler(
                conditions=[condition]
            )

    except GeneratorError:
        raise
    except Exception as exc:
        raise GeneratorError(
            f"SDV sampling failed: {exc}"
        ) from exc

    if not isinstance(synthetic, pd.DataFrame):
        synthetic = pd.DataFrame(synthetic)

    if len(synthetic) != num_samples:
        raise GeneratorError(
            f"SDV returned {len(synthetic)} rows; "
            f"expected {num_samples}."
        )

    return synthetic.reset_index(drop=True)


def validate_generated_output(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    expected_rows: int,
) -> pd.DataFrame:
    """Validate the basic structural properties of generated data."""

    if synthetic is None or synthetic.empty:
        raise GeneratorError("Generator produced an empty dataframe.")

    if len(synthetic) != expected_rows:
        raise GeneratorError(
            f"Generator returned {len(synthetic)} rows; "
            f"expected {expected_rows}."
        )

    missing_columns = [
        column
        for column in original.columns
        if column not in synthetic.columns
    ]

    if missing_columns:
        raise GeneratorError(
            f"Synthetic data is missing columns: {missing_columns}"
        )

    # Keep the exact original column ordering.
    synthetic = synthetic[original.columns].copy()

    return synthetic.reset_index(drop=True)
