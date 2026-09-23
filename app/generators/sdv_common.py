"""Shared SDV fitting helpers."""

from __future__ import annotations

from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
import pandas as pd

from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")

ID_NAME_HINTS = ("id", "_id", "uuid", "guid", "index", "key")
BOOLEAN_STRINGS = {"true", "false", "yes", "no"}


def build_metadata(df: pd.DataFrame):
    try:
        from sdv.metadata import SingleTableMetadata
    except Exception as exc:  # pragma: no cover
        raise GeneratorError(f"SDV is not available: {exc}") from exc
    if df.empty:
        raise GeneratorError("Cannot build SDV metadata for an empty dataframe.")

    metadata = SingleTableMetadata()
    try:
        metadata.detect_from_dataframe(data=df)
    except Exception as exc:
        raise GeneratorError(f"SDV metadata detection failed: {exc}") from exc

    _remove_auto_primary_key(metadata)
    detected = metadata.to_dict().get("columns", {})
    for column in df.columns:
        decision = _classify_column(df[column], column, detected.get(column, {}))
        _update_column(metadata, column, decision)

    try:
        metadata.validate()
        metadata.validate_data(df)
    except Exception as exc:
        raise GeneratorError(f"SDV metadata validation failed: {exc}") from exc
    return metadata


def sample_sdv(synthesizer, num_samples: int) -> pd.DataFrame:
    try:
        return synthesizer.sample(num_rows=num_samples)
    except Exception as exc:
        raise GeneratorError(f"SDV sampling failed: {exc}") from exc


def _remove_auto_primary_key(metadata) -> None:
    meta = metadata.to_dict()
    if "primary_key" not in meta:
        return
    logger.info(
        "Removing auto-detected SDV primary key '%s'; generated data should preserve schema without enforcing uniqueness.",
        meta["primary_key"],
    )
    try:
        metadata.remove_primary_key()
    except Exception as exc:
        raise GeneratorError(f"Could not remove auto-detected primary key: {exc}") from exc


def _update_column(metadata, column: str, decision: dict) -> None:
    try:
        metadata.update_column(column, **decision["metadata"])
    except Exception as exc:
        raise GeneratorError(
            f"Could not update SDV metadata for column '{column}' with {decision['metadata']}: {exc}"
        ) from exc
    logger.info(
        "Metadata column '%s': sdtype=%s%s (%s)",
        column,
        decision["metadata"].get("sdtype"),
        f", datetime_format={decision['metadata'].get('datetime_format')}"
        if decision["metadata"].get("datetime_format")
        else "",
        decision["reason"],
    )


def _classify_column(series: pd.Series, column: str, detected: dict) -> dict:
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))
    missing_count = int(series.isna().sum())
    if missing_count:
        logger.info("Column '%s' contains %s missing values; SDV will model missingness.", column, missing_count)

    if unique_count <= 1:
        return {
            "metadata": {"sdtype": "categorical"},
            "reason": f"constant/problematic column with {unique_count} unique non-null value(s)",
        }

    if _is_datetime(series, detected):
        metadata = {"sdtype": "datetime"}
        if detected.get("datetime_format"):
            metadata["datetime_format"] = detected["datetime_format"]
        return {"metadata": metadata, "reason": "datetime dtype or SDV-detected datetime"}

    if _is_boolean(series):
        return {"metadata": {"sdtype": "boolean"}, "reason": "boolean dtype or boolean-like values"}

    if _is_id_like(series, column):
        return {"metadata": {"sdtype": "id"}, "reason": "ID/index-like high-uniqueness column"}

    if is_numeric_dtype(series):
        if _is_integer_like(non_null) and unique_count <= 12:
            return {"metadata": {"sdtype": "categorical"}, "reason": "low-cardinality numeric column"}
        return {"metadata": {"sdtype": "numerical"}, "reason": "numeric dtype"}

    if is_categorical_dtype(series):
        return {"metadata": {"sdtype": "categorical"}, "reason": "pandas categorical dtype"}

    return {"metadata": {"sdtype": "categorical"}, "reason": "categorical or mixed object values"}


def _is_datetime(series: pd.Series, detected: dict) -> bool:
    return is_datetime64_any_dtype(series) or detected.get("sdtype") == "datetime"


def _is_boolean(series: pd.Series) -> bool:
    if is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    values = {str(value).strip().lower() for value in non_null.unique()}
    return values <= BOOLEAN_STRINGS


def _is_id_like(series: pd.Series, column: str) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    name = column.lower()
    name_suggests_id = name in ID_NAME_HINTS or any(name.endswith(hint) for hint in ID_NAME_HINTS)
    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    if name_suggests_id and unique_ratio >= 0.95:
        return True
    return False


def _is_integer_like(series: pd.Series) -> bool:
    if series.empty or not is_numeric_dtype(series):
        return False
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return bool((numeric % 1 == 0).all())
