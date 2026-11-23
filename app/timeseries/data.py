from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeSeriesSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp: str
    features: list[str] = Field(min_length=1)
    entity: str | None = None
    context: list[str] = Field(default_factory=list)
    sequence_length: int = Field(default=24, ge=8, le=4096)
    stride: int = Field(default=1, ge=1)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    missing_policy: Literal["reject", "forward_fill", "training_median"] = "reject"
    interval_tolerance: float = Field(default=0.01, ge=0, lt=1)
    generated_start: str = "2000-01-01T00:00:00Z"

    @model_validator(mode="after")
    def validate_roles(self):
        roles = [self.timestamp, *self.features, *self.context] + ([self.entity] if self.entity else [])
        if len(roles) != len(set(roles)):
            raise ValueError("Schema roles must refer to distinct columns.")
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("Temporal validation/test fractions must leave training observations.")
        pd.Timestamp(self.generated_start)
        return self


@dataclass
class PreparedSequences:
    windows: dict[str, np.ndarray]
    provenance: dict[str, list[dict]]
    metadata: dict


def profile_sequences(frame: pd.DataFrame, schema: TimeSeriesSchema) -> dict:
    missing = set([schema.timestamp, *schema.features, *schema.context] + ([schema.entity] if schema.entity else [])) - set(frame)
    if missing:
        raise ValueError(f"Missing temporal schema columns: {sorted(missing)}")
    data = frame.copy()
    data[schema.timestamp] = pd.to_datetime(data[schema.timestamp], errors="raise", utc=True)
    if data[schema.timestamp].isna().any() or (schema.entity and data[schema.entity].isna().any()):
        raise ValueError("Timestamps/entity identifiers must not be missing.")
    groups = data.groupby(schema.entity, sort=False) if schema.entity else [("single", data)]
    entities = []
    for entity, group in groups:
        stamps = group[schema.timestamp].sort_values()
        delta = np.diff(stamps.astype("int64"))
        if len(delta) == 0 or (delta <= 0).any():
            raise ValueError("Every entity needs at least two distinct timestamps; duplicate timestamps are unsupported.")
        interval = int(np.median(delta))
        irregular = np.abs(delta - interval) > max(1, interval * schema.interval_tolerance)
        entities.append({"entity": str(entity), "rows": len(group), "interval_ns": interval,
                         "irregular_intervals": int(irregular.sum())})
    return {"modality": "time_series", "rows": len(data), "features": schema.features,
            "entities": entities, "irregular": any(x["irregular_intervals"] for x in entities),
            "missing_by_feature": {c: int(data[c].isna().sum()) for c in schema.features},
            "context_policy": "retained as metadata; numerical generators do not synthesize or condition on context"}


def prepare_sequences(frame: pd.DataFrame, schema: TimeSeriesSchema) -> PreparedSequences:
    profile = profile_sequences(frame, schema)
    if profile["irregular"]:
        raise ValueError("Irregular sampling detected. Explicitly regularize data outside generation; no resampling was performed.")
    intervals = {item["interval_ns"] for item in profile["entities"]}
    if len(intervals) != 1:
        raise ValueError("Entities have different sampling intervals; run them as separate datasets.")
    data = frame.reset_index(drop=True).copy()
    if "__source_row__" in data:
        raise ValueError("Reserved temporal ingestion column __source_row__.")
    data["__source_row__"] = np.arange(len(data))
    data[schema.timestamp] = pd.to_datetime(data[schema.timestamp], utc=True)
    partitions = {key: [] for key in ("train", "validation", "test")}
    groups = data.groupby(schema.entity, sort=False) if schema.entity else [("single", data)]
    for entity, group in groups:
        group = group.sort_values(schema.timestamp, kind="stable")
        train_end = int(len(group) * (1-schema.test_fraction-schema.validation_fraction))
        valid_end = int(len(group) * (1-schema.test_fraction))
        for key, partition in zip(partitions, (group.iloc[:train_end], group.iloc[train_end:valid_end], group.iloc[valid_end:])):
            if len(partition) < schema.sequence_length:
                raise ValueError(f"Entity {entity} has too few {key} observations for sequence length {schema.sequence_length}.")
            values = partition[schema.features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError("Infinite temporal measurements require explicit cleaning.")
            partitions[key].append((str(entity), partition, values))
    raw_train = np.concatenate([x[2] for x in partitions["train"]])
    if np.isnan(raw_train).all(axis=0).any():
        raise ValueError("A training feature is entirely missing.")
    medians = np.nanmedian(raw_train, axis=0)
    cleaned = {}
    for key, groups in partitions.items():
        cleaned[key] = []
        for entity, partition, values in groups:
            if schema.missing_policy == "forward_fill":
                values = pd.DataFrame(values).ffill().to_numpy()
            elif schema.missing_policy == "training_median":
                values = np.where(np.isnan(values), medians, values)
            if np.isnan(values).any():
                raise ValueError(f"Missing measurements remain in {key}; choose an explicit supported policy.")
            cleaned[key].append((entity, partition, values))
    training = np.concatenate([x[2] for x in cleaned["train"]])
    low, high = training.min(0), training.max(0)
    span = np.where(high > low, high-low, 1.0)
    windows, provenance = {}, {}
    for key, groups in cleaned.items():
        batches, records = [], []
        for entity, partition, values in groups:
            normalized = (values-low)/span*2-1
            for start in range(0, len(values)-schema.sequence_length+1, schema.stride):
                chunk = partition.iloc[start:start+schema.sequence_length]
                batches.append(normalized[start:start+schema.sequence_length])
                records.append({"entity": entity, "rows": chunk.__source_row__.tolist(),
                    "timestamps": chunk[schema.timestamp].astype(str).tolist(),
                    "context": {c: chunk[c].astype(str).tolist() for c in schema.context}})
        windows[key] = np.asarray(batches, dtype="float32")
        provenance[key] = records
    return PreparedSequences(windows, provenance, {"schema": schema.model_dump(), "profile": profile,
        "scaler": {"minimum": low.tolist(), "span": span.tolist(), "training_median": medians.tolist()},
        "interval_ns": intervals.pop(), "tensor_shape": "sequence x timestep x feature"})


def inverse_sequences(values: np.ndarray, metadata: dict) -> np.ndarray:
    return (values+1)/2 * np.array(metadata["scaler"]["span"]) + np.array(metadata["scaler"]["minimum"])


def sequences_to_frame(values: np.ndarray, metadata: dict) -> pd.DataFrame:
    schema = TimeSeriesSchema(**metadata["schema"])
    physical = inverse_sequences(values, metadata)
    rows = []
    timestamps = pd.date_range(schema.generated_start, periods=values.shape[1], freq=pd.Timedelta(metadata["interval_ns"], unit="ns"))
    for index, sequence in enumerate(physical):
        frame = pd.DataFrame(sequence, columns=schema.features)
        frame.insert(0, schema.timestamp, timestamps)
        frame.insert(0, schema.entity or "synthetic_sequence_id", f"synthetic_{index}")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
