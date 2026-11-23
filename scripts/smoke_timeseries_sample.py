"""Validate the bundled temporal sample through the real preprocessing path."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.timeseries.data import TimeSeriesSchema, prepare_sequences
from app.utils.data_utils import create_sample_timeseries_dataset


DATA = ROOT / "datasets" / "sample" / "time_series_sensor.csv"
SCHEMA = ROOT / "datasets" / "sample" / "time_series_sensor.schema.json"


def main():
    if not DATA.exists():
        create_sample_timeseries_dataset(DATA)
    frame = pd.read_csv(DATA)
    schema = TimeSeriesSchema(**json.loads(SCHEMA.read_text(encoding="utf-8")))
    prepared = prepare_sequences(frame, schema)
    print(json.dumps({
        "dataset": str(DATA),
        "rows": len(frame),
        "profile": prepared.metadata["profile"],
        "window_shapes": {name: list(values.shape) for name, values in prepared.windows.items()},
        "tensor_layout": prepared.metadata["tensor_shape"],
    }, indent=2))


if __name__ == "__main__":
    main()
