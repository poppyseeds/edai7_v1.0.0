import pandas as pd

from app.generators.sdv_common import build_metadata


def _columns(metadata) -> dict:
    return metadata.to_dict()["columns"]


def test_metadata_handles_numerical_and_categorical_columns() -> None:
    df = pd.DataFrame(
        {
            "age": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            "city": ["Pune", "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi", "Pune"],
            "target": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        }
    )
    metadata = build_metadata(df)
    columns = _columns(metadata)
    assert columns["age"]["sdtype"] == "numerical"
    assert columns["city"]["sdtype"] == "categorical"
    assert columns["target"]["sdtype"] == "categorical"


def test_metadata_handles_missing_boolean_and_datetime_columns() -> None:
    df = pd.DataFrame(
        {
            "amount": [10.5, None, 13.0, 14.5, 18.0, None, 20.0],
            "is_active": pd.Series([True, False, True, None, False, True, False], dtype="boolean"),
            "event_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", None, "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]
            ),
            "segment": ["A", "B", None, "A", "C", "B", "A"],
        }
    )
    metadata = build_metadata(df)
    columns = _columns(metadata)
    assert columns["amount"]["sdtype"] == "numerical"
    assert columns["is_active"]["sdtype"] == "boolean"
    assert columns["event_date"]["sdtype"] == "datetime"
    assert columns["segment"]["sdtype"] == "categorical"


def test_metadata_handles_constant_and_id_like_columns() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(20)],
            "constant_flag": ["same"] * 20,
            "score": [float(i) for i in range(20)],
        }
    )
    metadata = build_metadata(df)
    meta = metadata.to_dict()
    columns = meta["columns"]
    assert "primary_key" not in meta
    assert columns["customer_id"]["sdtype"] == "id"
    assert columns["constant_flag"]["sdtype"] == "categorical"
    assert columns["score"]["sdtype"] == "numerical"


def test_metadata_handles_mixed_object_values_as_categorical() -> None:
    df = pd.DataFrame(
        {
            "mixed": (["1", "two", 3, "four", None] * 4),
            "numeric_text": [str(i) for i in range(20, 40)],
        }
    )
    metadata = build_metadata(df)
    columns = _columns(metadata)
    assert columns["mixed"]["sdtype"] == "categorical"
    assert columns["numeric_text"]["sdtype"] == "categorical"
