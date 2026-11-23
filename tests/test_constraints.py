import pandas as pd

from app.evaluation.constraints import discover_constraints, validate_constraints


def test_constraint_validation_accepts_valid_synthetic_rows() -> None:
    real = pd.DataFrame(
        {"age": [20, 30, 40, 50], "segment": ["A", "B", "A", "B"]}
    )
    synthetic = pd.DataFrame({"age": [25, 35], "segment": ["A", "B"]})

    result = validate_constraints(real, synthetic, discover_constraints(real))

    assert result["status"] == "computed"
    assert result["violations"] == 0
    assert result["overall_validity_score"] == 1.0


def test_constraint_validation_reports_out_of_range_unknown_and_non_integer() -> None:
    real = pd.DataFrame(
        {"age": [20, 30, 40, 50], "segment": ["A", "B", "A", "B"]}
    )
    synthetic = pd.DataFrame({"age": [19.5, 55], "segment": ["unknown", "A"]})

    result = validate_constraints(real, synthetic, discover_constraints(real))

    assert result["violations"] > 0
    assert result["violation_rate"] > 0
    assert result["per_column"]["age"]["violations"]["min"] == 1
    assert result["per_column"]["age"]["violations"]["integer_only"] == 1
    assert result["per_column"]["segment"]["violations"]["allowed_categories"] == 1
