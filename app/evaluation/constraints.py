"""Deterministic, evidence-based tabular constraint discovery and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.utils.data_utils import infer_semantic_types


def discover_constraints(
    df: pd.DataFrame,
    semantic_types: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Discover conservative column constraints without making domain assumptions."""

    semantics = semantic_types or infer_semantic_types(df)
    columns: dict[str, list[dict[str, Any]]] = {}
    for column in df.columns:
        rules: list[dict[str, Any]] = []
        semantic = semantics.get(str(column), {}).get("semantic_type")
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and semantic not in {"boolean", "identifier"}:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                rules.extend(
                    [
                        {"kind": "min", "value": float(numeric.min())},
                        {"kind": "max", "value": float(numeric.max())},
                    ]
                )
                if semantic == "integer" or np.allclose(numeric, np.round(numeric)):
                    rules.append({"kind": "integer_only"})
                if float(numeric.min()) >= 0:
                    rules.append({"kind": "non_negative"})
                if float(numeric.min()) >= 0 and float(numeric.max()) <= 1:
                    rules.append({"kind": "unit_interval"})
                if numeric.nunique() == 1:
                    rules.append({"kind": "constant", "value": float(numeric.iloc[0])})
        elif semantic == "datetime":
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().any():
                rules.extend(
                    [
                        {"kind": "min_datetime", "value": parsed.min().isoformat()},
                        {"kind": "max_datetime", "value": parsed.max().isoformat()},
                    ]
                )
        elif semantic in {"categorical", "ordinal", "boolean", "sensitive"}:
            values = series.dropna().astype(str)
            categories = values.value_counts()
            if 0 < len(categories) <= 100:
                rules.append({"kind": "allowed_categories", "values": sorted(categories.index.tolist())})
                rare = categories[categories / max(len(values), 1) <= 0.02].index.tolist()
                if rare:
                    rules.append({"kind": "rare_categories", "values": sorted(rare)})
        if rules:
            columns[str(column)] = rules
    return {"status": "computed" if columns else "not_applicable", "columns": columns}


def validate_constraints(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return violations rather than mutating synthetic records."""

    discovered = constraints or discover_constraints(real_df)
    per_column: dict[str, dict[str, Any]] = {}
    total_checks = 0
    total_violations = 0
    for column, rules in discovered.get("columns", {}).items():
        if column not in synthetic_df.columns:
            per_column[column] = {"missing_column": len(synthetic_df)}
            total_checks += len(synthetic_df)
            total_violations += len(synthetic_df)
            continue
        series = synthetic_df[column]
        violations: dict[str, int] = {}
        for rule in rules:
            kind = rule["kind"]
            bad = _rule_violations(series, rule)
            if bad is None:
                continue
            count = int(bad.sum())
            total_checks += int(bad.size)
            total_violations += count
            if count:
                violations[kind] = count
        per_column[column] = {"violations": violations, "violation_count": sum(violations.values())}
    if total_checks == 0:
        return {
            "status": "not_applicable",
            "violations": 0,
            "violation_rate": None,
            "overall_validity_score": None,
            "per_column": per_column,
        }
    rate = total_violations / total_checks
    return {
        "status": "computed",
        "violations": total_violations,
        "violation_rate": round(float(rate), 6),
        "overall_validity_score": round(float(np.clip(1.0 - rate, 0.0, 1.0)), 6),
        "per_column": per_column,
    }


def _rule_violations(series: pd.Series, rule: dict[str, Any]) -> pd.Series | None:
    kind = rule["kind"]
    if kind in {"min", "max", "integer_only", "non_negative", "unit_interval", "constant"}:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.notna()
        if kind == "min":
            return valid & (numeric < rule["value"])
        if kind == "max":
            return valid & (numeric > rule["value"])
        if kind == "integer_only":
            return valid & ~np.isclose(numeric, np.round(numeric))
        if kind == "non_negative":
            return valid & (numeric < 0)
        if kind == "unit_interval":
            return valid & ((numeric < 0) | (numeric > 1))
        return valid & ~np.isclose(numeric, rule["value"])
    if kind == "allowed_categories":
        return series.notna() & ~series.astype(str).isin(rule["values"])
    if kind in {"min_datetime", "max_datetime"}:
        parsed = pd.to_datetime(series, errors="coerce")
        bound = pd.Timestamp(rule["value"])
        return parsed.notna() & (parsed < bound if kind == "min_datetime" else parsed > bound)
    return None
