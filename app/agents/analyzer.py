"""Deterministic statistical analysis of tabular datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas.schemas import DatasetAnalysis
from app.utils.data_utils import infer_column_types
from app.utils.logging_config import get_logger

logger = get_logger("Analyzer")

TARGET_NAME_CANDIDATES = (
    "target",
    "label",
    "class",
    "y",
    "churn",
    "default",
    "approved",
    "outcome",
)


class DatasetAnalyzer:
    def analyze(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        auto_detect_target: bool = False,
    ) -> DatasetAnalysis:
        logger.info("Starting dataset analysis")
        if df.empty:
            raise ValueError("Cannot analyze an empty dataframe.")

        numerical, categorical = infer_column_types(df)
        n_rows, n_cols = df.shape
        missing_by_column = {
            col: float(df[col].isna().mean() * 100) for col in df.columns
        }
        missing_percentage = float(df.isna().mean().mean() * 100)
        duplicate_rows = int(df.duplicated().sum())
        duplicate_percentage = float(duplicate_rows / n_rows * 100)

        target, target_detected, target_reason = self._resolve_target(
            df, target_column, auto_detect_target=auto_detect_target
        )
        task_type = None
        class_distribution: dict[str, float] | None = None
        if target is not None:
            task_type = self._infer_task_type(df[target])
            if task_type == "classification":
                class_distribution = {
                    str(k): float(v)
                    for k, v in df[target].value_counts(normalize=True).items()
                }

        unique_value_stats = {col: int(df[col].nunique(dropna=True)) for col in df.columns}
        numerical_stats = self._numerical_stats(df, numerical)
        categorical_distributions = self._categorical_distributions(df, categorical)
        low_cardinality = [
            col
            for col, n in unique_value_stats.items()
            if n <= 12 and col != target
        ]
        correlations = self._correlations(df, numerical)
        issues, details = self._detect_issues(
            n_rows=n_rows,
            missing_percentage=missing_percentage,
            duplicate_percentage=duplicate_percentage,
            task_type=task_type,
            class_distribution=class_distribution,
            unique_value_stats=unique_value_stats,
            numerical_stats=numerical_stats,
            numerical=numerical,
            categorical=categorical,
        )

        for issue in issues:
            logger.info("Detected %s", issue)

        logger.info("Analysis complete: %s rows, %s columns", n_rows, n_cols)
        return DatasetAnalysis(
            rows=n_rows,
            columns=n_cols,
            column_names=list(df.columns),
            numerical_columns=numerical,
            categorical_columns=categorical,
            missing_percentage=round(missing_percentage, 4),
            missing_by_column={k: round(v, 4) for k, v in missing_by_column.items()},
            duplicate_percentage=round(duplicate_percentage, 4),
            duplicate_rows=duplicate_rows,
            target_column=target,
            target_detected=target_detected,
            target_detection_reason=target_reason,
            task_type=task_type,
            class_distribution=class_distribution,
            unique_value_stats=unique_value_stats,
            numerical_stats=numerical_stats,
            categorical_distributions=categorical_distributions,
            low_cardinality_columns=low_cardinality,
            correlations=correlations,
            issues=issues,
            issue_details=details,
        )

    def _resolve_target(
        self,
        df: pd.DataFrame,
        target_column: str | None,
        auto_detect_target: bool,
    ) -> tuple[str | None, bool, str | None]:
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' is not in the dataset.")
            return target_column, True, "Target column was explicitly selected by the user."
        if not auto_detect_target:
            logger.info("No target column selected")
            return None, False, None
        lowered = {c.lower(): c for c in df.columns}
        for candidate in TARGET_NAME_CANDIDATES:
            if candidate in lowered:
                logger.info("Auto-detected target column '%s'", lowered[candidate])
                return (
                    lowered[candidate],
                    True,
                    f"Auto-detected by common target-like column name '{candidate}'.",
                )
        logger.info("No target column auto-detected")
        return None, False, "No conservative target-column match was found."

    def _infer_task_type(self, series: pd.Series) -> str:
        nunique = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(series) and nunique > 12:
            return "regression"
        return "classification"

    def _numerical_stats(
        self, df: pd.DataFrame, numerical: list[str]
    ) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for col in numerical:
            s = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "mean": float(s.mean()) if s.notna().any() else 0.0,
                "std": float(s.std(ddof=0)) if s.notna().any() else 0.0,
                "min": float(s.min()) if s.notna().any() else 0.0,
                "max": float(s.max()) if s.notna().any() else 0.0,
                "skew": float(s.skew()) if s.notna().sum() > 2 else 0.0,
            }
        return stats

    def _categorical_distributions(
        self, df: pd.DataFrame, categorical: list[str]
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for col in categorical:
            vc = df[col].astype(str).value_counts(normalize=True).head(20)
            out[col] = {str(k): float(v) for k, v in vc.items()}
        return out

    def _correlations(
        self, df: pd.DataFrame, numerical: list[str]
    ) -> dict[str, dict[str, float]]:
        if len(numerical) < 2:
            return {}
        corr = df[numerical].corr(numeric_only=True).fillna(0.0)
        return {
            r: {c: float(corr.loc[r, c]) for c in corr.columns}
            for r in corr.index
        }

    def _detect_issues(
        self,
        n_rows: int,
        missing_percentage: float,
        duplicate_percentage: float,
        task_type: str | None,
        class_distribution: dict[str, float] | None,
        unique_value_stats: dict[str, int],
        numerical_stats: dict[str, dict[str, float]],
        numerical: list[str],
        categorical: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        issues: list[str] = []
        details: dict[str, str] = {}

        if missing_percentage > 1:
            key = "high_missing_values" if missing_percentage > 10 else "missing_values"
            issues.append(key)
            details[key] = f"Overall missing-value rate is {missing_percentage:.2f}%."

        if duplicate_percentage > 1:
            issues.append("duplicate_rows")
            details["duplicate_rows"] = f"{duplicate_percentage:.2f}% of rows are duplicates."

        if class_distribution:
            minority = min(class_distribution.values())
            majority = max(class_distribution.values())
            if minority <= 0.15 or majority >= 0.80:
                issues.append("severe_class_imbalance")
                details["severe_class_imbalance"] = (
                    f"Minority class share is {minority:.1%}; majority is {majority:.1%}."
                )
            elif minority <= 0.30:
                issues.append("class_imbalance")
                details["class_imbalance"] = f"Minority class share is {minority:.1%}."

        mean_unique_ratio = float(np.mean([v / max(n_rows, 1) for v in unique_value_stats.values()]))
        if mean_unique_ratio < 0.05:
            issues.append("low_diversity")
            details["low_diversity"] = (
                f"Average unique-value ratio is {mean_unique_ratio:.3f}."
            )

        skewed = [
            col
            for col, st in numerical_stats.items()
            if abs(st.get("skew", 0.0)) > 2
        ]
        if skewed:
            issues.append("skewed_numerical_distributions")
            details["skewed_numerical_distributions"] = (
                "Highly skewed columns: " + ", ".join(skewed[:8])
            )

        if numerical and categorical:
            issues.append("mixed_column_types")
            details["mixed_column_types"] = (
                f"{len(numerical)} numerical and {len(categorical)} categorical columns."
            )

        if n_rows < 200:
            issues.append("small_dataset")
            details["small_dataset"] = f"Only {n_rows} rows; simpler generators are safer."

        high_card = [c for c, n in unique_value_stats.items() if c in categorical and n > 50]
        if high_card:
            issues.append("high_cardinality_categorical")
            details["high_cardinality_categorical"] = ", ".join(high_card[:8])

        return issues, details
