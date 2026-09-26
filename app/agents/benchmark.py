"""Train/evaluate the same downstream model on original vs augmented training data."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from app.config import get_settings
from app.evaluation.utility import primary_improvement
from app.models.benchmark_model import DownstreamModel
from app.schemas.schemas import (
    BenchmarkResult,
    CrossValidatedModelResult,
    ModelMetrics,
    ModelSuiteResult,
)
from app.utils.exceptions import BenchmarkError
from app.utils.logging_config import get_logger

logger = get_logger("Benchmark")


class BenchmarkAgent:
    def evaluate_model_suite_cv(
        self,
        train: pd.DataFrame,
        target_column: str,
        task_type: str,
        random_state: int = 42,
        folds: int | None = None,
    ) -> ModelSuiteResult:
        """Evaluate several downstream models using training data only.

        This helper deliberately accepts only a training dataframe. The pipeline
        can use it for candidate ranking without exposing its final test split.
        """

        if target_column not in train.columns:
            raise BenchmarkError(f"Missing target column '{target_column}'.")
        requested_folds = folds or get_settings().cv_folds
        splitter = self._make_cv_splitter(train[target_column], task_type, requested_folds, random_state)
        names = (
            ("logistic_regression", "random_forest", "hist_gradient_boosting")
            if task_type == "classification"
            else ("ridge", "random_forest", "hist_gradient_boosting")
        )
        results: dict[str, CrossValidatedModelResult] = {}
        for name in names:
            values: list[float] = []
            for train_index, valid_index in splitter.split(
                train, train[target_column] if task_type == "classification" else None
            ):
                fold_train = train.iloc[train_index].reset_index(drop=True)
                fold_valid = train.iloc[valid_index].reset_index(drop=True)
                metrics = self.evaluate_split(
                    fold_train,
                    fold_valid,
                    target_column,
                    task_type,
                    random_state=random_state,
                    model_name=name,
                )
                values.append(metrics.primary_value)
            results[name] = CrossValidatedModelResult(
                model_name=name,
                primary_metric="f1" if task_type == "classification" else "r2",
                mean_primary_value=round(float(np.mean(values)), 6),
                std_primary_value=round(float(np.std(values, ddof=0)), 6),
                fold_primary_values=[round(float(value), 6) for value in values],
            )
        median = float(np.median([result.mean_primary_value for result in results.values()]))
        return ModelSuiteResult(
            task_type=task_type,
            primary_metric="f1" if task_type == "classification" else "r2",
            folds=requested_folds,
            models=results,
            median_primary_value=round(median, 6),
        )

    @staticmethod
    def _make_cv_splitter(
        target: pd.Series,
        task_type: str,
        folds: int,
        random_state: int):
        if folds < 2:
            raise BenchmarkError("Cross-validation requires at least two folds.")
        if task_type == "classification":
            minimum_class_count = int(target.value_counts().min())
            usable_folds = min(folds, minimum_class_count)
            if usable_folds < 2:
                raise BenchmarkError("Cross-validation requires at least two rows per class.")
            return StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=random_state)
        if len(target) < folds:
            raise BenchmarkError("Cross-validation folds exceed available training rows.")
        return KFold(n_splits=folds, shuffle=True, random_state=random_state)
    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        settings = get_settings()
        test_size = settings.test_size if test_size is None else test_size
        random_state = settings.random_state if random_state is None else random_state
        stratify = df[target_column] if task_type == "classification" else None
        try:
            train, test = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            train, test = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def evaluate_split(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_column: str,
        task_type: str,
        random_state: int = 42,
        model_name: str = "random_forest",
    ) -> ModelMetrics:
        if target_column not in train.columns:
            raise BenchmarkError(f"Missing target column '{target_column}'.")
        X_train = train.drop(columns=[target_column])
        y_train = train[target_column]
        X_test = test.drop(columns=[target_column])
        y_test = test[target_column]
        if task_type == "classification":
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)
        model = DownstreamModel(
            task_type=task_type, random_state=random_state, model_name=model_name
        )
        try:
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test, n_train=len(train))
        except Exception as exc:
            raise BenchmarkError(f"Downstream model training failed: {exc}") from exc
        return metrics

    def compare(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_column: str,
        task_type: str,
        random_state: int = 42,
        min_improvement: float | None = None,
        min_absolute_improvement: float | None = None,
    ) -> BenchmarkResult:
        settings = get_settings()
        min_improvement = (
            settings.min_improvement if min_improvement is None else min_improvement
        )
        min_absolute_improvement = (
            settings.min_absolute_improvement
            if min_absolute_improvement is None
            else min_absolute_improvement
        )
        baseline = self.evaluate_split(
            train, test, target_column, task_type, random_state=random_state
        )
        logger.info("Baseline %s: %.4f", baseline.primary_metric, baseline.primary_value)

        aligned = synthetic.copy()
        extra = [c for c in aligned.columns if c not in train.columns]
        if extra:
            aligned = aligned.drop(columns=extra)
        missing = [c for c in train.columns if c not in aligned.columns]
        if missing:
            raise BenchmarkError(f"Synthetic data missing columns required for ML: {missing}")
        aligned = aligned[train.columns]
        augmented_train = pd.concat([train, aligned], ignore_index=True)
        augmented = self.evaluate_split(
            augmented_train, test, target_column, task_type, random_state=random_state
        )
        logger.info("Augmented %s: %.4f", augmented.primary_metric, augmented.primary_value)

        keys = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "mae",
            "rmse",
            "r2",
            "primary_value",
        ]
        improvement: dict[str, float | None] = {}
        for key in keys:
            b = getattr(baseline, key)
            a = getattr(augmented, key)
            if b is None or a is None:
                improvement[key] = None
            elif key in {"mae", "rmse"}:
                improvement[key] = float(b - a)
            else:
                improvement[key] = float(a - b)

        rel = primary_improvement(baseline, augmented)
        absolute = improvement.get("primary_value")
        absolute = float(absolute) if absolute is not None else 0.0
        improved = rel >= min_improvement and absolute >= min_absolute_improvement
        note = (
            f"Relative change in {baseline.primary_metric}: {rel:.4f}; "
            f"absolute change: {absolute:+.4f}. Acceptance requires at least "
            f"{min_improvement:.1%} relative and +{min_absolute_improvement:.4f} absolute improvement. "
            "The test split contains held-out original rows only."
        )
        return BenchmarkResult(
            baseline=baseline,
            augmented=augmented,
            improvement=improvement,
            improved=improved,
            split_seed=random_state,
            relative_improvement=rel,
            absolute_improvement=absolute,
            min_relative_improvement=min_improvement,
            min_absolute_improvement=min_absolute_improvement,
            notes=note,
        )
