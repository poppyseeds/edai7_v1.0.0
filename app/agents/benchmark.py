"""Train/evaluate the same downstream model on original vs augmented training data."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.evaluation.utility import primary_improvement
from app.models.benchmark_model import DownstreamModel
from app.schemas.schemas import BenchmarkResult, ModelMetrics
from app.utils.exceptions import BenchmarkError
from app.utils.logging_config import get_logger

logger = get_logger("Benchmark")


class BenchmarkAgent:
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
    ) -> BenchmarkResult:
        settings = get_settings()
        min_improvement = (
            settings.min_improvement if min_improvement is None else min_improvement
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
        improved = rel >= min_improvement
        note = (
            f"Relative change in {baseline.primary_metric}: {rel:.4f}. "
            "Test split is held out from original data only."
        )
        return BenchmarkResult(
            baseline=baseline,
            augmented=augmented,
            improvement=improvement,
            improved=improved,
            split_seed=random_state,
            notes=note,
        )
