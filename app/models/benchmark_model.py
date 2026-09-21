"""Shared sklearn pipelines for downstream classification/regression."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas.schemas import ModelMetrics
from app.utils.data_utils import infer_column_types
from app.utils.exceptions import BenchmarkError


class DownstreamModel:
    def __init__(self, task_type: str, random_state: int = 42, model_name: str = "random_forest"):
        self.task_type = task_type
        self.random_state = random_state
        self.model_name = model_name
        self.pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DownstreamModel":
        numerical, categorical = infer_column_types(X)
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers = []
        if numerical:
            transformers.append(("num", numeric_pipe, numerical))
        if categorical:
            transformers.append(("cat", categorical_pipe, categorical))
        if not transformers:
            raise BenchmarkError("No usable features for the downstream model.")
        preprocessor = ColumnTransformer(transformers=transformers)
        estimator = self._make_estimator()
        self.pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise BenchmarkError("Model is not fitted.")
        return self.pipeline.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, n_train: int) -> ModelMetrics:
        if self.pipeline is None:
            raise BenchmarkError("Model is not fitted.")
        preds = self.pipeline.predict(X)
        if self.task_type == "classification":
            return self._classification_metrics(y, preds, X, n_train)
        return self._regression_metrics(y, preds, n_train)

    def _make_estimator(self):
        if self.task_type == "classification":
            if self.model_name == "logistic_regression":
                return LogisticRegression(max_iter=400, random_state=self.random_state)
            return RandomForestClassifier(
                n_estimators=80,
                random_state=self.random_state,
                n_jobs=1,
            )
        if self.model_name == "ridge":
            return Ridge(random_state=self.random_state)
        return RandomForestRegressor(
            n_estimators=80,
            random_state=self.random_state,
            n_jobs=1,
        )

    def _classification_metrics(
        self, y: pd.Series, preds, X: pd.DataFrame, n_train: int
    ) -> ModelMetrics:
        roc = None
        try:
            if hasattr(self.pipeline, "predict_proba") and y.nunique() == 2:
                proba = self.pipeline.predict_proba(X)
                pos = proba[:, 1]
                roc = float(roc_auc_score(y, pos))
        except Exception:
            roc = None
        f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        return ModelMetrics(
            task_type="classification",
            model_name=self.model_name,
            n_train=n_train,
            n_test=int(len(y)),
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, average="macro", zero_division=0)),
            recall=float(recall_score(y, preds, average="macro", zero_division=0)),
            f1=f1,
            roc_auc=roc,
            primary_metric="f1",
            primary_value=f1,
        )

    def _regression_metrics(self, y: pd.Series, preds, n_train: int) -> ModelMetrics:
        mae = float(mean_absolute_error(y, preds))
        rmse = float(mean_squared_error(y, preds) ** 0.5)
        r2 = float(r2_score(y, preds))
        return ModelMetrics(
            task_type="regression",
            model_name=self.model_name,
            n_train=n_train,
            n_test=int(len(y)),
            mae=mae,
            rmse=rmse,
            r2=r2,
            primary_metric="r2",
            primary_value=r2,
        )
