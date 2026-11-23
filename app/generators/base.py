"""Tabular lifecycle contract; sequences will use a separate input contract."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from app.utils.exceptions import GeneratorError

class AdvancedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    vae_epochs: int = Field(default=100, ge=1)
    diffusion_epochs: int = Field(default=100, ge=1)
    token_dim: int = Field(default=4, ge=2)
    hidden_dim: int = Field(default=128, ge=4)
    layers: int = Field(default=2, ge=1)
    factor: int = Field(default=8, ge=1)
    sampling_steps: int = Field(default=50, ge=2)
    learning_rate: float = Field(default=0.001, gt=0)
    beta: float = Field(default=0.001, ge=0)
    threads: int = Field(default=1, ge=1)


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    epochs: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    advanced: AdvancedConfig | None = None

class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    condition_column: str | None = None
    condition_value: Any = None
    # SDV's neural synthesizers use rejection sampling for conditions. The
    # library default of 100 attempts is too small for learned rare classes,
    # so conditional requests use a larger but bounded and explicit budget.
    max_tries_per_batch: int = Field(default=1000, ge=1, le=10000)

class BaseSyntheticGenerator(ABC):
    name: str

    def validate_input(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise GeneratorError("Generator input must be a non-empty pandas DataFrame.")
        if data.columns.duplicated().any() or not all(isinstance(c, str) for c in data.columns):
            raise GeneratorError("Generator columns must be unique strings.")
        numeric = data.select_dtypes(include="number").to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric).any():
            raise GeneratorError("Infinite numeric values require an explicit cleaning policy.")

    def get_capabilities(self) -> dict[str, Any]:
        from dataclasses import asdict
        from app.generators.registry import GENERATOR_REGISTRY
        return asdict(GENERATOR_REGISTRY[self.name].capabilities)

    @abstractmethod
    def fit(self, data: pd.DataFrame, config: GeneratorConfig | None = None) -> BaseSyntheticGenerator:
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_samples: int, config: SamplingConfig | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def generate(self, df: pd.DataFrame, num_samples: int, random_state: int = 42,
                 epochs: int | None = None, condition_column: str | None = None,
                 condition_value: object | None = None, *, config: GeneratorConfig | None = None) -> pd.DataFrame:
        """Backward-compatible train-and-sample convenience operation."""
        sampling = SamplingConfig(condition_column=condition_column, condition_value=condition_value)
        self._validate_sampling(num_samples, sampling)
        self.fit(df, config or GeneratorConfig(random_state=random_state, epochs=epochs))
        return self.sample(num_samples, sampling)

    def _validate_sampling(self, num_samples: int, config: SamplingConfig) -> None:
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise GeneratorError("num_samples must be a positive integer.")
        if config.condition_column is not None and config.condition_value is None:
            raise GeneratorError("condition_value is required with condition_column.")
        if config.condition_column is None and config.condition_value is not None:
            raise GeneratorError("condition_column is required with condition_value.")
