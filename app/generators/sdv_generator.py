"""Reusable SDV lifecycle; existing adapter names and generate calls remain valid."""
from __future__ import annotations

import inspect
from importlib.metadata import version
from typing import Any

import pandas as pd

from app.generators.base import BaseSyntheticGenerator, GeneratorConfig, SamplingConfig
from app.generators.runtime import resolve_device, seeded_execution
from app.generators.sdv_common import build_metadata, fit_sdv, sample_sdv, validate_generated_output
from app.utils.exceptions import GeneratorError


class SDVGenerator(BaseSyntheticGenerator):
    synthesizer_class: str
    neural = False

    def fit(self, data: pd.DataFrame, config: GeneratorConfig | None = None) -> SDVGenerator:
        self._synthesizer = None
        self.validate_input(data)
        if len(data) < 10:
            raise GeneratorError(f"{self.name} requires at least 10 training rows.")
        cfg = config or GeneratorConfig()
        if cfg.advanced is not None:
            raise GeneratorError(f"{self.name} does not accept advanced diffusion configuration.")
        if not self.neural and (cfg.epochs is not None or cfg.batch_size is not None):
            raise GeneratorError(f"{self.name} does not support epochs or batch_size.")
        if self.name == "ctgan" and cfg.batch_size is not None and cfg.batch_size % 10:
            raise GeneratorError("CTGAN batch_size must be a multiple of 10 (default pac=10).")
        device = resolve_device(cfg.device, self.neural)
        resolved = cfg.model_dump()
        resolved["device"] = device
        if self.neural:
            resolved["epochs"] = cfg.epochs or 100
            resolved["batch_size"] = cfg.batch_size or 500
        self._config = GeneratorConfig(**resolved)
        try:
            from sdv import single_table

            constructor = getattr(single_table, self.synthesizer_class)
            with seeded_execution(cfg.random_state, self.neural, device):
                metadata = build_metadata(data)
                kwargs: dict[str, Any] = {}
                if self.neural:
                    kwargs.update(epochs=self._config.epochs, batch_size=self._config.batch_size, verbose=False)
                    parameters = inspect.signature(constructor).parameters
                    if "enable_gpu" in parameters:
                        kwargs["enable_gpu"] = device == "cuda"
                    elif "cuda" in parameters:
                        kwargs["cuda"] = device == "cuda"
                    else:
                        raise GeneratorError("Installed SDV version has no supported device configuration API.")
                synthesizer = constructor(metadata, **kwargs)
                fit_sdv(synthesizer, data, random_state=cfg.random_state)
            self._columns = data.iloc[:0].copy()
            self._metadata = metadata.to_dict()
            self._synthesizer = synthesizer
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = "Out of memory. Choose an explicit smaller batch/model configuration; settings were not changed automatically."
            raise GeneratorError(f"{self.name} fitting failed: {message}") from exc
        return self

    def sample(self, num_samples: int, config: SamplingConfig | None = None) -> pd.DataFrame:
        cfg = config or SamplingConfig()
        self._validate_sampling(num_samples, cfg)
        if getattr(self, "_synthesizer", None) is None:
            raise GeneratorError(f"{self.name} must be fitted before sampling.")
        if cfg.condition_column is not None and cfg.condition_column not in self._columns.columns:
            raise GeneratorError(f"Unknown condition column '{cfg.condition_column}'.")
        try:
            with seeded_execution(self._config.random_state, self.neural, self._config.device):
                output = sample_sdv(
                    self._synthesizer,
                    num_samples,
                    cfg.condition_column,
                    cfg.condition_value,
                    max_tries_per_batch=cfg.max_tries_per_batch,
                )
            sampling_metadata = cfg.model_dump()
            value = sampling_metadata.get("condition_value")
            if hasattr(value, "item") and callable(value.item):
                sampling_metadata["condition_value"] = value.item()
            self._last_sampling_config = sampling_metadata
            return validate_generated_output(self._columns, output, num_samples)
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = "Out of memory during sampling. Choose an explicit smaller sampling configuration; no settings were changed automatically."
            raise GeneratorError(f"{self.name} sampling failed: {message}") from exc

    def get_metadata(self) -> dict[str, Any]:
        fitted = getattr(self, "_synthesizer", None) is not None
        return {
            "generator": self.name, "modality": "tabular", "fitted": fitted,
            "library": "sdv", "library_version": version("sdv"),
            "config": self._config.model_dump() if fitted else None,
            "last_sampling_config": getattr(self, "_last_sampling_config", None),
            "schema": self._metadata if fitted else None,
            "training_determinism": "seeded; bitwise reproducibility depends on library/device",
        }
