"""Sequence generator contract distinct from the tabular dataframe lifecycle."""
from pathlib import Path
from typing import Literal
import json
import os
import subprocess
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import get_settings
from app.research.sources import ROOT, SOURCES, source_path, worker_python
from app.generators.runtime import MODEL_LOCK
from app.utils.exceptions import GeneratorError


class SequenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    random_state: int = Field(default=42, ge=0, le=2**32-1)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    hidden_dim: int = Field(default=64, ge=4)
    layers: int = Field(default=2, ge=1)
    heads: int = Field(default=4, ge=1)
    timesteps: int = Field(default=500, ge=2)
    sampling_steps: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=0.0001, gt=0)
    threads: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def consistent(self):
        if self.hidden_dim % self.heads or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by heads.")
        if self.sampling_steps > self.timesteps:
            raise ValueError("sampling_steps cannot exceed training timesteps.")
        return self


class SequenceGenerator:
    name = "diffusion_ts"

    def _worker(self, operation: str, count: int = 0):
        command = [worker_python(), str(ROOT / "scripts" / "research_worker.py"), "--model", self.name,
                   "--operation", operation, "--directory", str(self.directory), "--count", str(count)]
        with MODEL_LOCK:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=int(os.environ.get("RESEARCH_TIMEOUT_SECONDS", "7200")))
        if result.returncode:
            raise GeneratorError(f"{self.name} worker failed: {result.stderr[-3000:]}")

    def fit(self, windows: np.ndarray, config: SequenceConfig | None = None, metadata: dict | None = None):
        self._fitted = False
        if windows.ndim != 3 or windows.shape[0] < 2 or windows.shape[1] < 8 or windows.shape[2] < 1 or not np.isfinite(windows).all():
            raise GeneratorError("Expected at least two finite windows shaped [sequence, timestep>=8, feature].")
        source = source_path(self.name, verify=True) if self.name in SOURCES else None
        cfg = config or SequenceConfig()
        self.directory = Path(get_settings().models_dir).resolve() / f"{self.name}_{uuid4().hex}"
        self.directory.mkdir(parents=True)
        np.save(self.directory / "training.npy", windows.astype("float32"))
        request = {"config": cfg.model_dump(), "shape": list(windows.shape), "temporal_metadata": metadata,
                   "source": str(source) if source else None, "source_commit": SOURCES[self.name]["commit"] if source else None}
        (self.directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
        self._worker("fit")
        self._fitted = True
        return self

    def sample(self, num_sequences: int):
        if not getattr(self, "_fitted", False) or isinstance(num_sequences, bool) or not isinstance(num_sequences, int) or num_sequences < 1:
            raise GeneratorError("Fit the model and request a positive integer number of sequences.")
        self._worker("sample", num_sequences)
        values = np.load(self.directory / "synthetic.npy", allow_pickle=False)
        shape = json.loads((self.directory / "request.json").read_text(encoding="utf-8"))["shape"]
        if values.shape != (num_sequences, *shape[1:]) or not np.isfinite(values).all():
            raise GeneratorError("Generated sequence shape/finiteness validation failed.")
        return values

    def get_metadata(self):
        return {**json.loads((self.directory / "metadata.json").read_text(encoding="utf-8")), "checkpoint_dir": str(self.directory)}

    @classmethod
    def load(cls, directory: str | Path):
        model = cls()
        model.directory = Path(directory).resolve()
        request = json.loads((model.directory / "request.json").read_text(encoding="utf-8"))
        if cls.name in SOURCES:
            source_path(cls.name, verify=True)
            if request["source_commit"] != SOURCES[cls.name]["commit"]:
                raise GeneratorError("Checkpoint source revision mismatch.")
        if not (model.directory / "checkpoint.pt").exists():
            raise GeneratorError("Missing sequence checkpoint.")
        model._fitted = True
        return model


class TimeGANGenerator(SequenceGenerator):
    name = "timegan_torch"


class WindowBootstrapGenerator:
    """Explicit sequence-resampling baseline; exact-copy privacy risk is expected."""
    name = "window_bootstrap"

    def fit(self, windows, config=None, metadata=None):
        import time
        start = time.perf_counter()
        self.values = np.asarray(windows).copy()
        self.config = config or SequenceConfig()
        if self.config.device == "cuda":
            raise GeneratorError("Window bootstrap is a CPU-only baseline.")
        if self.values.ndim != 3 or not np.isfinite(self.values).all():
            raise GeneratorError("Expected finite sequence windows.")
        self.metadata = {"generator": self.name, "training_seconds": time.perf_counter()-start,
                         "config": {"random_state": self.config.random_state}, "device": "cpu",
                         "model_size_bytes": None, "peak_gpu_allocated_bytes": None}
        return self

    def sample(self, count):
        return self.values[np.random.default_rng(self.config.random_state).integers(0, len(self.values), count)].copy()

    def get_metadata(self):
        return self.metadata


SEQUENCE_GENERATORS = {"diffusion_ts": SequenceGenerator, "timegan_torch": TimeGANGenerator,
                       "window_bootstrap": WindowBootstrapGenerator}
