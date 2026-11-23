"""TabSyn core adapter, running pinned upstream modules in a separate process."""
from pathlib import Path
import json
import os
import subprocess
import sys
from uuid import uuid4

import pandas as pd

from app.config import get_settings
from app.generators.base import BaseSyntheticGenerator, GeneratorConfig, SamplingConfig
from app.research.sources import ROOT, SOURCES, source_path, worker_python
from app.utils.exceptions import GeneratorError


class TabSynGenerator(BaseSyntheticGenerator):
    name = "tabsyn"

    def _worker(self, operation: str, count: int = 0):
        command = [worker_python(), str(ROOT / "scripts" / "research_worker.py"),
                   "--model", self.name, "--operation", operation, "--directory", str(self.directory), "--count", str(count)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=int(os.environ.get("RESEARCH_TIMEOUT_SECONDS", "7200")))
        if completed.returncode:
            raise GeneratorError(f"{self.name} worker failed: {completed.stderr[-3000:]}; see {self.directory}")

    def fit(self, data: pd.DataFrame, config: GeneratorConfig | None = None):
        self.validate_input(data)
        self._fitted = False
        source = source_path(self.name, verify=True)
        cfg = config or GeneratorConfig()
        self.directory = Path(get_settings().models_dir).resolve() / f"{self.name}_{uuid4().hex}"
        self.directory.mkdir(parents=True)
        data.to_csv(self.directory / "training.csv", index=False)
        (self.directory / "request.json").write_text(json.dumps({"config": cfg.model_dump(), "source": str(source),
            "source_commit": SOURCES[self.name]["commit"]}), encoding="utf-8")
        self._worker("fit")
        self._fitted = True
        return self

    def sample(self, num_samples: int, config: SamplingConfig | None = None):
        cfg = config or SamplingConfig()
        self._validate_sampling(num_samples, cfg)
        if not getattr(self, "_fitted", False):
            raise GeneratorError("TabSyn must be fitted before sampling.")
        if cfg.condition_column is not None:
            raise GeneratorError("This TabSyn adapter supports unconditional sampling only; no labels are overwritten.")
        self._worker("sample", num_samples)
        result = pd.read_csv(self.directory / "synthetic.csv")
        if len(result) != num_samples:
            raise GeneratorError("TabSyn returned an incorrect row count.")
        return result

    def get_metadata(self):
        if not getattr(self, "_fitted", False):
            return {"generator": self.name, "fitted": False}
        metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        return {**metadata, "checkpoint_dir": str(self.directory), "fitted": True}

    @classmethod
    def load(cls, directory: str | Path):
        model = cls()
        model.directory = Path(directory).resolve()
        request = json.loads((model.directory / "request.json").read_text(encoding="utf-8"))
        source_path(cls.name, verify=True)
        if request["source_commit"] != SOURCES[cls.name]["commit"]:
            raise GeneratorError("Checkpoint source revision does not match this adapter.")
        if not (model.directory / "checkpoint.pt").exists():
            raise GeneratorError("Missing TabSyn checkpoint.")
        model._fitted = True
        return model
