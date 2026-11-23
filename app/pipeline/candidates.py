"""Bounded, reproducible generator candidate proposals and run-local cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import pandas as pd

from app.config import get_settings
from app.generators.registry import get_available_generators
from app.schemas.schemas import GenerationPlan


GENERATOR_ORDER = ("gaussian_copula", "bootstrap", "ctgan", "tvae")
SAMPLE_FACTORS = (1.0, 0.75, 1.25)


@dataclass(frozen=True)
class GeneratorCandidate:
    candidate_id: str
    generator_name: str
    sample_count: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def cache_key(self, dataframe: pd.DataFrame, random_state: int, condition: dict[str, Any] | None) -> str:
        digest = sha256()
        digest.update(pd.util.hash_pandas_object(dataframe, index=True).values.tobytes())
        digest.update(str(list(dataframe.columns)).encode())
        digest.update(self.generator_name.encode())
        digest.update(str(self.sample_count).encode())
        digest.update(repr(sorted(self.hyperparameters.items())).encode())
        digest.update(str(random_state).encode())
        digest.update(repr(sorted((condition or {}).items())).encode())
        return digest.hexdigest()


class CandidateCache:
    """In-memory cache scoped to a single orchestration run."""

    def __init__(self) -> None:
        self._values: dict[str, pd.DataFrame] = {}

    def get(self, key: str) -> pd.DataFrame | None:
        value = self._values.get(key)
        return value.copy() if value is not None else None

    def put(self, key: str, value: pd.DataFrame) -> None:
        self._values[key] = value.copy()


def propose_candidates(plan: GenerationPlan) -> list[GeneratorCandidate]:
    """Propose a small ordered search including Bootstrap as a real baseline."""

    settings = get_settings()
    if not settings.enable_candidate_search:
        return [GeneratorCandidate("candidate-1", plan.generator, plan.num_samples, _plan_hyperparameters(plan))]
    availability = get_available_generators(settings.enable_experimental_generators)
    order = [plan.generator] + [name for name in GENERATOR_ORDER if name != plan.generator]
    order = [name for name in order if availability.get(name, {}).get("available")]
    candidates: list[GeneratorCandidate] = []
    maximum = max(1, settings.max_generator_candidates)
    max_samples = plan.max_allowed_samples or plan.num_samples
    for generator_name in order:
        for factor in SAMPLE_FACTORS:
            sample_count = min(max(1, int(round(plan.num_samples * factor))), max_samples)
            if any(item.generator_name == generator_name and item.sample_count == sample_count for item in candidates):
                continue
            candidates.append(
                GeneratorCandidate(
                    candidate_id=f"candidate-{len(candidates) + 1}",
                    generator_name=generator_name,
                    sample_count=sample_count,
                    hyperparameters=_plan_hyperparameters(plan, generator_name),
                )
            )
            if len(candidates) >= maximum:
                return candidates
    return candidates


def _plan_hyperparameters(plan: GenerationPlan, generator_name: str | None = None) -> dict[str, Any]:
    chosen = generator_name or plan.generator
    return {"epochs": plan.epochs} if chosen in {"ctgan", "tvae"} and plan.epochs else {}
