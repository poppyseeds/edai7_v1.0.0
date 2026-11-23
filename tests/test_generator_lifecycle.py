import random

import numpy as np
import pandas as pd
import pytest

from app.agents.generator import GeneratorAgent
from app.agents.planner import GenerationPlanner
from app.agents.analyzer import DatasetAnalyzer
from app.generators.base import GeneratorConfig, SamplingConfig
from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator
from app.generators.ctgan_generator import CTGANGenerator
from app.generators.tvae_generator import TVAEGenerator
from app.generators.registry import (
    GENERATOR_REGISTRY, GeneratorRegistration, GeneratorCapabilities,
    register_generator, create_generator,
)
from app.generators.runtime import resolve_device
from app.schemas.schemas import GenerationPlan
from app.utils.exceptions import GeneratorError


def table():
    return pd.DataFrame({"amount": np.linspace(10, 100, 40), "segment": ["a", "b"] * 20})


@pytest.mark.parametrize("factory", [BootstrapGenerator, GaussianCopulaGenerator])
def test_fit_once_sample_twice_and_metadata(factory):
    original = table()
    before = original.copy(deep=True)
    model = factory()
    with pytest.raises(GeneratorError, match="fitted"):
        model.sample(4)
    model.fit(original, GeneratorConfig(random_state=8))
    assert len(model.sample(4)) == 4
    conditioned = model.sample(7, SamplingConfig(condition_column="segment", condition_value="a"))
    assert len(conditioned) == 7 and set(conditioned.segment) == {"a"}
    assert model.get_metadata()["fitted"] is True
    assert model.get_metadata()["config"]["device"] == "cpu"
    assert model.get_capabilities()["modalities"] == ("tabular",)
    pd.testing.assert_frame_equal(original, before)
    with pytest.raises(GeneratorError, match="positive integer"):
        model.sample(0)
    with pytest.raises(GeneratorError, match="condition_value"):
        model.sample(2, SamplingConfig(condition_column="segment"))


def test_registered_adapter_works_without_schema_or_orchestrator_changes(monkeypatch, tmp_path):
    class NewBaseline(BootstrapGenerator):
        name = "research_baseline"

    monkeypatch.setattr("app.generators.registry.GENERATOR_REGISTRY", GENERATOR_REGISTRY.copy())
    register_generator("research_baseline", GeneratorRegistration(
        NewBaseline, None, GeneratorCapabilities("resampling", True), "test-only baseline"))
    analysis = DatasetAnalyzer().analyze(table())
    plan = GenerationPlanner().plan(analysis, preferred_generator="research_baseline")
    out = GeneratorAgent().generate(table(), plan)
    assert len(out) == plan.num_samples
    assert create_generator("research_baseline").name == "research_baseline"
    from app.pipeline.orchestrator import PipelineConfig, run_pipeline
    output = run_pipeline(table(), config=PipelineConfig(preferred_generator="research_baseline",
        max_iterations=1, output_dir=tmp_path))
    assert output.result.iterations[0].plan.generator == "research_baseline"
    assert output.result.iterations[0].generator_metadata["generator"] == "research_baseline"
    with pytest.raises(GeneratorError, match="already registered"):
        register_generator("research_baseline", GENERATOR_REGISTRY["bootstrap"])
    with pytest.raises(GeneratorError, match="lowercase"):
        register_generator("../bad", GENERATOR_REGISTRY["bootstrap"])


def test_failure_never_substitutes_bootstrap(monkeypatch, tmp_path):
    class Broken:
        def generate(self, **kwargs):
            raise RuntimeError("training exploded")

    agent = GeneratorAgent()
    monkeypatch.setitem(agent._registry, "ctgan", Broken())
    path = tmp_path / "output.csv"
    with pytest.raises(GeneratorError, match="no substitute model.*training exploded"):
        agent.generate(table(), GenerationPlan(generator="ctgan", num_samples=5, reason="test"), path)
    assert not path.exists()
    assert agent.last_metadata == {}


def test_unavailable_and_unknown_generators_rejected_before_training():
    analysis = DatasetAnalyzer().analyze(table())
    with pytest.raises(GeneratorError, match="unavailable"):
        GenerationPlanner().plan(analysis, preferred_generator="tabddpm")
    with pytest.raises(GeneratorError, match="Unknown"):
        GenerationPlanner().plan(analysis, preferred_generator="made_up")


def test_cuda_preflight_and_model_specific_options(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GeneratorError, match="unavailable"):
        resolve_device("cuda", True)
    with pytest.raises(GeneratorError, match="CPU execution only"):
        resolve_device("cuda", False)
    assert resolve_device("auto", True) == "cpu"
    with pytest.raises(GeneratorError, match="multiple of 10"):
        CTGANGenerator().fit(table(), GeneratorConfig(batch_size=11))
    with pytest.raises(GeneratorError, match="does not support"):
        GaussianCopulaGenerator().fit(table(), GeneratorConfig(epochs=1))
    invalid = table()
    invalid.loc[0, "amount"] = np.inf
    with pytest.raises(GeneratorError, match="Infinite"):
        BootstrapGenerator().fit(invalid)
    with pytest.raises(ValueError):
        GeneratorConfig(epochs=0)


def test_neural_training_seeded_before_fit_and_rngs_restored(monkeypatch):
    import torch
    from sdv import single_table

    seen = []
    class Metadata:
        def to_dict(self):
            return {}

    class FakeCTGAN:
        def __init__(self, metadata, epochs, batch_size, verbose, enable_gpu):
            assert epochs == 1 and batch_size == 20 and enable_gpu is False

        def fit(self, data):
            seen.append((random.random(), np.random.rand(), torch.rand(1).item()))

        def _set_random_state(self, value):
            assert value == 9

    monkeypatch.setattr(single_table, "CTGANSynthesizer", FakeCTGAN)
    monkeypatch.setattr("app.generators.sdv_generator.build_metadata", lambda data: Metadata())
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    for _ in range(2):
        CTGANGenerator().fit(table(), GeneratorConfig(random_state=9, epochs=1, batch_size=20))
    assert seen[0] == seen[1]
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("factory", [CTGANGenerator, TVAEGenerator])
def test_real_neural_cpu_fit_and_sample(factory):
    # Actual tiny SDV training; no injected model and no fallback allowed.
    model = factory().fit(table(), GeneratorConfig(random_state=7, epochs=1, batch_size=20))
    output = model.sample(3)
    assert len(output) == 3 and list(output.columns) == list(table().columns)
    assert model.get_metadata()["config"]["epochs"] == 1
    assert model.get_metadata()["config"]["device"] == "cpu"


def test_sdv_conditional_sampling_uses_configured_retry_budget(monkeypatch):
    from app.generators.sdv_common import sample_sdv

    observed = {}

    class FakeSynthesizer:
        def sample_from_conditions(self, conditions, max_tries_per_batch):
            observed["conditions"] = conditions
            observed["max_tries_per_batch"] = max_tries_per_batch
            return pd.DataFrame({"target": [1, 1]})

    output = sample_sdv(
        FakeSynthesizer(),
        2,
        condition_column="target",
        condition_value=1,
        max_tries_per_batch=750,
    )

    assert len(output) == 2
    assert observed["max_tries_per_batch"] == 750
    assert observed["conditions"][0].get_num_rows() == 2
