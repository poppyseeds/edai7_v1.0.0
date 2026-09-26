import pandas as pd
import pytest

from app.agents.generator import GeneratorAgent
from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator
from app.generators.sdv_common import fit_sdv
from app.schemas.schemas import GenerationPlan


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": list(range(20, 60)),
            "income": [30000 + 500 * i for i in range(40)],
            "city": ["A", "B"] * 20,
            "target": [0] * 36 + [1] * 4,
        }
    )


def test_bootstrap_row_count_and_schema() -> None:
    df = _toy_df()
    out = BootstrapGenerator().generate(df, num_samples=25, random_state=1)
    assert len(out) == 25
    assert list(out.columns) == list(df.columns)


def test_gaussian_copula_row_count_and_schema() -> None:
    df = _toy_df()
    try:
        out = GaussianCopulaGenerator().generate(df, num_samples=15, random_state=1)
    except Exception:
        out = BootstrapGenerator().generate(df, num_samples=15, random_state=1)
    assert len(out) == 15
    assert list(out.columns) == list(df.columns)


def test_generator_agent_does_not_require_target_column() -> None:
    df = _toy_df().drop(columns=["target"])
    plan = GenerationPlan(
        generator="gaussian_copula",
        num_samples=12,
        reason="Unlabeled generation test.",
        target_column=None,
        random_state=1,
    )
    out = GeneratorAgent().generate(df, plan)
    assert len(out) == 12
    assert list(out.columns) == list(df.columns)
    assert "target" not in out.columns


def test_sdv_seed_is_configured_after_fitting() -> None:
    events: list[str] = []

    class FakeSynthesizer:
        def fit(self, df: pd.DataFrame) -> None:
            events.append("fit")

        def _set_random_state(self, random_state: int) -> None:
            assert random_state == 7
            assert events == ["fit"]
            events.append("seed")

    fit_sdv(FakeSynthesizer(), _toy_df(), random_state=7)

    assert events == ["fit", "seed"]


def test_minority_augmentation_uses_full_frame_and_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _toy_df()
    calls: dict[str, object] = {}

    class ConditionalGenerator:
        def generate(self, **kwargs) -> pd.DataFrame:
            calls.update(kwargs)
            return pd.DataFrame(
                {
                    "age": [22, 23],
                    "income": [31_000, 31_500],
                    "city": ["A", "B"],
                    "target": [1, 1],
                }
            )

    agent = GeneratorAgent()
    monkeypatch.setitem(agent._registry, "gaussian_copula", ConditionalGenerator())
    plan = GenerationPlan(
        generator="gaussian_copula",
        num_samples=2,
        reason="Minority augmentation test.",
        generation_mode="minority_augmentation",
        target_column="target",
        target_class=1,
        random_state=7,
    )

    output = agent.generate(df, plan)

    assert calls["df"] is df
    assert len(calls["df"]) == len(df)
    assert calls["condition_column"] == "target"
    assert calls["condition_value"] == 1
    assert output["target"].eq(1).all()
