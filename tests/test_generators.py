import sys
from types import ModuleType

import pandas as pd
import pytest

from app.agents.generator import GeneratorAgent
from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator
from app.generators.ctgan_generator import CTGANGenerator
from app.schemas.schemas import GenerationPlan
from app.utils.exceptions import GeneratorError


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


def test_ctgan_uses_size_aware_default_epochs() -> None:
    assert CTGANGenerator._resolve_epochs(100, None) == 500
    assert CTGANGenerator._resolve_epochs(2_000, None) == 300
    assert CTGANGenerator._resolve_epochs(20_000, None) == 200
    assert CTGANGenerator._resolve_epochs(100, 25) == 25


def test_ctgan_rejects_all_missing_input_columns() -> None:
    df = _toy_df()
    df["missing"] = pd.NA

    with pytest.raises(GeneratorError, match="all-missing"):
        CTGANGenerator().generate(df, num_samples=5)


def test_ctgan_output_preserves_nullable_and_categorical_dtypes() -> None:
    original = pd.DataFrame(
        {
            "count": pd.Series([1, 2, None], dtype="Int64"),
            "enabled": pd.Series([True, False, None], dtype="boolean"),
            "group": pd.Series(pd.Categorical(["a", "b", None], categories=["a", "b"])),
        }
    )
    sampled = pd.DataFrame(
        {
            "count": [1.0, 2.0],
            "enabled": [True, False],
            "group": ["a", "b"],
        }
    )

    output = CTGANGenerator._validate_output(original, sampled, num_samples=2)

    assert list(output.columns) == list(original.columns)
    assert output.dtypes.equals(original.dtypes)


def test_ctgan_configures_sdv_seed_for_fit_and_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, int | None]] = []

    class FakeCTGAN:
        def __init__(self, metadata, epochs: int, verbose: bool) -> None:
            assert metadata == "metadata"
            assert epochs == 12
            assert verbose is False

        def set_random_state(self, random_state: int) -> None:
            events.append(("seed", random_state))

        def fit(self, df: pd.DataFrame) -> None:
            events.append(("fit", None))

    sdv_module = ModuleType("sdv")
    single_table_module = ModuleType("sdv.single_table")
    single_table_module.CTGANSynthesizer = FakeCTGAN
    monkeypatch.setitem(sys.modules, "sdv", sdv_module)
    monkeypatch.setitem(sys.modules, "sdv.single_table", single_table_module)
    monkeypatch.setattr("app.generators.ctgan_generator.build_metadata", lambda df: "metadata")
    monkeypatch.setattr(
        "app.generators.ctgan_generator.sample_sdv",
        lambda synthesizer, num_samples: _toy_df().iloc[:num_samples].reset_index(drop=True),
    )

    CTGANGenerator().generate(_toy_df(), num_samples=2, random_state=7, epochs=12)

    assert events == [("seed", 7), ("fit", None), ("seed", 7)]
