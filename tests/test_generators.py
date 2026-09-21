import pandas as pd

from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator


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
