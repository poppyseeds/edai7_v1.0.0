"""Selects and runs a tabular synthesizer from a GenerationPlan."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator
from app.generators.ctgan_generator import CTGANGenerator
from app.generators.tvae_generator import TVAEGenerator
from app.schemas.schemas import GenerationPlan
from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")


class GeneratorAgent:
    def __init__(self) -> None:
        self._registry = {
            "ctgan": CTGANGenerator(),
            "tvae": TVAEGenerator(),
            "gaussian_copula": GaussianCopulaGenerator(),
        }
        self._fallback = BootstrapGenerator()

    def generate(
        self,
        df: pd.DataFrame,
        plan: GenerationPlan,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        if len(df) < 10:
            raise GeneratorError("Need at least 10 rows to fit a generator.")
        generator = self._registry.get(plan.generator)
        if generator is None:
            raise GeneratorError(f"Unknown generator '{plan.generator}'.")

        try:
            synthetic = generator.generate(
                df=df,
                num_samples=plan.num_samples,
                random_state=plan.random_state,
                epochs=plan.epochs,
            )
        except Exception as exc:
            logger.warning("%s failed (%s); using bootstrap fallback", plan.generator, exc)
            synthetic = self._fallback.generate(
                df=df,
                num_samples=plan.num_samples,
                random_state=plan.random_state,
            )

        synthetic = self._align_schema(df, synthetic)
        if plan.target_strategy == "minority_oversampling" and plan.target_column:
            synthetic = self._boost_minority(df, synthetic, plan)

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            synthetic.to_csv(path, index=False)
            logger.info("Saved synthetic data to %s", path)
        return synthetic

    def _align_schema(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in original.columns if c not in synthetic.columns]
        if missing:
            raise GeneratorError(f"Synthetic data missing columns: {missing}")
        out = synthetic[original.columns].copy()
        for col in original.columns:
            if pd.api.types.is_numeric_dtype(original[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.dropna(how="all").reset_index(drop=True)

    def _boost_minority(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plan: GenerationPlan,
    ) -> pd.DataFrame:
        target = plan.target_column
        if target not in original.columns or target not in synthetic.columns:
            return synthetic
        counts = original[target].value_counts()
        minority_label = counts.idxmin()
        minority = synthetic[synthetic[target] == minority_label]
        if minority.empty:
            return synthetic
        needed = plan.num_samples
        if len(minority) >= needed:
            logger.info("Keeping %s minority synthetic rows", needed)
            return minority.sample(n=needed, random_state=plan.random_state).reset_index(drop=True)
        extra = synthetic[synthetic[target] != minority_label]
        n_extra = max(0, needed - len(minority))
        if extra.empty or n_extra == 0:
            return minority.reset_index(drop=True)
        mixed = pd.concat(
            [minority, extra.sample(n=min(n_extra, len(extra)), random_state=plan.random_state)],
            ignore_index=True,
        )
        return mixed.reset_index(drop=True)
