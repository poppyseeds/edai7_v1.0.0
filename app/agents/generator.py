"""Selects and runs a tabular synthesizer from a GenerationPlan."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator
from app.generators.ctgan_generator import CTGANGenerator
from app.generators.tvae_generator import TVAEGenerator
from app.schemas.schemas import GenerationPlan
from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")


MINORITY_MODES = {
    "minority_augmentation",
    "moderate_minority_augmentation",
}


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

        if not plan.generation_needed or plan.num_samples <= 0:
            logger.info(
                "Generation not needed; returning empty synthetic dataframe."
            )
            return df.iloc[0:0].copy()

        if df.empty:
            raise GeneratorError(
                "Cannot generate synthetic data from an empty dataframe."
            )

        if len(df) < 10:
            raise GeneratorError(
                "Need at least 10 rows to fit a synthetic-data generator."
            )

        generator = self._registry.get(plan.generator)

        if generator is None:
            raise GeneratorError(
                f"Unknown generator '{plan.generator}'."
            )

        # IMPORTANT:
        # Always train on the full real training dataframe.
        #
        # For minority augmentation, the target condition is passed
        # to SDV's conditional sampler instead of training only on
        # the minority subset.
        condition_column: str | None = None
        condition_value: object | None = None

        if (
            plan.generation_mode in MINORITY_MODES
            and plan.target_column
            and plan.target_class is not None
        ):
            if plan.target_column not in df.columns:
                raise GeneratorError(
                    f"Target column '{plan.target_column}' "
                    "is not present in the training dataframe."
                )

            condition_column = plan.target_column
            condition_value = self._resolve_target_value(
                df[plan.target_column],
                plan.target_class,
            )

            logger.info(
                "Conditional generation requested: %s=%r; rows=%d",
                condition_column,
                condition_value,
                plan.num_samples,
            )

        try:
            synthetic = generator.generate(
                df=df,
                num_samples=plan.num_samples,
                random_state=plan.random_state,
                epochs=plan.epochs,
                condition_column=condition_column,
                condition_value=condition_value,
            )

        except Exception as exc:
            logger.warning(
                "%s failed: %s",
                plan.generator,
                exc,
            )

            # For conditional generation, fallback is also conditional.
            # This prevents us from returning arbitrary classes.
            try:
                synthetic = self._fallback.generate(
                    df=df,
                    num_samples=plan.num_samples,
                    random_state=plan.random_state,
                    condition_column=condition_column,
                    condition_value=condition_value,
                )
            except Exception as fallback_exc:
                raise GeneratorError(
                    f"{plan.generator} failed and bootstrap fallback "
                    f"also failed: {fallback_exc}"
                ) from fallback_exc

        synthetic = self._align_schema(
            df,
            synthetic,
        )

        if condition_column is not None:
            synthetic = self._validate_condition(
                synthetic,
                condition_column,
                condition_value,
            )

        if len(synthetic) != plan.num_samples:
            raise GeneratorError(
                f"Requested {plan.num_samples} synthetic rows but "
                f"received {len(synthetic)}."
            )

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            synthetic.to_csv(
                path,
                index=False,
            )

            logger.info(
                "Saved synthetic dataset to %s",
                path,
            )

        return synthetic.reset_index(drop=True)

    @staticmethod
    def _resolve_target_value(
        series: pd.Series,
        requested_value: object,
    ) -> object:
        """
        Resolve planner's target value against the actual dtype/value
        representation in the dataframe.
        """

        # Exact match first.
        matches = series[series == requested_value]

        if not matches.empty:
            return matches.iloc[0]

        # String representation fallback.
        string_matches = series[
            series.astype(str) == str(requested_value)
        ]

        if not string_matches.empty:
            return string_matches.iloc[0]

        raise GeneratorError(
            f"Target class {requested_value!r} does not exist "
            f"in the training data."
        )

    @staticmethod
    def _align_schema(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> pd.DataFrame:

        missing = [
            column
            for column in original.columns
            if column not in synthetic.columns
        ]

        if missing:
            raise GeneratorError(
                f"Synthetic data is missing columns: {missing}"
            )

        out = synthetic[
            original.columns
        ].copy()

        # Restore numeric columns.
        for column in original.columns:
            if pd.api.types.is_numeric_dtype(
                original[column]
            ):
                out[column] = pd.to_numeric(
                    out[column],
                    errors="coerce",
                )

        return out.reset_index(drop=True)

    @staticmethod
    def _validate_condition(
        synthetic: pd.DataFrame,
        condition_column: str,
        condition_value: object,
    ) -> pd.DataFrame:

        if condition_column not in synthetic.columns:
            raise GeneratorError(
                f"Conditional column '{condition_column}' "
                "missing from synthetic output."
            )

        matches = (
            synthetic[condition_column].astype(str)
            == str(condition_value)
        )

        if not matches.all():
            actual_values = (
                synthetic[condition_column]
                .value_counts(dropna=False)
                .to_dict()
            )

            raise GeneratorError(
                "Conditional generation returned rows outside "
                f"the requested condition "
                f"{condition_column}={condition_value!r}. "
                f"Observed values: {actual_values}"
            )

        return synthetic