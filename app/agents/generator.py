"""Selects and runs a tabular synthesizer from a GenerationPlan."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.generators.registry import create_generator, require_generator
from app.config import get_settings
from app.generators.runtime import MODEL_LOCK, resolve_device
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
        # Retained as an injection seam for tests and local integrations.
        self._registry = {}
        self.last_metadata: dict = {}

    def generate(
        self, df: pd.DataFrame, plan: GenerationPlan,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        # Hold the lock through model creation, fit, sample, metadata capture and
        # adapter release, so background jobs cannot fit two GPU models at once.
        with MODEL_LOCK:
            return self._generate(df, plan, output_path)

    def _generate(
        self,
        df: pd.DataFrame,
        plan: GenerationPlan,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:

        self.last_metadata = {}
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

        registration = require_generator(plan.generator, get_settings().enable_experimental_generators)
        if "tabular" not in registration.capabilities.modalities:
            raise GeneratorError(f"{plan.generator} does not support tabular inputs.")
        if registration.capabilities.execution == "in_process":
            resolve_device(plan.generator_config.device, registration.capabilities.supports_gpu,
                           registration.capabilities.supports_cpu)
        generator = self._registry.get(plan.generator) or create_generator(plan.generator, get_settings().enable_experimental_generators)

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

        if condition_column and not registration.capabilities.supports_conditional_sampling:
            raise GeneratorError(f"{plan.generator} does not support conditional sampling.")
        resolved_config = plan.generator_config.model_copy(update={
            "random_state": plan.random_state,
            "epochs": plan.epochs if plan.epochs is not None else plan.generator_config.epochs,
        })
        try:
            # All registered production adapters use the lifecycle contract. Legacy injected
            # test doubles retain the original convenience signature.
            if hasattr(generator, "fit") and hasattr(generator, "sample"):
                synthetic = generator.generate(df, plan.num_samples, condition_column=condition_column,
                    condition_value=condition_value, config=resolved_config)
                self.last_metadata = generator.get_metadata()
            else:
                synthetic = generator.generate(df=df, num_samples=plan.num_samples,
                    random_state=plan.random_state, epochs=plan.epochs,
                    condition_column=condition_column, condition_value=condition_value)
                self.last_metadata = {"generator": plan.generator, "injected_adapter": True}
        except Exception as exc:
            raise GeneratorError(f"{plan.generator} failed; no substitute model was used: {exc}") from exc

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
