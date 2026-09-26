from __future__ import annotations

import pandas as pd

from app.generators.base import BaseSyntheticGenerator
from app.generators.sdv_common import (
    build_metadata,
    fit_sdv,
    sample_sdv,
    validate_generated_output,
)
from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")


class TVAEGenerator(BaseSyntheticGenerator):
    name = "tvae"

    def generate(
        self,
        df: pd.DataFrame,
        num_samples: int,
        random_state: int = 42,
        epochs: int | None = None,
        condition_column: str | None = None,
        condition_value: object | None = None,
    ) -> pd.DataFrame:

        if len(df) < 10:
            raise GeneratorError(
                "TVAE requires at least 10 training rows."
            )

        epochs = 100 if epochs is None else max(1, int(epochs))

        logger.info(
            "Training TVAE: rows=%d, epochs=%d, requested_samples=%d",
            len(df),
            epochs,
            num_samples,
        )

        try:
            from sdv.single_table import TVAESynthesizer
        except Exception as exc:
            raise GeneratorError(
                f"TVAE unavailable: {exc}"
            ) from exc

        metadata = build_metadata(df)

        try:
            synthesizer = TVAESynthesizer(
                metadata,
                epochs=epochs,
                verbose=False,
            )
        except Exception as exc:
            raise GeneratorError(
                f"Could not initialize TVAE: {exc}"
            ) from exc

        fit_sdv(
            synthesizer,
            df,
            random_state=random_state,
        )

        synthetic = sample_sdv(
            synthesizer,
            num_samples=num_samples,
            condition_column=condition_column,
            condition_value=condition_value,
        )

        synthetic = validate_generated_output(
            original=df,
            synthetic=synthetic,
            expected_rows=num_samples,
        )

        logger.info(
            "TVAE generated %d rows",
            len(synthetic),
        )

        return synthetic