from __future__ import annotations

import pandas as pd

from app.generators.base import BaseSyntheticGenerator
from app.generators.sdv_common import build_metadata, sample_sdv
from app.utils.exceptions import GeneratorError
from app.utils.logging_config import get_logger

logger = get_logger("Generator")


class CTGANGenerator(BaseSyntheticGenerator):
    name = "ctgan"

    def generate(
        self,
        df: pd.DataFrame,
        num_samples: int,
        random_state: int = 42,
        epochs: int | None = None,
    ) -> pd.DataFrame:
        epochs = 10 if epochs is None else epochs
        logger.info("Training CTGAN (%s epochs)", epochs)
        try:
            from sdv.single_table import CTGANSynthesizer
        except Exception as exc:
            raise GeneratorError(f"CTGAN unavailable: {exc}") from exc
        metadata = build_metadata(df)
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=epochs,
            verbose=False,
        )
        try:
            synthesizer.fit(df)
        except Exception as exc:
            raise GeneratorError(f"CTGAN training failed: {exc}") from exc
        synthetic = sample_sdv(synthesizer, num_samples)
        logger.info("Generated %s rows", len(synthetic))
        return synthetic
