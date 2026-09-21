from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSyntheticGenerator(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        df: pd.DataFrame,
        num_samples: int,
        random_state: int = 42,
        epochs: int | None = None,
    ) -> pd.DataFrame:
        """Train on `df` and return `num_samples` synthetic rows."""
