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
        condition_column: str | None = None,
        condition_value: object | None = None,
    ) -> pd.DataFrame:
        """
        Train on df and return synthetic rows.

        If condition_column and condition_value are provided,
        the generator should attempt conditional sampling.
        """
        raise NotImplementedError