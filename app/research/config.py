from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.generators.base import GeneratorConfig
from app.timeseries.data import TimeSeriesSchema
from app.timeseries.generators import SequenceConfig


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    modality: Literal["tabular", "time_series"] = "tabular"
    generator: str = "gaussian_copula"
    target: str | None = None
    sensitive_column: str | None = None
    num_samples: int = Field(default=100, ge=2, le=100000)
    seed: int = Field(default=42, ge=0, le=2**32-2)
    split_seed: int = Field(default=42, ge=0, le=2**32-2)
    generator_config: GeneratorConfig = Field(default_factory=GeneratorConfig)
    sequence_schema: TimeSeriesSchema | None = None
    sequence_config: SequenceConfig = Field(default_factory=SequenceConfig)
    forecast_target: str | None = None
    forecast_horizon: int = Field(default=1, ge=1)
    selection: Literal["explicit", "rule", "adaptive"] = "explicit"
    candidates: list[str] = Field(default_factory=list, max_length=8)
    ablation: str | None = None
    allow_experimental: bool = False
    cost_weight: float = Field(default=0.05, ge=0, le=1)
    temporal_max_ks: float = Field(default=0.4, ge=0, le=1)
    temporal_max_acf_error: float = Field(default=0.3, ge=0, le=2)

    @model_validator(mode="after")
    def modalities(self):
        if self.modality == "time_series" and self.sequence_schema is None:
            raise ValueError("Time-series experiments require an explicit sequence_schema.")
        if self.forecast_target is not None and (self.sequence_schema is None or self.forecast_target not in self.sequence_schema.features):
            raise ValueError("forecast_target must be a numerical sequence feature.")
        if self.selection == "adaptive" and not self.candidates:
            raise ValueError("Adaptive selection requires an explicit bounded candidates list.")
        return self
