"""schema.py — Pydantic v2 config models for experiment configuration.

Every experiment reads one YAML file. All HP, data, model, and runtime settings
are captured here. New model = new ModelConfig subclass + new YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    data_dir: Path = Path("data/processed")
    labeller: Literal["fvg", "fvg_valid"] = "fvg_valid"
    splits: str = "default"
    window_size: int = 60
    stride: int = 1
    drop_cross_session: bool = False


class LSTMModelConfig(BaseModel):
    arch: Literal["lstm"] = "lstm"
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.318
    head_dropout: float = 0.526


class XGBModelConfig(BaseModel):
    arch: Literal["xgb"] = "xgb"
    n_estimators: int = 513
    max_depth: int = 4
    learning_rate: float = 0.1311
    min_child_weight: int = 1
    subsample: float = 0.826
    colsample_bytree: float = 0.725


class CNNLSTMModelConfig(BaseModel):
    arch: Literal["cnn_lstm"] = "cnn_lstm"
    n_conv_layers: int = 2
    conv_filters: int = 32
    kernel_size: int = 3
    use_pool: bool = False
    pool_type: str = "max"       # "max" | "avg" — only used if use_pool=True
    lstm_hidden: int = 64
    lstm_layers: int = 1
    dropout: float = 0.318       # between LSTM layers (active only if lstm_layers > 1)
    head_dropout: float = 0.526


ModelConfig = Annotated[
    LSTMModelConfig | XGBModelConfig | CNNLSTMModelConfig,
    Field(discriminator="arch"),
]


class TrainConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [42])
    batch_size: int = 32
    batch_eval: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw"] = "adam"
    scheduler: Literal["none", "onecycle"] = "none"
    max_epochs: int = 100
    patience: int = 15
    ema_alpha: float = 0.3
    max_grad_norm: float = 1.0
    loss: Literal["weighted_ce", "focal"] = "weighted_ce"
    focal_gamma: float = 2.0
    device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"
    ablation_no_dropout: bool = False
    ablation_no_l2: bool = False


class EvalConfig(BaseModel):
    lookahead_bars: int = 20
    bootstrap_n_iter: int = 1000
    bootstrap_block_size: int = 60


class RuntimeConfig(BaseModel):
    output_dir: Path = Path("reports/rigor")
    checkpoint_dir: Path = Path("checkpoints")
    n_jobs: int = -1


class ExperimentConfig(BaseModel):
    name: str = "unnamed"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=LSTMModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict:
        """Return nested dict (serialisable)."""
        return self.model_dump(mode="json")
