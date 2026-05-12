"""src/config — Experiment configuration module.

Public API:
    load_experiment(path, overrides)   — load YAML → ExperimentConfig
    experiment_from_json(path)         — load legacy best_hp_*.json → ExperimentConfig
    MODELS, LOSSES                     — populated registries
    register_model, register_loss      — decorators for new models/losses
"""

from src.config.schema import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LSTMModelConfig,
    ModelConfig,
    RuntimeConfig,
    TrainConfig,
    XGBModelConfig,
)
from src.config.loader import experiment_from_json, load_experiment, parse_set_args
from src.config.registry import LOSSES, MODELS, register_loss, register_model

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "LSTMModelConfig",
    "XGBModelConfig",
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "RuntimeConfig",
    "load_experiment",
    "experiment_from_json",
    "parse_set_args",
    "MODELS",
    "LOSSES",
    "register_model",
    "register_loss",
]
