"""tests/config/test_schema.py — Unit tests for Pydantic v2 config schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.schema import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LSTMModelConfig,
    RuntimeConfig,
    TrainConfig,
    XGBModelConfig,
)


class TestLSTMModelConfig:
    def test_valid_fields_parse_correctly(self):
        cfg = LSTMModelConfig(hidden_size=256, num_layers=2, dropout=0.3, head_dropout=0.5)
        assert cfg.hidden_size == 256
        assert cfg.num_layers == 2
        assert cfg.arch == "lstm"

    def test_defaults(self):
        cfg = LSTMModelConfig()
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 1
        assert cfg.dropout == pytest.approx(0.318, abs=1e-3)


class TestXGBModelConfig:
    def test_valid_fields_parse_correctly(self):
        cfg = XGBModelConfig(n_estimators=100, max_depth=3, learning_rate=0.1)
        assert cfg.n_estimators == 100
        assert cfg.arch == "xgb"

    def test_defaults(self):
        cfg = XGBModelConfig()
        assert cfg.n_estimators == 513
        assert cfg.max_depth == 4


class TestDiscriminatedUnion:
    def test_discriminated_union_lstm(self):
        exp = ExperimentConfig.model_validate({"model": {"arch": "lstm", "hidden_size": 64}})
        assert isinstance(exp.model, LSTMModelConfig)
        assert exp.model.hidden_size == 64

    def test_discriminated_union_xgb(self):
        exp = ExperimentConfig.model_validate({"model": {"arch": "xgb", "n_estimators": 200}})
        assert isinstance(exp.model, XGBModelConfig)
        assert exp.model.n_estimators == 200

    def test_missing_required_arch_raises(self):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate({"model": {"hidden_size": 128}})

    def test_wrong_type_hidden_size_raises(self):
        with pytest.raises(ValidationError):
            LSTMModelConfig.model_validate({"arch": "lstm", "hidden_size": "not_an_int"})


class TestExperimentConfigDefaults:
    def test_defaults_populate_all_sections(self):
        exp = ExperimentConfig()
        assert exp.name == "unnamed"
        assert isinstance(exp.data, DataConfig)
        assert isinstance(exp.train, TrainConfig)
        assert isinstance(exp.eval, EvalConfig)
        assert isinstance(exp.runtime, RuntimeConfig)

    def test_train_config_seeds_default(self):
        cfg = TrainConfig()
        assert cfg.seeds == [42]

    def test_optimizer_literal_invalid(self):
        with pytest.raises(ValidationError):
            TrainConfig.model_validate({"optimizer": "sgd"})

    def test_device_literal_invalid(self):
        with pytest.raises(ValidationError):
            TrainConfig.model_validate({"device": "tpu"})

    def test_data_labeller_literal_invalid(self):
        with pytest.raises(ValidationError):
            DataConfig.model_validate({"labeller": "invalid_label"})
