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
from src.data.timeframe import H1, M5, Timeframe


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


class TestDataConfigTimeframe:
    """WS-4: DataConfig.timeframe field + timeframe_obj property."""

    def test_default_timeframe_is_h1(self):
        """YAML with no 'timeframe' key must default to 'h1'."""
        cfg = DataConfig()
        assert cfg.timeframe == "h1"

    def test_explicit_5m_token(self):
        """Explicit '5m' token must round-trip through the field."""
        cfg = DataConfig.model_validate({"timeframe": "5m"})
        assert cfg.timeframe == "5m"

    def test_explicit_15m_token(self):
        cfg = DataConfig.model_validate({"timeframe": "15m"})
        assert cfg.timeframe == "15m"

    def test_timeframe_obj_default_resolves_to_H1(self):
        """timeframe_obj property must return the canonical H1 Timeframe."""
        cfg = DataConfig()
        tf = cfg.timeframe_obj
        assert tf is H1
        assert tf.minutes == 60
        assert tf.token == "h1"

    def test_timeframe_obj_5m_resolves(self):
        cfg = DataConfig.model_validate({"timeframe": "5m"})
        tf = cfg.timeframe_obj
        assert tf is M5
        assert tf.minutes == 5

    def test_timeframe_obj_invalid_token_raises(self):
        """An unrecognised token stored on the model must raise ValueError on access."""
        cfg = DataConfig.model_validate({"timeframe": "bad_token"})
        with pytest.raises(ValueError, match="Unknown timeframe token"):
            _ = cfg.timeframe_obj

    def test_experiment_config_no_timeframe_key_defaults_h1(self):
        """ExperimentConfig loaded from dict with no timeframe key → h1 default."""
        exp = ExperimentConfig.model_validate({"name": "test"})
        assert exp.data.timeframe == "h1"
        assert exp.data.timeframe_obj is H1

    def test_timeframe_raw_token_preserved_in_serialisation(self):
        """model_dump must emit the raw string, not the Timeframe object."""
        cfg = DataConfig.model_validate({"timeframe": "5m"})
        d = cfg.model_dump()
        assert d["timeframe"] == "5m"
        assert isinstance(d["timeframe"], str)
