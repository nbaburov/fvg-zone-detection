"""tests/config/test_loader.py — Unit tests for YAML/JSON config loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.config.loader import experiment_from_json, load_experiment
from src.config.schema import LSTMModelConfig, XGBModelConfig


REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
G1_DIR = REPO / "reports" / "rigor" / "2026-05-13" / "G1"


class TestLoadExperiment:
    def test_load_experiment_lstm_g1(self):
        cfg = load_experiment(EXPERIMENTS / "lstm_g1.yaml")
        assert isinstance(cfg.model, LSTMModelConfig)
        assert cfg.model.hidden_size == 128
        assert cfg.model.num_layers == 1
        assert cfg.train.optimizer == "adam"
        assert cfg.train.scheduler == "none"
        assert cfg.train.lr == pytest.approx(5.3015e-4, rel=1e-3)

    def test_load_experiment_xgb_g1(self):
        cfg = load_experiment(EXPERIMENTS / "xgboost_g1.yaml")
        assert isinstance(cfg.model, XGBModelConfig)
        assert cfg.model.n_estimators == 513
        assert cfg.model.max_depth == 4

    def test_override_train_seeds(self):
        cfg = load_experiment(EXPERIMENTS / "lstm_g1.yaml", overrides={"train.seeds": [99]})
        assert cfg.train.seeds == [99]

    def test_override_model_hidden_size(self):
        cfg = load_experiment(EXPERIMENTS / "lstm_g1.yaml", overrides={"model.hidden_size": 64})
        assert cfg.model.hidden_size == 64

    def test_deep_merge_base_yaml(self):
        # Fields not in lstm_g1.yaml should be filled from _base.yaml
        cfg = load_experiment(EXPERIMENTS / "lstm_g1.yaml")
        assert cfg.eval.lookahead_bars == 20
        assert cfg.eval.bootstrap_n_iter == 1000
        assert cfg.runtime.n_jobs == -1

    def test_nested_override_path(self):
        cfg = load_experiment(EXPERIMENTS / "lstm_g1.yaml", overrides={"data.window_size": 30})
        assert cfg.data.window_size == 30

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_experiment("experiments/nonexistent_config.yaml")


class TestExperimentFromJson:
    def test_experiment_from_json_lstm(self):
        cfg = experiment_from_json(G1_DIR / "best_hp_lstm.json")
        assert isinstance(cfg.model, LSTMModelConfig)
        assert cfg.model.hidden_size == 128
        assert cfg.model.dropout == pytest.approx(0.31753, rel=1e-3)

    def test_experiment_from_json_xgb(self):
        cfg = experiment_from_json(G1_DIR / "best_hp_xgb.json")
        assert isinstance(cfg.model, XGBModelConfig)
        assert cfg.model.n_estimators == 513

    def test_json_metadata_keys_stripped(self):
        cfg = experiment_from_json(G1_DIR / "best_hp_lstm.json")
        # Metadata keys must not appear as model fields
        model_fields = set(type(cfg.model).model_fields.keys())
        assert "best_value" not in model_fields
        assert "trial_number" not in model_fields
        assert "study_name" not in model_fields

    def test_json_metadata_keys_stripped_xgb(self):
        cfg = experiment_from_json(G1_DIR / "best_hp_xgb.json")
        model_fields = set(type(cfg.model).model_fields.keys())
        assert "val_macro_f1" not in model_fields
        assert "n_trials_completed" not in model_fields

    def test_missing_json_file_raises(self):
        with pytest.raises(FileNotFoundError):
            experiment_from_json("reports/rigor/nonexistent.json")

    def test_arch_inferred_from_filename(self):
        # Write a minimal JSON with xgb in name
        data = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
                 "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.7}
        with tempfile.NamedTemporaryFile(suffix="_xgb_test.json", mode="w", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            cfg = experiment_from_json(tmp_path)
            assert isinstance(cfg.model, XGBModelConfig)
        finally:
            tmp_path.unlink(missing_ok=True)
