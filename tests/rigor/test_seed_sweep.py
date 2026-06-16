"""tests/rigor/test_seed_sweep.py — Unit tests for src/rigor/seed_sweep.py.

Uses mock training to verify DataFrame shape, column names, and checkpoint-skip logic.
Does NOT train real models — patches the training functions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import load_experiment
from src.rigor.seed_sweep import SeedSweepConfig, _checkpoint_name, run_seed_sweep


# ---------------------------------------------------------------------------
# SeedSweepConfig validation
# ---------------------------------------------------------------------------

def test_seed_sweep_config_defaults() -> None:
    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={"hidden_size": 64},
        seeds=[42, 17],
    )
    assert cfg.loss_type == "weighted_ce"
    assert cfg.focal_gamma == 2.0
    assert isinstance(cfg.output_dir, Path)
    assert isinstance(cfg.checkpoint_dir, Path)


def test_seed_sweep_config_focal() -> None:
    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={},
        seeds=[42],
        loss_type="focal",
        focal_gamma=3.0,
    )
    assert cfg.loss_type == "focal"
    assert cfg.focal_gamma == 3.0


def test_checkpoint_name_weighted_ce() -> None:
    cfg = SeedSweepConfig(model_type="lstm", hyperparams={}, seeds=[42])
    assert _checkpoint_name(cfg, 42) == "lstm_seed42"


def test_checkpoint_name_focal() -> None:
    cfg = SeedSweepConfig(
        model_type="lstm", hyperparams={}, seeds=[42],
        loss_type="focal", focal_gamma=2.0,
    )
    assert _checkpoint_name(cfg, 42) == "lstm_focal_g2_seed42"


def test_checkpoint_name_xgb() -> None:
    cfg = SeedSweepConfig(model_type="xgb", hyperparams={}, seeds=[0])
    assert _checkpoint_name(cfg, 0) == "xgb_seed0"


# ---------------------------------------------------------------------------
# SeedSweepConfig.from_experiment_config adapter
# ---------------------------------------------------------------------------

def test_from_experiment_config_lstm() -> None:
    """from_experiment_config maps lstm_g1.yaml fields correctly."""
    cfg = load_experiment("experiments/lstm_g1.yaml")
    sweep = SeedSweepConfig.from_experiment_config(cfg)
    assert sweep.model_type == "lstm"
    assert sweep.seeds == cfg.train.seeds
    assert sweep.loss_type == cfg.train.loss
    assert sweep.focal_gamma == cfg.train.focal_gamma
    assert sweep.hyperparams["hidden_size"] == cfg.model.hidden_size
    assert sweep.hyperparams["lr"] == cfg.train.lr
    assert sweep.hyperparams["batch_size"] == cfg.train.batch_size
    assert sweep.ablation_no_dropout == cfg.train.ablation_no_dropout
    assert sweep.ablation_no_l2 == cfg.train.ablation_no_l2


def test_from_experiment_config_output_dir_override(tmp_path: Path) -> None:
    """output_dir kwarg overrides cfg.runtime.output_dir."""
    cfg = load_experiment("experiments/lstm_g1.yaml")
    sweep = SeedSweepConfig.from_experiment_config(cfg, output_dir=tmp_path / "smoke")
    assert sweep.output_dir == tmp_path / "smoke"


def test_from_experiment_config_xgb() -> None:
    """from_experiment_config maps xgboost_g1.yaml arch correctly."""
    cfg = load_experiment("experiments/xgboost_g1.yaml")
    sweep = SeedSweepConfig.from_experiment_config(cfg)
    assert sweep.model_type == "xgb"
    assert sweep.hyperparams["n_estimators"] == cfg.model.n_estimators


# ---------------------------------------------------------------------------
# run_seed_sweep — mock training
# ---------------------------------------------------------------------------

def _make_fake_result(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "test_macro_f1": float(rng.uniform(0.7, 0.9)),
        "test_per_class_f1": [
            float(rng.uniform(0.8, 0.95)),
            float(rng.uniform(0.5, 0.75)),
            float(rng.uniform(0.5, 0.75)),
        ],
        "y_true": np.array([0, 1, 2, 0, 1], dtype=np.int64),
        "y_pred": np.array([0, 1, 2, 0, 0], dtype=np.int64),
    }


def test_run_seed_sweep_returns_correct_shape(tmp_path: Path) -> None:
    seeds = [42, 17, 0]
    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={"hidden_size": 64, "lr": 1e-3},
        seeds=seeds,
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    with patch("src.rigor.seed_sweep._train_lstm") as mock_train:
        mock_train.side_effect = lambda config, seed, ckpt_path, meta_path: (
            _write_fake_meta(meta_path, seed) or _make_fake_result(seed)
        )
        df = run_seed_sweep(cfg)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(seeds)
    assert list(df.columns) == ["seed", "macro_f1", "none_f1", "bull_f1", "bear_f1"]


def test_run_seed_sweep_checkpoint_skip(tmp_path: Path) -> None:
    """If meta.json already exists with test_macro_f1, seed should be skipped."""
    seeds = [42]
    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={},
        seeds=seeds,
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    # Pre-write a valid meta for seed 42 — new flat scheme: lstm_h1_spy
    ckpt_dir = tmp_path / "checkpoints" / "lstm_h1_spy"
    ckpt_dir.mkdir(parents=True)
    meta_path = ckpt_dir / "lstm_seed42.meta.json"
    meta = {
        "test_macro_f1": 0.85,
        "test_per_class_f1": [0.90, 0.70, 0.72],
    }
    meta_path.write_text(json.dumps(meta))

    with patch("src.rigor.seed_sweep._train_lstm") as mock_train:
        df = run_seed_sweep(cfg)
        mock_train.assert_not_called()  # should be skipped

    assert len(df) == 1
    assert df.iloc[0]["macro_f1"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_meta(meta_path: Path, seed: int) -> None:
    """Write a minimal meta JSON so checkpoint-skip logic triggers on re-run."""
    rng = np.random.default_rng(seed)
    meta = {
        "test_macro_f1": float(rng.uniform(0.7, 0.9)),
        "test_per_class_f1": [0.88, 0.65, 0.66],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# H1: _train_lstm honors swept window_size (was hardcoded 60)
# ---------------------------------------------------------------------------

def test_train_lstm_honors_window_size_from_hyperparams(tmp_path: Path) -> None:
    """_train_lstm must build SMCWindowDataset with the swept window_size, not 60.

    Patches SMCWindowDataset to a sentinel that captures its window_size kwarg
    and short-circuits the rest of training. Mirrors the accessor used by
    _train_cnn_lstm / _train_torch_generic: int(hp.get("window_size", 60)).
    """
    from src.rigor.seed_sweep import SeedSweepConfig, _train_lstm

    captured: list[int] = []

    class _Sentinel(Exception):
        pass

    def _fake_dataset(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.append(kwargs["window_size"])
        raise _Sentinel  # stop before the (heavy) training loop

    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={"window_size": 40},
        seeds=[0],
        data_dir=tmp_path,
    )

    import pandas as _pd

    with patch(
        "src.rigor.seed_sweep._load_splits",
        return_value=(_pd.DataFrame(), _pd.DataFrame(), _pd.DataFrame()),
    ), patch("src.data.window.SMCWindowDataset", side_effect=_fake_dataset):
        with pytest.raises(_Sentinel):
            _train_lstm(cfg, seed=0, ckpt_path=tmp_path / "c.pt", meta_path=tmp_path / "m.json")

    assert captured, "SMCWindowDataset was never constructed"
    assert all(ws == 40 for ws in captured), (
        f"_train_lstm ignored window_size; built datasets with {captured} (expected all 40)"
    )


# ---------------------------------------------------------------------------
# train_stride — configurable train stride, val/test stay stride=1
# ---------------------------------------------------------------------------

def test_seed_sweep_config_train_stride_default() -> None:
    """SeedSweepConfig.train_stride defaults to 1 (H1 backward-compat)."""
    cfg = SeedSweepConfig(model_type="lstm", hyperparams={}, seeds=[42])
    assert cfg.train_stride == 1


def test_from_experiment_config_train_stride_from_data_stride() -> None:
    """from_experiment_config populates train_stride from cfg.data.stride."""
    cfg = load_experiment("experiments/lstm_5m.yaml")
    sweep = SeedSweepConfig.from_experiment_config(cfg)
    # lstm_5m.yaml sets data.stride: 9
    assert sweep.train_stride == 9


def test_from_experiment_config_train_stride_h1_default() -> None:
    """H1 YAML (no data.stride key) → train_stride=1 (DataConfig default)."""
    cfg = load_experiment("experiments/lstm_g1.yaml")
    sweep = SeedSweepConfig.from_experiment_config(cfg)
    assert sweep.train_stride == 1


def test_from_experiment_config_15m_stride() -> None:
    """15m YAML sets data.stride: 3 → train_stride=3."""
    cfg = load_experiment("experiments/lstm_15m.yaml")
    sweep = SeedSweepConfig.from_experiment_config(cfg)
    assert sweep.train_stride == 3


def test_train_lstm_uses_train_stride_for_train_only(tmp_path: Path) -> None:
    """_train_lstm builds TRAIN dataset with config.train_stride, VAL/TEST with stride=1.

    Patches SMCWindowDataset, captures stride kwarg per call (train=first, val=second,
    test=third), then raises to short-circuit the loop. Asserts train gets train_stride
    and val/test get 1.
    """
    from src.rigor.seed_sweep import _train_lstm

    stride_calls: list[int] = []

    class _Sentinel(Exception):
        pass

    def _fake_dataset(*args, **kwargs):  # noqa: ANN002, ANN003
        stride_calls.append(kwargs.get("stride", 1))
        if len(stride_calls) >= 3:
            raise _Sentinel

    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={"window_size": 60},
        seeds=[0],
        data_dir=tmp_path,
        train_stride=5,
    )

    import pandas as _pd

    with patch(
        "src.rigor.seed_sweep._load_splits",
        return_value=(_pd.DataFrame(), _pd.DataFrame(), _pd.DataFrame()),
    ), patch("src.data.window.SMCWindowDataset", side_effect=_fake_dataset):
        with pytest.raises(_Sentinel):
            _train_lstm(cfg, seed=0, ckpt_path=tmp_path / "c.pt", meta_path=tmp_path / "m.json")

    assert len(stride_calls) == 3, f"Expected 3 dataset constructions, got {len(stride_calls)}"
    assert stride_calls[0] == 5, f"TRAIN stride should be train_stride=5, got {stride_calls[0]}"
    assert stride_calls[1] == 1, f"VAL stride should be 1, got {stride_calls[1]}"
    assert stride_calls[2] == 1, f"TEST stride should be 1, got {stride_calls[2]}"


def test_5m_yaml_loads_with_stride_9() -> None:
    """cnn_lstm_5m.yaml parses cleanly and has data.stride=9."""
    cfg = load_experiment("experiments/cnn_lstm_5m.yaml")
    assert cfg.data.stride == 9


def test_15m_yaml_loads_with_stride_3() -> None:
    """transformer_15m.yaml parses cleanly and has data.stride=3."""
    cfg = load_experiment("experiments/transformer_15m.yaml")
    assert cfg.data.stride == 3
