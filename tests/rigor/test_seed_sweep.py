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

    # Pre-write a valid meta for seed 42
    ckpt_dir = tmp_path / "checkpoints" / "lstm"
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
