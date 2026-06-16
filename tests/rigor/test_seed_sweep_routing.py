"""Regression tests for run_seed_sweep arch routing.

Bug (06-Jun-26): transformer/xlstm fell through the dispatch `else` branch to
_train_xgb, whose in-process xgboost.fit segfaults on macOS arm64 after torch
import. These tests pin the routing WITHOUT running real training (the per-arch
trainer is stubbed) so the suite stays fast.
"""
from __future__ import annotations

import numpy as np
import pytest

import src.rigor.seed_sweep as ss
from src.rigor.seed_sweep import SeedSweepConfig, run_seed_sweep


def _cfg(model_type: str, tmp_path) -> SeedSweepConfig:
    return SeedSweepConfig(
        model_type=model_type,
        hyperparams={"batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4},
        seeds=[42],
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",  # empty → no cache-skip
        data_dir=tmp_path / "data",
    )


def _fake_result() -> dict:
    return {
        "test_macro_f1": 0.5,
        "test_per_class_f1": [0.9, 0.3, 0.3],
        "y_true": np.array([0, 1, 2], dtype=np.int64),
        "y_pred": np.array([0, 1, 0], dtype=np.int64),
    }


@pytest.mark.parametrize("arch", ["transformer", "xlstm"])
def test_new_archs_route_to_torch_generic_not_xgb(arch, tmp_path, monkeypatch):
    """transformer/xlstm must hit _train_torch_generic, never _train_xgb."""
    calls = {"generic": 0, "xgb": 0}

    def fake_generic(config, seed, ckpt_path, meta_path):
        calls["generic"] += 1
        return _fake_result()

    def fake_xgb(config, seed, ckpt_path, meta_path):
        calls["xgb"] += 1
        return _fake_result()

    monkeypatch.setattr(ss, "_train_torch_generic", fake_generic)
    monkeypatch.setattr(ss, "_train_xgb", fake_xgb)

    df = run_seed_sweep(_cfg(arch, tmp_path))

    assert calls["generic"] == 1, f"{arch} did not route to _train_torch_generic"
    assert calls["xgb"] == 0, f"{arch} leaked into the XGB path (segfault risk)"
    assert df.iloc[0]["macro_f1"] == 0.5


def test_unknown_arch_raises_not_silent_xgb(tmp_path):
    """An unknown arch must raise, not silently fall through to XGB."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        run_seed_sweep(_cfg("bogus_arch", tmp_path))


def test_get_device_is_cpu():
    """All torch training is CPU-only (MPS gradient bug)."""
    assert ss._get_device().type == "cpu"


def test_save_predictions_rejects_length_mismatch(tmp_path):
    """Guard: mismatched y_true/y_pred must raise, not silently corrupt bootstrap CI."""
    cfg = _cfg("transformer", tmp_path)
    bad = {
        "test_macro_f1": 0.5,
        "test_per_class_f1": [0.9, 0.3, 0.3],
        "y_true": np.array([0, 1, 2, 0], dtype=np.int64),   # len 4
        "y_pred": np.array([0, 1, 2], dtype=np.int64),       # len 3
    }
    with pytest.raises(ValueError, match="pred length mismatch"):
        ss._save_predictions(cfg, 42, bad)


def test_cached_train_py_meta_format_not_zeroed(tmp_path):
    """Cache loader must read train.py-format metas (test_bull_f1/test_bear_f1),
    not silently zero minority-class columns by only checking test_per_class_f1."""
    import json
    arch = "transformer"
    ckpt_dir = tmp_path / "ckpt" / "transformer_h1_spy"
    ckpt_dir.mkdir(parents=True)
    # train.py-format meta: separate keys, NO test_per_class_f1 array
    (ckpt_dir / f"{arch}_seed42.meta.json").write_text(json.dumps({
        "test_macro_f1": 0.55,
        "test_none_f1": 0.98, "test_bull_f1": 0.42, "test_bear_f1": 0.25,
    }))
    df = run_seed_sweep(_cfg(arch, tmp_path))
    row = df.iloc[0]
    assert row["bull_f1"] == 0.42 and row["bear_f1"] == 0.25, "minority F1 silently zeroed from train.py-format cache"
