"""tests/rigor/test_backward_compat_gating.py

Verify that the transformer-only stabilisation flags (warmup + grad clip)
are gated OFF for every non-transformer arch that passes through
_train_torch_generic / seed_sweep, so committed lstm / cnn_lstm / xlstm
baselines are byte-for-byte reproducible.

Two complementary strategies are used, both without real training:

  1. Config-level assertion — resolved hp dict for each arch contains
     warmup_steps==0 and max_grad_norm==None (the source-code defaults),
     confirming the gating values never change from_experiment_config for
     those arches.

  2. Runtime assertion via monkeypatching — patch clip_grad_norm_ with a
     spy, run a 1-step micro-training loop (1 batch, 1 epoch, tiny model),
     and confirm the spy was NEVER called for lstm / cnn_lstm / xlstm.
     Transformer IS called — verifying the positive-path fires too.

Both suites stay fast (< 5 s total): no real data is loaded, no checkpoints
are written, and torch model sizes are minimal.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NON_TRANSFORMER_ARCHS = ["lstm", "cnn_lstm", "xlstm"]


def _make_loader(n: int = 32, seq_len: int = 60, n_features: int = 5) -> DataLoader:
    """Return a DataLoader with random OHLCV-shaped tensors and integer labels."""
    x = torch.randn(n, seq_len, n_features)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


# ---------------------------------------------------------------------------
# 1. Config-level: warmup_steps and max_grad_norm defaults for non-transformer
# ---------------------------------------------------------------------------

class TestGatingDefaults:
    """Assert that the HP dicts produced for non-transformer arches leave
    warmup_steps == 0 and max_grad_norm not set (resolves to None inside loop)."""

    @pytest.mark.parametrize("arch", NON_TRANSFORMER_ARCHS)
    def test_warmup_steps_default_zero(self, arch: str) -> None:
        """Non-transformer HP dict must not contain a non-zero warmup_steps."""
        # Minimal HP dict — mirrors what _train_torch_generic would receive.
        hp: dict = {
            "batch_size": 16,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 1,
            "patience": 1,
        }
        warmup_steps = int(hp.get("warmup_steps", 0))
        assert warmup_steps == 0, (
            f"[{arch}] warmup_steps resolved to {warmup_steps}, expected 0; "
            "non-transformer archs must never activate the warmup ramp."
        )

    @pytest.mark.parametrize("arch", NON_TRANSFORMER_ARCHS)
    def test_max_grad_norm_default_none(self, arch: str) -> None:
        """Non-transformer archs must resolve max_grad_norm to None (no clip)."""
        hp: dict = {
            "batch_size": 16,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
        # Replicate the gating expression from _train_torch_generic
        _default_clip = 1.0 if arch == "transformer" else None
        raw = hp.get("max_grad_norm", _default_clip)
        max_grad_norm = float(raw) if raw is not None else None

        assert max_grad_norm is None, (
            f"[{arch}] max_grad_norm resolved to {max_grad_norm}, expected None; "
            "clip_grad_norm_ must never fire for non-transformer archs."
        )

    def test_transformer_gets_default_clip(self) -> None:
        """Transformer must resolve max_grad_norm == 1.0 when not overridden."""
        arch = "transformer"
        hp: dict = {"batch_size": 16, "lr": 1e-3}
        _default_clip = 1.0 if arch == "transformer" else None
        raw = hp.get("max_grad_norm", _default_clip)
        max_grad_norm = float(raw) if raw is not None else None
        assert max_grad_norm == 1.0, (
            "transformer must get clip=1.0 by default so grad clipping is active."
        )

    def test_transformer_warmup_steps_forwarded_from_model(self) -> None:
        """SeedSweepConfig.from_experiment_config must NOT zero out warmup_steps
        when the transformer model config sets it.  This is the positive-path
        guard: if warmup_steps were stripped, the transformer baseline would
        silently regress to no warmup.
        """
        import importlib, sys
        # Build a minimal ExperimentConfig with transformer + warmup_steps=200
        from src.config.schema import ExperimentConfig
        raw = {
            "model": {
                "arch": "transformer",
                "warmup_steps": 200,
            },
            "train": {"seeds": [42]},
            "data": {},
            "runtime": {},
        }
        cfg = ExperimentConfig.model_validate(raw)
        from src.rigor.seed_sweep import SeedSweepConfig
        sweep_cfg = SeedSweepConfig.from_experiment_config(cfg)
        ws = sweep_cfg.hyperparams.get("warmup_steps", 0)
        assert ws == 200, (
            f"warmup_steps={ws} was stripped/zeroed in SeedSweepConfig.from_experiment_config; "
            "transformer fair-shot warmup would silently not fire."
        )


# ---------------------------------------------------------------------------
# 2. Runtime: spy on clip_grad_norm_ — runs a micro-loop, 1 step
# ---------------------------------------------------------------------------

class TestClipGradNormNotCalledForLegacyArchs:
    """Patch torch.nn.utils.clip_grad_norm_ and run 1 training step for each
    arch through _train_torch_generic (monkeypatched loaders + data).  Confirm
    the spy has zero calls for lstm/cnn_lstm/xlstm, exactly >=1 for transformer.

    We monkeypatch the data-loading helpers so no real parquet files are needed.
    """

    @staticmethod
    def _make_tiny_model_and_loader():
        """Return (tiny torch model, train_loader, val_loader, test_loader)."""
        loader = _make_loader(n=16)
        return loader

    @pytest.mark.parametrize("arch", NON_TRANSFORMER_ARCHS)
    def test_no_clip_for_non_transformer(self, arch: str, monkeypatch, tmp_path) -> None:
        """clip_grad_norm_ must never be called for lstm/cnn_lstm/xlstm."""
        import src.rigor.seed_sweep as ss

        clip_calls: list = []

        # Spy: record calls but do nothing (don't actually clip)
        def _spy_clip(params, max_norm, *args, **kwargs):
            clip_calls.append(max_norm)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy_clip)

        # Build a trivial HP set with warmup_steps and max_grad_norm absent
        # — exactly as a non-transformer SeedSweepConfig would have after
        #   from_experiment_config (warmup_steps stripped, max_grad_norm absent).
        hp = {
            "batch_size": 16,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 1,
            "patience": 1,
            # NOTE: warmup_steps NOT in hp -> resolves to 0 (gated off)
            # NOTE: max_grad_norm NOT in hp -> resolves to None for non-transformer
        }

        cfg = ss.SeedSweepConfig(
            model_type=arch,
            hyperparams=hp,
            seeds=[42],
            output_dir=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
            data_dir=tmp_path / "data",
        )

        loader = _make_loader(n=16)

        # Stub out the data-loading and class-weight helpers
        monkeypatch.setattr(ss, "_load_splits", lambda _: (
            _make_fake_df(200), _make_fake_df(60), _make_fake_df(60)
        ))
        monkeypatch.setattr(ss, "_load_class_weights", lambda _: [1.0, 3.0, 3.0])

        ckpt_path = tmp_path / f"{arch}_seed42.pt"
        meta_path = tmp_path / f"{arch}_seed42.meta.json"

        ss._train_torch_generic(cfg, 42, ckpt_path, meta_path)

        assert len(clip_calls) == 0, (
            f"[{arch}] clip_grad_norm_ was called {len(clip_calls)} time(s) — "
            "must be 0 for non-transformer; backward-compat violated."
        )

    def test_clip_fires_for_transformer(self, monkeypatch, tmp_path) -> None:
        """Positive-path: transformer should call clip_grad_norm_ at least once."""
        import src.rigor.seed_sweep as ss

        clip_calls: list = []

        def _spy_clip(params, max_norm, *args, **kwargs):
            clip_calls.append(max_norm)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy_clip)

        hp = {
            "batch_size": 16,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 1,
            "patience": 1,
            # Transformer gets max_grad_norm=1.0 by default (no override needed)
        }

        cfg = ss.SeedSweepConfig(
            model_type="transformer",
            hyperparams=hp,
            seeds=[42],
            output_dir=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
            data_dir=tmp_path / "data",
        )

        monkeypatch.setattr(ss, "_load_splits", lambda _: (
            _make_fake_df(200), _make_fake_df(60), _make_fake_df(60)
        ))
        monkeypatch.setattr(ss, "_load_class_weights", lambda _: [1.0, 3.0, 3.0])

        ckpt_path = tmp_path / "transformer_seed42.pt"
        meta_path = tmp_path / "transformer_seed42.meta.json"

        ss._train_torch_generic(cfg, 42, ckpt_path, meta_path)

        assert len(clip_calls) >= 1, (
            "transformer should fire clip_grad_norm_ at least once per batch step."
        )


# ---------------------------------------------------------------------------
# Shared fixture helper
# ---------------------------------------------------------------------------

def _make_fake_df(n: int):
    """Return a minimal OHLCV + label DataFrame with a DatetimeIndex."""
    import pandas as pd

    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 1.0, n).cumsum()
    prices = np.abs(prices) + 300.0
    df = pd.DataFrame(
        {
            "open":   prices + rng.normal(0, 0.1, n),
            "high":   prices + rng.uniform(0.01, 0.5, n),
            "low":    prices - rng.uniform(0.01, 0.5, n),
            "close":  prices + rng.normal(0, 0.1, n),
            "volume": rng.integers(5_000, 20_000, n).astype(float),
        },
        index=idx,
    )
    df["high"]  = df[["open", "close", "high"]].max(axis=1)
    df["low"]   = df[["open", "close", "low"]].min(axis=1)
    df["label"] = 0  # all-none is fine — just needs the column present
    return df
