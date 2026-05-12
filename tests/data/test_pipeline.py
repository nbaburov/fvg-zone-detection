"""Tests for pipeline.py — end-to-end integration."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.labels.fvg import FVGLabeller


# ---------------------------------------------------------------------------
# Synthetic H1 fixture factory
# ---------------------------------------------------------------------------


def _make_labelled_h1(seed: int = 42) -> pd.DataFrame:
    """
    Synthetic labelled H1 DataFrame spanning 2018–2024 with no cross-session gaps.
    Uses continuous 1h timestamps so windows never get dropped by the gap filter.

    The index is fake (not real trading hours) but covers all required split boundary years.
    n_bars is large enough for at least one 60-bar window per split.
    """
    rng = np.random.default_rng(seed)

    # Continuous 1h bars — no gaps > 90 min so no windows get dropped by gap filter.
    # Cover 2018-2024 with enough bars for 60-bar windows per split.
    # 1h * 61320 hours ≈ 7 years of continuous hourly bars.
    start = pd.Timestamp("2018-01-02 09:30", tz="America/New_York")
    end = pd.Timestamp("2024-12-31 23:30", tz="America/New_York")
    idx = pd.date_range(start, end, freq="1h")
    n = len(idx)

    prices = 400.0 + rng.normal(0, 0.5, n).cumsum() + 350.0
    prices = np.abs(prices)

    labeller = FVGLabeller()
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.01, 1.0, n),
            "low": prices - rng.uniform(0.01, 1.0, n),
            "close": prices + rng.normal(0, 0.2, n),
            "volume": rng.integers(5000, 20000, n).astype(float),
            "session_type": pd.Categorical(["full"] * n, categories=["full", "half"]),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    raw = labeller.label(df)
    df["raw_label"] = raw
    df["label"] = labeller.encode(raw)

    return df


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_pipeline_returns_three_datasets_and_weights():
    """build_pipeline returns (train_ds, val_ds, test_ds, class_weights)."""
    full_df = _make_labelled_h1()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.data.pipeline.build_labelled_dataset", return_value=full_df):
            with patch("src.data.pipeline.PROCESSED_DIR", tmpdir):
                from src.data.pipeline import build_pipeline
                train_ds, val_ds, test_ds, weights = build_pipeline(
                    labeller_name="fvg_valid",
                    window_size=60,
                    
                )

    assert train_ds is not None
    assert val_ds is not None
    assert test_ds is not None
    assert isinstance(weights, torch.Tensor)


def test_class_weights_shape_and_sum():
    """class_weights shape == (3,) and sum ≈ 3.0."""
    full_df = _make_labelled_h1()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.data.pipeline.build_labelled_dataset", return_value=full_df):
            with patch("src.data.pipeline.PROCESSED_DIR", tmpdir):
                from src.data.pipeline import build_pipeline
                _, _, _, weights = build_pipeline(labeller_name="fvg_valid", window_size=60)

    assert weights.shape == (3,), f"Expected shape (3,), got {weights.shape}"
    assert abs(float(weights.sum()) - 3.0) < 0.01, f"weights.sum()={float(weights.sum())}, expected ≈ 3.0"


def test_no_candle_overlap_between_splits():
    """No candle index appears in both train and val, or val and test."""
    full_df = _make_labelled_h1()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.data.pipeline.build_labelled_dataset", return_value=full_df):
            with patch("src.data.pipeline.PROCESSED_DIR", tmpdir):
                from src.data.pipeline import build_pipeline
                # We test at DataFrame level (written parquets)
                build_pipeline(labeller_name="fvg_valid", window_size=60)

                train_df = pd.read_parquet(os.path.join(tmpdir, "spy_h1_train.parquet"))
                val_df = pd.read_parquet(os.path.join(tmpdir, "spy_h1_val.parquet"))
                test_df = pd.read_parquet(os.path.join(tmpdir, "spy_h1_test.parquet"))

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert len(train_idx & val_idx) == 0, "Overlap between train and val splits"
    assert len(val_idx & test_idx) == 0, "Overlap between val and test splits"


def test_class_weights_json_written():
    """class_weights.json must be written with keys '0', '1', '2'."""
    full_df = _make_labelled_h1()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.data.pipeline.build_labelled_dataset", return_value=full_df):
            with patch("src.data.pipeline.PROCESSED_DIR", tmpdir):
                from src.data.pipeline import build_pipeline
                build_pipeline(labeller_name="fvg_valid", window_size=60)

        weights_path = os.path.join(tmpdir, "class_weights.json")
        assert os.path.exists(weights_path), "class_weights.json not written"
        with open(weights_path) as f:
            w = json.load(f)
        assert set(w.keys()) == {"0", "1", "2"}, f"Unexpected keys: {set(w.keys())}"


def test_train_stride1_val_test_stride60():
    """Train dataset uses stride=1, val and test use stride=60."""
    full_df = _make_labelled_h1()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.data.pipeline.build_labelled_dataset", return_value=full_df):
            with patch("src.data.pipeline.PROCESSED_DIR", tmpdir):
                from src.data.pipeline import build_pipeline
                train_ds, val_ds, test_ds, _ = build_pipeline(
                    labeller_name="fvg_valid", window_size=60                )

    # train should have many more windows than val/test (stride=1 vs stride=60)
    assert len(train_ds) > len(val_ds), (
        f"train ({len(train_ds)}) should have more windows than val ({len(val_ds)})"
    )
    assert len(train_ds) > len(test_ds), (
        f"train ({len(train_ds)}) should have more windows than test ({len(test_ds)})"
    )
