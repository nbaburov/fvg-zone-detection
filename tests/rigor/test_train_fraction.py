"""tests/rigor/test_train_fraction.py — Unit tests for --train-fraction head-slice logic.

Tests are parametrized over all four DL architectures and cover:
  1. _head_slice_train(df, 0.5) yields n_rows ≈ 0.5 * full (±5 %).
  2. Default fraction 1.0 is an exact no-op (same object length, contiguous rows).
  3. Window counts from SMCWindowDataset scale proportionally with sliced rows.

No real model training is performed — this purely exercises the data-slicing path.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts.training.train import _head_slice_train
from src.data.labels.valid_fvg import ValidFVGLabeller
from src.data.window import build_windows

# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------

ARCHS = ["lstm", "cnn_lstm", "xlstm", "transformer"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labeled_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Return a minimal OHLCV + label DataFrame of length *n* with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 1.0, n).cumsum()
    prices = np.abs(prices) + 300.0
    df = pd.DataFrame(
        {
            "open":  prices + rng.normal(0, 0.1, n),
            "high":  prices + rng.uniform(0.01, 0.5, n),
            "low":   prices - rng.uniform(0.01, 0.5, n),
            "close": prices + rng.normal(0, 0.1, n),
            "volume": rng.integers(5_000, 20_000, n).astype(float),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    labeller = ValidFVGLabeller()
    df["label"] = labeller.label(df)
    return df


# ---------------------------------------------------------------------------
# _head_slice_train — row-level assertions
# ---------------------------------------------------------------------------

class TestHeadSliceTrain:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_half_fraction_row_count(self, arch: str) -> None:
        """--train-fraction 0.5 yields ≈50 % of rows (±5 %)."""
        df = _make_labeled_df(500)
        sliced = _head_slice_train(df, 0.5)
        expected = math.ceil(len(df) * 0.5)
        assert len(sliced) == expected, (
            f"[{arch}] expected {expected} rows, got {len(sliced)}"
        )

    @pytest.mark.parametrize("arch", ARCHS)
    def test_half_fraction_within_tolerance(self, arch: str) -> None:
        """Sliced row count is within ±5 % of 0.5 * full for a realistic-size df."""
        df = _make_labeled_df(1_000)
        sliced = _head_slice_train(df, 0.5)
        ratio = len(sliced) / len(df)
        assert abs(ratio - 0.5) <= 0.05, (
            f"[{arch}] ratio {ratio:.3f} outside ±5 % of 0.5"
        )

    @pytest.mark.parametrize("arch", ARCHS)
    def test_default_fraction_is_noop(self, arch: str) -> None:
        """train_fraction=1.0 returns the original DataFrame unchanged (same length, same rows)."""
        df = _make_labeled_df(400)
        result = _head_slice_train(df, 1.0)
        assert len(result) == len(df), (
            f"[{arch}] 1.0 fraction should be a no-op, got {len(result)} != {len(df)}"
        )
        # Verify it is an identical slice — same index and values (not shuffled)
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_contiguous_head_order_preserved(self, arch: str) -> None:
        """Head slice must be the earliest rows — temporal order must not be broken."""
        df = _make_labeled_df(300)
        sliced = _head_slice_train(df, 0.5)
        expected_idx = df.index[: len(sliced)]
        assert sliced.index.equals(expected_idx), (
            f"[{arch}] sliced index does not match earliest rows of original df"
        )

    @pytest.mark.parametrize("arch", ARCHS)
    def test_fraction_ceiling_rounds_up(self, arch: str) -> None:
        """math.ceil semantics: fraction applied to odd-length df rounds up."""
        df = _make_labeled_df(101)
        sliced = _head_slice_train(df, 0.5)
        # math.ceil(101 * 0.5) = math.ceil(50.5) = 51
        assert len(sliced) == math.ceil(101 * 0.5), (
            f"[{arch}] ceiling rounding failed"
        )


# ---------------------------------------------------------------------------
# Window count proportionality via build_windows
# ---------------------------------------------------------------------------

class TestWindowCountProportionality:
    """Verify that slicing rows before build_windows scales window counts correctly."""

    WINDOW_SIZE = 60
    STRIDE = 1

    @pytest.mark.parametrize("arch", ARCHS)
    def test_half_fraction_windows_approx_half(self, arch: str) -> None:
        """Windows from 0.5-slice are within ±5 % of half the full window count."""
        df = _make_labeled_df(800)
        labeller = ValidFVGLabeller()

        full_windows  = build_windows(df, labeller, stride=self.STRIDE, window_size=self.WINDOW_SIZE)
        sliced_df     = _head_slice_train(df, 0.5)
        half_windows  = build_windows(sliced_df, labeller, stride=self.STRIDE, window_size=self.WINDOW_SIZE)

        n_full = len(full_windows)
        n_half = len(half_windows)
        assert n_full > 0, f"[{arch}] full window set is empty"
        ratio = n_half / n_full
        assert abs(ratio - 0.5) <= 0.05, (
            f"[{arch}] window count ratio {ratio:.3f} not within ±5 % of 0.5 "
            f"(full={n_full}, half={n_half})"
        )

    @pytest.mark.parametrize("arch", ARCHS)
    def test_full_fraction_windows_equals_default(self, arch: str) -> None:
        """Windows from fraction=1.0 must equal windows from unsliced df."""
        df = _make_labeled_df(400)
        labeller = ValidFVGLabeller()

        full_windows     = build_windows(df, labeller, stride=self.STRIDE, window_size=self.WINDOW_SIZE)
        noop_df          = _head_slice_train(df, 1.0)
        noop_windows     = build_windows(noop_df, labeller, stride=self.STRIDE, window_size=self.WINDOW_SIZE)

        assert len(full_windows) == len(noop_windows), (
            f"[{arch}] fraction=1.0 window count {len(noop_windows)} != "
            f"unsliced count {len(full_windows)}"
        )
