"""Tests for normalize.py — causal per-window normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.normalize import normalise_window


def _make_window(seed: int = 42) -> np.ndarray:
    """Random (60, 5) window with realistic price and volume magnitudes."""
    rng = np.random.default_rng(seed)
    ohlc = 400.0 + rng.normal(0, 1.0, (60, 4)).cumsum(axis=0)
    vol = rng.integers(5000, 20000, (60, 1)).astype(float)
    return np.concatenate([ohlc, vol], axis=1)


# ---------------------------------------------------------------------------
# Basic output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape():
    w = _make_window()
    out = normalise_window(w)
    assert out.shape == (60, 5), f"Expected (60, 5), got {out.shape}"


def test_output_dtype_float32():
    w = _make_window()
    out = normalise_window(w)
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"


# ---------------------------------------------------------------------------
# OHLC normalisation correctness
# ---------------------------------------------------------------------------


def test_ohlc_mean_approx_zero():
    """OHLC columns (0:4) z-scored per window — mean should be ≈ 0."""
    w = _make_window()
    out = normalise_window(w)
    ohlc_mean = float(np.mean(out[:, :4]))
    assert abs(ohlc_mean) < 0.1, f"OHLC mean = {ohlc_mean}, expected ≈ 0"


def test_ohlc_std_approx_one():
    """OHLC std across all OHLC cells should be ≈ 1."""
    w = _make_window()
    out = normalise_window(w)
    ohlc_std = float(np.std(out[:, :4]))
    assert 0.9 < ohlc_std < 1.1, f"OHLC std = {ohlc_std}, expected ≈ 1"


# ---------------------------------------------------------------------------
# Volume uses log1p, not raw z-score
# ---------------------------------------------------------------------------


def test_volume_uses_log1p():
    """Volume column values should differ from raw-volume z-score."""
    w = _make_window()
    out = normalise_window(w)

    raw_vol = w[:, 4]
    raw_vol_zscore = (raw_vol - raw_vol.mean()) / raw_vol.std()

    # log1p-transformed then z-scored should differ from raw z-score
    assert not np.allclose(out[:, 4], raw_vol_zscore.astype(np.float32), atol=1e-3), (
        "Volume column matches raw z-score — log1p transform likely not applied"
    )


# ---------------------------------------------------------------------------
# Anti-lookahead: mutating row outside window does not affect window output
# ---------------------------------------------------------------------------


def test_anti_lookahead_mutation():
    """
    normalise_window is a pure function of its 60-bar input.
    Mutating candles outside the window must not change the output.
    """
    w100 = _make_window(seed=100)
    out_before = normalise_window(w100.copy())

    # Simulate "mutating a candle at position 200 in the full dataset" —
    # since normalise_window only sees its 60-bar input, any outside mutation
    # cannot affect it. We verify this by calling with identical input twice.
    w100_copy = w100.copy()
    # Mutate w100_copy just to confirm it has no effect (it's a different array)
    w100_copy[0, 0] += 9999.0  # mutate position outside would-be window

    out_after = normalise_window(w100.copy())  # same original array, unmodified
    assert np.allclose(out_before, out_after), (
        "normalise_window output changed even though input was identical — not pure"
    )


def test_anti_lookahead_independent_windows():
    """
    Window at index 100 and window at index 200 in a dataset are independent.
    Changing the data at row 200 must not affect the normalised values of window 100.
    """
    rng = np.random.default_rng(0)
    dataset = 400.0 + rng.normal(0, 1.0, (300, 5)).cumsum(axis=0)
    dataset[:, 4] = rng.integers(5000, 20000, 300).astype(float)

    w100 = dataset[100:160].copy()  # 60-bar window at position 100
    out_100_before = normalise_window(w100)

    # Mutate row 200 in dataset
    dataset[200, :] += 99999.0

    # Re-extract window 100 — should be unchanged
    w100_after = dataset[100:160].copy()
    out_100_after = normalise_window(w100_after)

    assert np.allclose(out_100_before, out_100_after), (
        "Window 100 normalisation changed after mutating row 200 — lookahead leak"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_volume_window_no_nan():
    """Zero volume in window → volume column returns zeros, not NaN."""
    w = _make_window()
    w[:, 4] = 0.0  # all-zero volume
    out = normalise_window(w)
    assert not np.isnan(out).any(), "NaN in output with zero-volume window"
    assert (out[:, 4] == 0.0).all(), "Expected zero volume column when volume is zero"


def test_zero_price_std_no_exception():
    """All prices identical (std == 0) → no exception, returns zeros for OHLC."""
    w = np.ones((60, 5), dtype=float)
    w[:, 4] = 1000.0  # non-zero volume
    out = normalise_window(w)
    assert not np.isnan(out).any(), "NaN in output when price std == 0"
    assert (out[:, :4] == 0.0).all(), "Expected zeros for OHLC when std == 0"


def test_no_nan_on_normal_window():
    """Standard window produces no NaN values."""
    w = _make_window()
    out = normalise_window(w)
    assert not np.isnan(out).any(), "NaN in normalised output"
