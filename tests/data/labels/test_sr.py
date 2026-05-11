"""Tests for src/data/labels/sr.py — causal pivot high/low computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labels.sr import compute_pivot_levels


def _make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02 09:30", periods=len(highs), freq="1h", tz="America/New_York")
    return pd.DataFrame(
        {
            "open": [h - 0.1 for h in highs],
            "high": highs,
            "low": lows,
            "close": [h - 0.05 for h in highs],
            "volume": [1000.0] * len(highs),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# NaN prefix — first lookback rows must be NaN
# ---------------------------------------------------------------------------


def test_pivot_nan_for_first_lookback_bars():
    """First lookback rows of pivot_high and pivot_low must be NaN."""
    highs = [float(i) for i in range(1, 21)]  # 20 bars
    lows = [float(i) - 0.5 for i in range(1, 21)]
    df = _make_df(highs, lows)
    ph, pl = compute_pivot_levels(df, lookback=5)

    # First 5 bars must be NaN (lookback=5 means we need 5 bars to get a value; shift(1) pushes by 1)
    assert ph.iloc[:5].isna().all(), f"Expected NaN in first 5 rows of pivot_high, got {ph.iloc[:5].tolist()}"
    assert pl.iloc[:5].isna().all(), f"Expected NaN in first 5 rows of pivot_low, got {pl.iloc[:5].tolist()}"


def test_pivot_non_nan_after_lookback():
    """After the warmup period, pivot_high and pivot_low should be non-NaN."""
    n = 20
    highs = [float(i) for i in range(1, n + 1)]
    lows = [float(i) - 0.5 for i in range(1, n + 1)]
    df = _make_df(highs, lows)
    ph, pl = compute_pivot_levels(df, lookback=5)

    # After index 5, values should be non-NaN
    assert ph.iloc[5:].notna().all(), "pivot_high has NaN after warmup"
    assert pl.iloc[5:].notna().all(), "pivot_low has NaN after warmup"


# ---------------------------------------------------------------------------
# Correct values — hand-computed
# ---------------------------------------------------------------------------


def test_pivot_high_correct_value():
    """pivot_high[i] = max of highs in [i-lookback..i-1]."""
    # highs: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    highs = [float(i) for i in range(1, 11)]
    lows = [float(i) - 0.5 for i in range(1, 11)]
    df = _make_df(highs, lows)
    ph, pl = compute_pivot_levels(df, lookback=3)

    # At index 3: shift(1) means rolling window was computed at index 2.
    # rolling(3).max() at index 2 = max(highs[0:3]) = max(1,2,3) = 3.
    # Then shift(1) moves it to index 3. So ph[3] = 3.
    assert ph.iloc[3] == pytest.approx(3.0), f"Expected pivot_high[3]=3.0, got {ph.iloc[3]}"

    # At index 5: rolling window at index 4 covers [2,3,4] (0-indexed highs 3,4,5) = max=5.
    # Actually rolling(3) at index 4 = max(high[2..4]) = max(3,4,5) = 5.
    assert ph.iloc[5] == pytest.approx(5.0), f"Expected pivot_high[5]=5.0, got {ph.iloc[5]}"


def test_pivot_low_correct_value():
    """pivot_low[i] = min of lows in [i-lookback..i-1]."""
    highs = [float(i) for i in range(10, 20)]
    lows = [float(i) - 0.5 for i in range(10, 20)]  # lows: 9.5, 10.5, 11.5, ...
    df = _make_df(highs, lows)
    ph, pl = compute_pivot_levels(df, lookback=3)

    # At index 3: rolling(3).min() at index 2 = min(lows[0..2]) = min(9.5, 10.5, 11.5) = 9.5
    assert pl.iloc[3] == pytest.approx(9.5), f"Expected pivot_low[3]=9.5, got {pl.iloc[3]}"


# ---------------------------------------------------------------------------
# Causality: mutating bar i does not change pivot_high[i]
# ---------------------------------------------------------------------------


def test_mutating_bar_i_high_does_not_change_pivot_high_at_i():
    """
    pivot_high[i] uses only bars before i.
    Mutating high[i] must NOT change pivot_high[i].
    """
    highs = [float(x) for x in range(1, 16)]
    lows = [float(x) - 0.3 for x in range(1, 16)]
    df_orig = _make_df(highs, lows)
    ph_orig, _ = compute_pivot_levels(df_orig, lookback=5)

    # Mutate bar at index 7 (within valid range)
    df_mut = df_orig.copy()
    df_mut.iloc[7, df_mut.columns.get_loc("high")] = 9999.0

    ph_mut, _ = compute_pivot_levels(df_mut, lookback=5)

    # pivot_high[7] should be unchanged (9999 is at bar 7, not in its past)
    assert ph_mut.iloc[7] == pytest.approx(ph_orig.iloc[7]), (
        f"pivot_high[7] changed after mutating high[7]: "
        f"before={ph_orig.iloc[7]}, after={ph_mut.iloc[7]}"
    )

    # pivot_high[8] should now reflect the mutation (bar 7 is in its past)
    assert ph_mut.iloc[8] != ph_orig.iloc[8], (
        "Expected pivot_high[8] to change after mutating high[7]"
    )


def test_mutating_bar_i_low_does_not_change_pivot_low_at_i():
    """
    pivot_low[i] uses only bars before i.
    Mutating low[i] must NOT change pivot_low[i].
    """
    highs = [float(x) for x in range(1, 16)]
    lows = [float(x) - 0.3 for x in range(1, 16)]
    df_orig = _make_df(highs, lows)
    _, pl_orig = compute_pivot_levels(df_orig, lookback=5)

    df_mut = df_orig.copy()
    df_mut.iloc[7, df_mut.columns.get_loc("low")] = -9999.0

    _, pl_mut = compute_pivot_levels(df_mut, lookback=5)

    assert pl_mut.iloc[7] == pytest.approx(pl_orig.iloc[7]), (
        f"pivot_low[7] changed after mutating low[7]"
    )
    assert pl_mut.iloc[8] != pl_orig.iloc[8], (
        "Expected pivot_low[8] to change after mutating low[7]"
    )


# ---------------------------------------------------------------------------
# Return type and index
# ---------------------------------------------------------------------------


def test_return_type_and_index():
    """Both outputs are pandas Series with same index as input df."""
    highs = [float(i) for i in range(1, 11)]
    lows = [float(i) - 0.5 for i in range(1, 11)]
    df = _make_df(highs, lows)
    ph, pl = compute_pivot_levels(df, lookback=3)

    assert isinstance(ph, pd.Series)
    assert isinstance(pl, pd.Series)
    assert ph.index.equals(df.index)
    assert pl.index.equals(df.index)
