"""Tests for src/data/labels/bos.py — causal BOS detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labels.bos import compute_bos


def _make_df(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02 09:30", periods=len(closes), freq="1h", tz="America/New_York")
    if highs is None:
        highs = [c + 0.5 for c in closes]
    if lows is None:
        lows = [c - 0.5 for c in closes]
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# No BOS case — all closes within prior swing range
# ---------------------------------------------------------------------------


def test_no_bull_bos_when_price_flat():
    """When price never exceeds prior swing high, bull_bos_recent should be False."""
    # Flat market: closes all at 100.0. prior_swing_high = 100.5 (from high). close never > high.
    n = 60
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n
    df = _make_df(closes, highs, lows)
    bull_bos, bear_bos = compute_bos(df, swing_lookback=10, bos_lookback=5)

    # No closes exceed prior swing high — all False after warmup
    # (first bar might be NaN/False due to warmup)
    assert not bull_bos.iloc[15:].any(), "Expected no bull BOS in flat market"


def test_no_bear_bos_when_price_flat():
    """When price never falls below prior swing low, bear_bos_recent should be False."""
    n = 60
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n
    df = _make_df(closes, highs, lows)
    bull_bos, bear_bos = compute_bos(df, swing_lookback=10, bos_lookback=5)

    assert not bear_bos.iloc[15:].any(), "Expected no bear BOS in flat market"


# ---------------------------------------------------------------------------
# BOS present: correct detection and recency window
# ---------------------------------------------------------------------------


def test_bull_bos_detected_after_breakout():
    """A single close above prior swing high triggers bull_bos_recent for bos_lookback bars."""
    n = 80
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n

    # Inject breakout at bar 40: close > max high seen in past 50 bars (100.5)
    closes[40] = 102.0
    highs[40] = 102.5

    df = _make_df(closes, highs, lows)
    bull_bos, _ = compute_bos(df, swing_lookback=10, bos_lookback=5)

    # bull_bos_recent[41] should be True (BOS at 40, checked at 41 via shift(1))
    assert bull_bos.iloc[41], "Expected bull_bos_recent True at bar 41 after BOS at bar 40"

    # bull_bos_recent[46] should also be True (within 5-bar bos_lookback)
    assert bull_bos.iloc[45], "Expected bull_bos_recent True at bar 45 (within lookback)"

    # bull_bos_recent[47] should be False (BOS at bar 40 is > 5 bars ago)
    # Bar 47: BOS was at bar 40, which is 7 bars before. bos_lookback=5 — not recent.
    assert not bull_bos.iloc[47], "Expected bull_bos_recent False at bar 47 (BOS too old)"


def test_bear_bos_detected_after_breakdown():
    """A close below prior swing low triggers bear_bos_recent for bos_lookback bars."""
    n = 80
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n

    # Inject breakdown at bar 40: close < min low in past 10 bars (99.5)
    closes[40] = 97.0
    lows[40] = 96.5

    df = _make_df(closes, highs, lows)
    _, bear_bos = compute_bos(df, swing_lookback=10, bos_lookback=5)

    assert bear_bos.iloc[41], "Expected bear_bos_recent True at bar 41 after breakdown at bar 40"
    assert not bear_bos.iloc[47], "Expected bear_bos_recent False at bar 47 (breakdown too old)"


# ---------------------------------------------------------------------------
# Causality: bar i result does NOT depend on bar i's close
# ---------------------------------------------------------------------------


def test_bull_bos_does_not_use_current_bar_close():
    """
    bull_bos_recent[i] must be based on bars before i.
    Mutating close[i] to an extreme value must NOT change bull_bos_recent[i].
    """
    n = 60
    closes = [100.0] * n
    df_orig = _make_df(closes)
    bull_orig, _ = compute_bos(df_orig, swing_lookback=10, bos_lookback=5)

    df_mut = df_orig.copy()
    # Set close at bar 30 to an extreme that would trigger a BOS
    df_mut.iloc[30, df_mut.columns.get_loc("close")] = 9999.0
    df_mut.iloc[30, df_mut.columns.get_loc("high")] = 9999.0

    bull_mut, _ = compute_bos(df_mut, swing_lookback=10, bos_lookback=5)

    # bull_bos_recent[30] should be unchanged (bar 30's OWN close doesn't count for bar 30's BOS)
    assert bull_orig.iloc[30] == bull_mut.iloc[30], (
        "bull_bos_recent[30] changed when close[30] was mutated — indicates lookahead"
    )

    # But bull_bos_recent[31] should now be True (bar 30 BOS is now in the past of bar 31)
    assert bull_mut.iloc[31], "Expected bull_bos_recent[31] True after injecting BOS at bar 30"


# ---------------------------------------------------------------------------
# Return type and index
# ---------------------------------------------------------------------------


def test_return_type_and_index():
    """Both outputs are bool pandas Series with same index as input df."""
    n = 30
    closes = [100.0 + i * 0.1 for i in range(n)]
    df = _make_df(closes)
    bull_bos, bear_bos = compute_bos(df, swing_lookback=10, bos_lookback=5)

    assert isinstance(bull_bos, pd.Series)
    assert isinstance(bear_bos, pd.Series)
    assert bull_bos.index.equals(df.index)
    assert bear_bos.index.equals(df.index)
    assert bull_bos.dtype == bool
    assert bear_bos.dtype == bool
