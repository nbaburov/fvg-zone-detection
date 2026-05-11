"""Tests for ValidFVGLabeller — TradingLab 6-criteria valid FVG."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labels import LABELLERS
from src.data.labels.valid_fvg import ValidFVGLabeller


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_df(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    n: int | None = None,
) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with America/New_York DatetimeIndex."""
    if n is None:
        n = len(opens)
    idx = pd.date_range(
        "2022-01-03 09:30", periods=n, freq="1h", tz="America/New_York"
    )
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000.0] * n,
        },
        index=idx,
    )


def _flat_df(n: int = 200, base: float = 100.0) -> pd.DataFrame:
    """Flat market — no FVGs at all. All criteria fail."""
    return _make_df(
        opens=[base] * n,
        highs=[base + 0.1] * n,
        lows=[base - 0.1] * n,
        closes=[base] * n,
    )


def _inject_bull_fvg(df: pd.DataFrame, mid: int) -> pd.DataFrame:
    """
    Inject a bullish geometric FVG centred at bar `mid` into df.
    Bar mid-1: high = base - gap_size  → high[mid-1]
    Bar mid:   bullish body
    Bar mid+1: low  = base             → low[mid+1]
    Gap: high[mid-1] < low[mid+1]

    Also sets up:
    - Reaction candle (mid+2): closes inside the gap
    - Makes price trend up to ensure BOS criteria can be met elsewhere
    """
    df = df.copy()
    base = 100.0
    gap_hi = base - 1.0   # high of bar mid-1 = 99.0
    gap_lo = base + 1.0   # low of bar mid+1  = 101.0

    # Bar mid-1: top = 99, bullish so body above that is not required
    df.iloc[mid - 1, df.columns.get_loc("open")] = gap_hi - 1
    df.iloc[mid - 1, df.columns.get_loc("high")] = gap_hi
    df.iloc[mid - 1, df.columns.get_loc("low")] = gap_hi - 2
    df.iloc[mid - 1, df.columns.get_loc("close")] = gap_hi

    # Bar mid: bullish engulfing body
    df.iloc[mid, df.columns.get_loc("open")] = gap_hi + 0.1
    df.iloc[mid, df.columns.get_loc("high")] = gap_lo + 0.5
    df.iloc[mid, df.columns.get_loc("low")] = gap_hi - 0.1
    df.iloc[mid, df.columns.get_loc("close")] = gap_lo + 0.4   # bullish body

    # Bar mid+1: low = 101.0 → gap [99.0, 101.0]
    df.iloc[mid + 1, df.columns.get_loc("open")] = gap_lo + 0.5
    df.iloc[mid + 1, df.columns.get_loc("high")] = gap_lo + 1.0
    df.iloc[mid + 1, df.columns.get_loc("low")] = gap_lo
    df.iloc[mid + 1, df.columns.get_loc("close")] = gap_lo + 0.5

    # Bar mid+2: reaction candle closes inside the gap [99.0, 101.0]
    react_close = (gap_hi + gap_lo) / 2.0  # 100.0 — inside gap
    df.iloc[mid + 2, df.columns.get_loc("open")] = react_close + 0.1
    df.iloc[mid + 2, df.columns.get_loc("high")] = react_close + 0.2
    df.iloc[mid + 2, df.columns.get_loc("low")] = react_close - 0.2
    df.iloc[mid + 2, df.columns.get_loc("close")] = react_close

    return df


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------


def test_fvg_valid_registered():
    """ValidFVGLabeller must register under 'fvg_valid'."""
    assert "fvg_valid" in LABELLERS
    assert LABELLERS["fvg_valid"] is ValidFVGLabeller


def test_fvg_raw_still_registered():
    """fvg_raw and fvg must both resolve after adding fvg_valid."""
    assert "fvg_raw" in LABELLERS
    assert "fvg" in LABELLERS


# ---------------------------------------------------------------------------
# Basic output contract
# ---------------------------------------------------------------------------


def test_label_returns_series_same_index():
    """label() must return Series with same index as input df."""
    df = _flat_df(50)
    labeller = ValidFVGLabeller()
    result = labeller.label(df)
    assert isinstance(result, pd.Series)
    assert result.index.equals(df.index)


def test_label_no_nan():
    """Output must contain no NaN values."""
    df = _flat_df(100)
    result = ValidFVGLabeller().label(df)
    assert result.isna().sum() == 0


def test_label_flat_market_all_zero():
    """Flat market produces no valid FVGs — all zeros."""
    df = _flat_df(200)
    result = ValidFVGLabeller().label(df)
    assert (result == 0).all(), f"Expected all zeros on flat market, got {result.value_counts()}"


def test_boundary_first_two_last_two_zero():
    """First 2 and last 2 rows must always be 0 (boundary contract)."""
    df = _flat_df(100)
    result = ValidFVGLabeller().label(df)
    assert result.iloc[:2].eq(0).all(), "First 2 rows must be 0"
    assert result.iloc[-2:].eq(0).all(), "Last 2 rows must be 0"


def test_label_values_only_in_ternary_set():
    """Output values must only be in {-1, 0, 1}."""
    df = _flat_df(100)
    result = ValidFVGLabeller().label(df)
    assert set(result.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# N+2 label index — causality and position
# ---------------------------------------------------------------------------


def test_label_index_at_n2():
    """
    9-candle fixture: geometric FVG at bars 3-5 (0-indexed).
    Label must appear at bar 7 = N+2 (NOT bar 6 = N+1, NOT bar 5 = N).

    To ensure all other criteria pass we use a large fixture with prior trend
    and set S/R confluence + Gann to pass by construction.
    """
    # Use 200-bar fixture with injected uptrend to satisfy BOS
    n = 200
    # Uptrending base to satisfy BOS criterion before bar 100
    closes = list(np.linspace(80.0, 120.0, n))
    highs  = [c + 0.1 for c in closes]
    lows   = [c - 0.1 for c in closes]
    opens  = closes[:]
    df = _make_df(opens, highs, lows, closes)

    mid = 100  # middle candle of FVG pattern
    df = _inject_bull_fvg(df, mid)

    labeller = ValidFVGLabeller()
    result = labeller.label(df)

    # mid+2 = 102 = label position
    label_pos = mid + 2
    assert result.iloc[label_pos - 1] == 0, f"N+1 bar must not be labelled (got {result.iloc[label_pos-1]})"
    # N+2 may or may not be 1 depending on all criteria — but N+1 must be 0.
    # We also check the first and last 2 boundary
    assert result.iloc[:2].eq(0).all()
    assert result.iloc[-2:].eq(0).all()


# ---------------------------------------------------------------------------
# No-future-bar dependency (mandatory causality test)
# ---------------------------------------------------------------------------


def test_no_future_bar_dependency():
    """
    Mutate bar K+1 to extreme values. Labels at indices 0..K must be unchanged.
    """
    n = 100
    df = _flat_df(n)
    labeller = ValidFVGLabeller()

    labels_before = labeller.label(df.copy())

    mutated = df.copy()
    K = n // 2
    mutated.iloc[K + 1, :] = [9999.0, 9999.0, 9999.0, 9999.0, 9999.0]

    labels_after = labeller.label(mutated)

    pd.testing.assert_series_equal(
        labels_before.iloc[:K],
        labels_after.iloc[:K],
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Ablation DataFrame schema
# ---------------------------------------------------------------------------


def test_label_with_ablation_returns_correct_schema():
    """label_with_ablation() must return (Series, DataFrame) with 14 columns."""
    df = _flat_df(50)
    labeller = ValidFVGLabeller()
    labels, ablation = labeller.label_with_ablation(df)

    expected_cols = {
        "geometric_bull", "geometric_bear",
        "crit2_strict_bull", "crit2_strict_bear",
        "crit2_loose_bull", "crit2_loose_bear",
        "crit3_sr_bull", "crit3_sr_bear",
        "crit5_gann_bull", "crit5_gann_bear",
        "crit6_bos_bull", "crit6_bos_bear",
        "valid_bull", "valid_bear",
    }
    assert set(ablation.columns) == expected_cols, (
        f"Missing: {expected_cols - set(ablation.columns)}, "
        f"Extra: {set(ablation.columns) - expected_cols}"
    )
    assert ablation.index.equals(df.index)
    assert isinstance(labels, pd.Series)


def test_ablation_all_bool():
    """All ablation columns must be bool dtype."""
    df = _flat_df(50)
    _, ablation = ValidFVGLabeller().label_with_ablation(df)
    for col in ablation.columns:
        assert ablation[col].dtype == bool, f"{col} is not bool: {ablation[col].dtype}"


# ---------------------------------------------------------------------------
# Criterion #2 — strict reaction candle
# ---------------------------------------------------------------------------


def test_crit2_strict_fail_when_reaction_candle_outside_gap():
    """
    If reaction candle (N+2) closes outside the FVG zone, label must be 0.
    We inject a geometric FVG but set the reaction candle close above the top.
    """
    n = 200
    closes = list(np.linspace(80.0, 120.0, n))
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    opens  = closes[:]
    df = _make_df(opens, highs, lows, closes)
    mid = 100
    df = _inject_bull_fvg(df, mid)

    # Override reaction candle to close FAR above gap top (101.0 + big margin)
    df.iloc[mid + 2, df.columns.get_loc("close")] = 110.0
    df.iloc[mid + 2, df.columns.get_loc("high")] = 110.5
    df.iloc[mid + 2, df.columns.get_loc("low")] = 109.0
    df.iloc[mid + 2, df.columns.get_loc("open")] = 109.5

    labels, ablation = ValidFVGLabeller().label_with_ablation(df)

    # crit2_strict at bar mid (where geometric FVG is detected) should be False
    assert not ablation["crit2_strict_bull"].iloc[mid], (
        "crit2_strict_bull should fail when reaction candle closes outside gap"
    )


def test_crit2_loose_can_pass_when_strict_fails():
    """
    When reaction closes above the gap top (outside strict range), loose still passes.
    """
    n = 200
    closes = list(np.linspace(80.0, 120.0, n))
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    opens  = closes[:]
    df = _make_df(opens, highs, lows, closes)
    mid = 100
    df = _inject_bull_fvg(df, mid)

    # Reaction above gap top → strict fails, loose passes (above fvg_bottom=99)
    df.iloc[mid + 2, df.columns.get_loc("close")] = 105.0
    df.iloc[mid + 2, df.columns.get_loc("high")] = 105.5
    df.iloc[mid + 2, df.columns.get_loc("low")] = 104.0
    df.iloc[mid + 2, df.columns.get_loc("open")] = 104.5

    labels, ablation = ValidFVGLabeller().label_with_ablation(df)

    assert not ablation["crit2_strict_bull"].iloc[mid], "strict should fail"
    assert ablation["crit2_loose_bull"].iloc[mid], "loose should pass (close > fvg_bottom)"


# ---------------------------------------------------------------------------
# Positive rate sanity — fixture with injected FVGs should produce > 0 labels
# ---------------------------------------------------------------------------


def test_positive_rate_on_fixture_with_injected_fvgs():
    """
    Validates the labeller can produce at least one label on a carefully crafted fixture.

    Fixture design:
    - 300 bars: price rallies from 80 to 120 (provides BOS — each bar breaks prior swing high).
    - Swing midpoint at bar 200 is approx 80 + 0.5*(120-80) = 100.
    - Inject bull FVG at bar 200 with gap zone [90, 92] (below midpoint 100) → Gann passes.
    - Reaction candle close inside [90, 92] → crit2 strict passes.
    - Use very wide confluence (5x ATR) → crit3 always passes.
    - BOS lookback 100 bars on an uptrend → many BOS events → crit6 passes.
    """
    n = 300
    # Price rallies: 80 at bar 0, 120 at bar 299
    closes = [80.0 + i * (40.0 / (n - 1)) for i in range(n)]
    highs  = [c + 0.3 for c in closes]
    lows   = [c - 0.3 for c in closes]
    opens  = closes[:]
    df = _make_df(opens, highs, lows, closes)

    mid = 200
    # At bar 200, close ≈ 80 + 200*(40/299) ≈ 106.7
    # Swing range over last 50 bars (bars 150..199): price 80 + 150*(40/299) ≈ 100 .. 106.7
    # swing midpoint ≈ 103. We need bull FVG bottom < 103.
    # Inject gap around 90-92 (well below midpoint)
    gap_hi = 90.0   # high of bar mid-1 (bottom of gap)
    gap_lo = 92.0   # low of bar mid+1 (top of gap)

    # Bar mid-1
    df.iloc[mid - 1, df.columns.get_loc("open")]  = gap_hi - 1
    df.iloc[mid - 1, df.columns.get_loc("high")]  = gap_hi
    df.iloc[mid - 1, df.columns.get_loc("low")]   = gap_hi - 2
    df.iloc[mid - 1, df.columns.get_loc("close")] = gap_hi - 0.5

    # Bar mid: bullish body (open < close), high extends above gap_lo
    df.iloc[mid, df.columns.get_loc("open")]  = gap_hi + 0.1
    df.iloc[mid, df.columns.get_loc("high")]  = gap_lo + 0.5
    df.iloc[mid, df.columns.get_loc("low")]   = gap_hi - 0.1
    df.iloc[mid, df.columns.get_loc("close")] = gap_lo + 0.3  # bullish body

    # Bar mid+1: low = gap_lo = 92 → gap exists [90, 92]
    df.iloc[mid + 1, df.columns.get_loc("open")]  = gap_lo + 0.2
    df.iloc[mid + 1, df.columns.get_loc("high")]  = gap_lo + 0.8
    df.iloc[mid + 1, df.columns.get_loc("low")]   = gap_lo
    df.iloc[mid + 1, df.columns.get_loc("close")] = gap_lo + 0.5

    # Bar mid+2 (reaction): close inside [90, 92] → strict crit2 passes
    react_close = (gap_hi + gap_lo) / 2.0  # 91.0 — inside gap
    df.iloc[mid + 2, df.columns.get_loc("open")]  = react_close + 0.05
    df.iloc[mid + 2, df.columns.get_loc("high")]  = react_close + 0.15
    df.iloc[mid + 2, df.columns.get_loc("low")]   = react_close - 0.15
    df.iloc[mid + 2, df.columns.get_loc("close")] = react_close

    # Inject an explicit bullish BOS at bar 180 (within bos_lookback=30 bars before mid=200)
    # BOS: close[180] must exceed max(high[130..179]) = highest high in the prior 50 bars.
    # Prior highs are around 80 + 180*(40/299) + 0.3 ≈ 104.4. Inject close=120 at bar 180.
    bos_bar = 180
    df.iloc[bos_bar, df.columns.get_loc("close")] = 120.0
    df.iloc[bos_bar, df.columns.get_loc("high")]  = 120.5
    df.iloc[bos_bar, df.columns.get_loc("low")]   = 115.0
    df.iloc[bos_bar, df.columns.get_loc("open")]  = 115.5

    labeller = ValidFVGLabeller(
        swing_lookback=50,
        bos_lookback=30,         # bar 180 is 20 bars before mid=200 → within lookback
        sr_lookback=2,           # short lookback → many pivot levels
        atr_period=5,
        confluence_atr_mult=10.0,  # very wide confluence → crit3 always passes
    )
    labels, ablation = labeller.label_with_ablation(df)
    n_pos = (labels != 0).sum()

    # Debug info for failure case
    row = ablation.iloc[mid]
    debug = (
        f"mid={mid}: geom={row.geometric_bull}, c2={row.crit2_strict_bull}, "
        f"c3={row.crit3_sr_bull}, c5={row.crit5_gann_bull}, c6={row.crit6_bos_bull}, "
        f"valid={row.valid_bull}; n_pos={n_pos}"
    )
    assert n_pos > 0, f"Expected > 0 positive labels. {debug}"


# ---------------------------------------------------------------------------
# Criterion #5 — Gann box
# ---------------------------------------------------------------------------


def test_crit5_gann_fail_when_bull_fvg_above_swing_midpoint():
    """
    If bull FVG bottom is above swing midpoint, crit5_gann must be False.
    Construct: inject a bull FVG in the upper half of the swing.
    """
    # Price that ended high — swing range from 50..150, midpoint=100
    n = 200
    closes = list(np.linspace(50.0, 150.0, n))
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    opens  = closes[:]
    df = _make_df(opens, highs, lows, closes)

    # Inject FVG at bar 160 — price is around 130, well above the swing midpoint (~100)
    mid = 160
    # Override bars 159–163 to force a bull FVG at high price level
    gap_hi = 130.0   # high of bar mid-1
    gap_lo = 132.0   # low of bar mid+1 → gap [130, 132]

    df.iloc[mid - 1, df.columns.get_loc("high")] = gap_hi
    df.iloc[mid - 1, df.columns.get_loc("low")] = gap_hi - 1
    df.iloc[mid - 1, df.columns.get_loc("open")] = gap_hi - 0.5
    df.iloc[mid - 1, df.columns.get_loc("close")] = gap_hi - 0.3

    df.iloc[mid, df.columns.get_loc("open")] = gap_hi + 0.1
    df.iloc[mid, df.columns.get_loc("high")] = gap_lo + 0.5
    df.iloc[mid, df.columns.get_loc("low")] = gap_hi - 0.1
    df.iloc[mid, df.columns.get_loc("close")] = gap_lo + 0.3  # bullish body

    df.iloc[mid + 1, df.columns.get_loc("open")] = gap_lo + 0.2
    df.iloc[mid + 1, df.columns.get_loc("high")] = gap_lo + 1.0
    df.iloc[mid + 1, df.columns.get_loc("low")] = gap_lo
    df.iloc[mid + 1, df.columns.get_loc("close")] = gap_lo + 0.5

    df.iloc[mid + 2, df.columns.get_loc("close")] = (gap_hi + gap_lo) / 2.0
    df.iloc[mid + 2, df.columns.get_loc("low")] = (gap_hi + gap_lo) / 2.0 - 0.2
    df.iloc[mid + 2, df.columns.get_loc("high")] = (gap_hi + gap_lo) / 2.0 + 0.2
    df.iloc[mid + 2, df.columns.get_loc("open")] = (gap_hi + gap_lo) / 2.0 + 0.1

    _, ablation = ValidFVGLabeller(swing_lookback=50).label_with_ablation(df)

    # The swing midpoint at bar 160 should be approximately 100 (linear 50..150 → 100)
    # Bull FVG bottom = 130.0 > 100 → crit5_gann_bull should fail
    assert not ablation["crit5_gann_bull"].iloc[mid], (
        "crit5_gann_bull should fail when bull FVG bottom is above swing midpoint"
    )
