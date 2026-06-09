"""test_fvg_exits.py — Unit tests for src/strategy/exits.py.

Geometry conventions (from exits.py docstring):
  bar_56 = window[56]  candle-1 (pre-impulse)
  bar_57 = window[57]  candle-2 (impulse)
  bar_58 = window[58]  candle-3 (reaction / gap near-edge)
  Column order: [open=0, high=1, low=2, close=3, volume=4]
  TICK = 0.01

All expected values in this file are hand-computed from the synthetic geometry
defined in each fixture.  No mocking of the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.strategy.exits import (
    TICK,
    ExitConfig,
    TradeOutcome,
    compute_exit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(
    bar56_high: float,
    bar56_low: float,
    bar57_high: float,
    bar57_low: float,
    bar58_high: float,
    bar58_low: float,
    swing_high: float = 110.0,
    swing_low: float = 90.0,
) -> np.ndarray:
    """Build a (60, 5) window with known candle prices.

    Bars 0-38  : flat neutral candles with high=swing_high, low=swing_low
                 (so swing_lookback=20 covers bars 39-58, and bars 39-55 are
                 controlled via the swing_* params — we set them to force a
                 specific swing level).
    Bars 39-55 : highs = swing_high, lows = swing_low  (lookback window,
                 non-critical bars)
    Bar 56     : candle-1
    Bar 57     : candle-2 (impulse)
    Bar 58     : candle-3 (reaction)
    """
    window = np.zeros((60, 5), dtype=np.float64)
    # neutral filler
    for i in range(60):
        window[i] = [100.0, swing_high, swing_low, 100.0, 1000.0]
    # candle-1
    window[56] = [bar56_low, bar56_high, bar56_low, bar56_high, 1000.0]
    # candle-2
    window[57] = [bar57_low, bar57_high, bar57_low, bar57_high, 1000.0]
    # candle-3
    window[58] = [bar58_low, bar58_high, bar58_low, bar58_high, 1000.0]
    return window


def _make_future(*bars: tuple[float, float, float, float]) -> np.ndarray:
    """Build future OHLCV array from (open, high, low, close) tuples."""
    arr = np.zeros((len(bars), 5), dtype=np.float64)
    for i, (o, h, l, c) in enumerate(bars):
        arr[i] = [o, h, l, c, 1000.0]
    return arr


# ---------------------------------------------------------------------------
# V1 — fixed_2r
# ---------------------------------------------------------------------------

class TestFixed2R:
    """V1: market entry at future[0].open; SL at bar_56 gap edge; TP = entry +/- 2R."""

    # Bull geometry:
    #   bar_56: high=100, low=95    => SL = 100 - 0.01 = 99.99
    #   future[0].open = 103        => entry = 103
    #   risk = 103 - 99.99 = 3.01
    #   tp = 103 + 2 * 3.01 = 109.02
    BULL_WINDOW = _make_window(100, 95, 105, 98, 102, 101)
    BULL_SL = 100.0 - TICK   # 99.99
    BULL_ENTRY = 103.0
    BULL_RISK = BULL_ENTRY - BULL_SL  # 3.01
    BULL_TP = BULL_ENTRY + 2.0 * BULL_RISK  # 109.02

    # Bear geometry:
    #   bar_56: high=105, low=100   => SL = 100 + 0.01 = 100.01
    #   future[0].open = 98         => entry = 98
    #   risk = 100.01 - 98 = 2.01
    #   tp = 98 - 2 * 2.01 = 93.98
    BEAR_WINDOW = _make_window(105, 100, 103, 98, 104, 101)
    BEAR_SL = 100.0 + TICK   # 100.01
    BEAR_ENTRY = 98.0
    BEAR_RISK = BEAR_SL - BEAR_ENTRY  # 2.01
    BEAR_TP = BEAR_ENTRY - 2.0 * BEAR_RISK  # 93.98

    def test_bull_entry_sl_tp(self):
        """V1 bull: entry = future[0].open, SL = bar_56.high - TICK, TP = entry+2R."""
        future = _make_future((self.BULL_ENTRY, 104, 102, 103))
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.entry == pytest.approx(self.BULL_ENTRY)
        assert result.sl == pytest.approx(self.BULL_SL)
        assert result.tp == pytest.approx(self.BULL_TP)
        assert result.filled is True
        assert result.swing_fallback is False

    def test_bear_entry_sl_tp(self):
        """V1 bear: entry = future[0].open, SL = bar_56.low + TICK, TP = entry-2R."""
        future = _make_future((self.BEAR_ENTRY, 99, 97, 98))
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("fixed_2r"))
        assert result.entry == pytest.approx(self.BEAR_ENTRY)
        assert result.sl == pytest.approx(self.BEAR_SL)
        assert result.tp == pytest.approx(self.BEAR_TP)
        assert result.filled is True

    def test_bull_tp_hit(self):
        """V1 bull: future bar that touches TP => outcome=tp, r_multiple=2.0."""
        # bar0: no hit, bar1: high >= tp
        future = _make_future(
            (self.BULL_ENTRY, 106, 102, 104),      # bar0: no hit
            (105, self.BULL_TP + 0.5, 104, 105),   # bar1: TP hit
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.outcome == "tp"
        assert result.r_multiple == pytest.approx(2.0)
        assert result.exit_idx_in_future == 1

    def test_bull_sl_hit(self):
        """V1 bull: future bar that touches SL => outcome=sl, r_multiple=-1.0."""
        future = _make_future(
            (self.BULL_ENTRY, 104, self.BULL_SL - 0.5, 103),  # SL hit on bar0
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.outcome == "sl"
        assert result.r_multiple == pytest.approx(-1.0)
        assert result.exit_idx_in_future == 0

    def test_bear_tp_hit(self):
        """V1 bear: TP hit => outcome=tp, r_multiple=2.0."""
        future = _make_future(
            (self.BEAR_ENTRY, 99, 96, 97),           # bar0: no hit
            (96, 97, self.BEAR_TP - 0.5, 96),        # bar1: TP hit (low <= tp)
        )
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("fixed_2r"))
        assert result.outcome == "tp"
        assert result.r_multiple == pytest.approx(2.0)

    def test_bear_sl_hit(self):
        """V1 bear: SL hit => outcome=sl, r_multiple=-1.0."""
        future = _make_future(
            (self.BEAR_ENTRY, self.BEAR_SL + 0.5, 97, 98),  # SL hit bar0
        )
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("fixed_2r"))
        assert result.outcome == "sl"
        assert result.r_multiple == pytest.approx(-1.0)

    def test_undecided_no_hit(self):
        """V1: neither TP nor SL hit in future => outcome=undecided."""
        mid = (self.BULL_SL + self.BULL_TP) / 2
        future = _make_future(
            (self.BULL_ENTRY, mid, mid, mid),
            (mid, mid, mid, mid),
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.outcome == "undecided"

    def test_no_future(self):
        """V1: empty future => outcome=no_future, filled=False."""
        future = np.empty((0, 4), dtype=np.float64)
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.outcome == "no_future"
        assert result.filled is False
        # SL is still computed
        assert result.sl == pytest.approx(self.BULL_SL)

    def test_tie_bar_tp_and_sl_same_bar_gives_sl(self):
        """Tie: bar where both TP and SL are touched => SL (conservative)."""
        future = _make_future(
            (self.BULL_ENTRY, self.BULL_TP + 1, self.BULL_SL - 1, 105),  # both hit
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.outcome == "sl"
        assert result.r_multiple == pytest.approx(-1.0)

    def test_v1_sl_anchored_to_bar56_not_bar57(self):
        """REVERT-SENSITIVE: V1 SL must use bar_56.high, NOT bar_57.high.
        If SL were computed from bar_57 the value would differ.
        """
        # bar_56.high = 100, bar_57.high = 120 (very different)
        window = _make_window(100, 95, 120, 115, 102, 101)
        expected_sl = 100.0 - TICK   # from bar_56 only
        wrong_sl = 120.0 - TICK      # would be bar_57

        future = _make_future((103.0, 106, 102, 104))
        result = compute_exit(window, future, 1, ExitConfig("fixed_2r"))
        assert result.sl == pytest.approx(expected_sl), (
            "V1 SL must come from bar_56.high; got bar_57 value instead"
        )
        assert result.sl != pytest.approx(wrong_sl)


# ---------------------------------------------------------------------------
# V2 — ict_iofed
# ---------------------------------------------------------------------------

class TestIctIofed:
    """V2: limit at bar_58 near-edge; SL at bar_56 far edge (candle-1, wide)."""

    # Bull geometry:
    #   bar_56: high=100, low=95    gap_high = bar_56.high = 100
    #   bar_57: high=108, low=96
    #   bar_58: high=103, low=101   gap_low = bar_58.low = 101
    #   limit = bar_58.low = 101
    #   SL = bar_56.low - TICK = 95 - 0.01 = 94.99
    #   swing: bars 39-58 highs = 110 (swing_high fixture default)
    #           swing_tp = 110 > limit=101 → valid
    BULL_WINDOW = _make_window(100, 95, 108, 96, 103, 101, swing_high=110.0)
    BULL_LIMIT = 101.0       # bar_58.low
    BULL_SL = 95.0 - TICK    # bar_56.low - TICK = 94.99
    BULL_SWING_TP = 110.0    # max high bars 39-58

    # Bear geometry:
    #   bar_56: high=105, low=100   gap_low = bar_56.low = 100
    #   bar_57: high=103, low=92
    #   bar_58: high=103, low=99    gap_high = bar_58.high = 103
    #   limit = bar_58.high = 103
    #   SL = bar_56.high + TICK = 105 + 0.01 = 105.01
    #   swing: bars 39-58 lows = 90 (swing_low fixture default)
    #           swing_tp = 90 < limit=103 → valid
    BEAR_WINDOW = _make_window(105, 100, 103, 92, 103, 99, swing_low=90.0)
    BEAR_LIMIT = 103.0       # bar_58.high
    BEAR_SL = 105.0 + TICK   # bar_56.high + TICK = 105.01
    BEAR_SWING_TP = 90.0     # min low bars 39-58

    def test_bull_entry_is_limit_price(self):
        """V2 bull: limit = bar_58.low; entry price = limit on fill."""
        # future[0].low <= limit fills immediately
        future = _make_future((102, 103, self.BULL_LIMIT - 0.1, 102))
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ict_iofed"))
        assert result.entry == pytest.approx(self.BULL_LIMIT)
        assert result.filled is True

    def test_bear_entry_is_limit_price(self):
        """V2 bear: limit = bar_58.high; entry price = limit on fill."""
        future = _make_future((101, self.BEAR_LIMIT + 0.1, 100, 101))
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("ict_iofed"))
        assert result.entry == pytest.approx(self.BEAR_LIMIT)
        assert result.filled is True

    def test_bull_sl_is_bar56_low(self):
        """V2 bull: SL = bar_56.low - TICK (candle-1 far edge)."""
        future = _make_future((102, 103, self.BULL_LIMIT - 0.1, 102))
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ict_iofed"))
        assert result.sl == pytest.approx(self.BULL_SL)

    def test_bear_sl_is_bar56_high(self):
        """V2 bear: SL = bar_56.high + TICK (candle-1 far edge)."""
        future = _make_future((101, self.BEAR_LIMIT + 0.1, 100, 101))
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("ict_iofed"))
        assert result.sl == pytest.approx(self.BEAR_SL)

    def test_bull_swing_tp(self):
        """V2 bull: TP = swing high from bars 39-58 (causal, no future influence).
        Two future bars: bar0 fills, bar1 provides the walk context.
        """
        future = _make_future(
            (102, 103, self.BULL_LIMIT - 0.1, 102),  # bar0: fill
            (105, 108, 104, 107),                     # bar1: no TP hit yet
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ict_iofed"))
        assert result.tp == pytest.approx(self.BULL_SWING_TP)
        assert result.swing_fallback is False

    def test_bear_swing_tp(self):
        """V2 bear: TP = swing low from bars 39-58.
        Two future bars: bar0 fills, bar1 provides the walk context.
        """
        future = _make_future(
            (101, self.BEAR_LIMIT + 0.1, 100, 101),  # bar0: fill
            (101, 102, 101, 101),                     # bar1: no TP hit yet
        )
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("ict_iofed"))
        assert result.tp == pytest.approx(self.BEAR_SWING_TP)
        assert result.swing_fallback is False

    def test_no_fill_within_timeout(self):
        """V2: future never touches limit within timeout => no_fill, filled=False."""
        # Bull limit = 101.0; future bars all stay above 102
        future = _make_future(
            (103, 104, 102, 103),
            (103, 104, 102, 103),
            (103, 104, 102, 103),
        )
        cfg = ExitConfig("ict_iofed", fill_timeout_bars=3)
        result = compute_exit(self.BULL_WINDOW, future, 1, cfg)
        assert result.outcome == "no_fill"
        assert result.filled is False

    def test_fill_on_bar_j_then_walk_from_j_plus_1(self):
        """V2: fill on bar j; walk starts at j+1; outcome reflects j+1 onward."""
        # Bull limit = 101, fill on bar1 (low=100.9 <= 101)
        # bar2: TP hit (high >= swing_tp=110)
        future = _make_future(
            (103, 104, 102, 103),                         # bar0: no fill
            (102, 103, self.BULL_LIMIT - 0.2, 102),      # bar1: fill
            (105, self.BULL_SWING_TP + 1, 104, 105),     # bar2: TP hit
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ict_iofed"))
        assert result.filled is True
        assert result.outcome == "tp"
        # exit_idx_in_future should be 2 (absolute index in future array)
        assert result.exit_idx_in_future == 2

    def test_no_fill_excluded_from_win_rate(self):
        """REVERT-SENSITIVE: no_fill trades must have filled=False (excluded from win-rate/total-R)."""
        future = _make_future(
            (103, 104, 102, 103),  # low=102 > limit=101, no fill
        )
        cfg = ExitConfig("ict_iofed", fill_timeout_bars=1)
        result = compute_exit(self.BULL_WINDOW, future, 1, cfg)
        assert result.filled is False
        assert result.outcome == "no_fill"
        assert result.r_multiple == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# V3 — ce_50pct
# ---------------------------------------------------------------------------

class TestCe50Pct:
    """V3: limit at FVG 50% midpoint (CE); SL at candle-1 far edge (same as V2)."""

    # Bull geometry:
    #   bar_56: high=100, low=95    gap_high = 100
    #   bar_57: high=108, low=96
    #   bar_58: high=103, low=102   gap_low = 102
    #   CE = (100 + 102) / 2 = 101.0
    #   SL = bar_56.low - TICK = 94.99
    #   swing_high = 110
    BULL_WINDOW = _make_window(100, 95, 108, 96, 103, 102, swing_high=110.0)
    BULL_CE = (100.0 + 102.0) / 2   # = 101.0
    BULL_SL = 95.0 - TICK

    # Bear geometry:
    #   bar_56: high=106, low=100   gap_low = 100
    #   bar_57: high=103, low=92
    #   bar_58: high=104, low=99    gap_high = 104
    #   CE = (100 + 104) / 2 = 102.0
    #   SL = bar_56.high + TICK = 106.01
    BEAR_WINDOW = _make_window(106, 100, 103, 92, 104, 99, swing_low=90.0)
    BEAR_CE = (100.0 + 104.0) / 2   # = 102.0
    BEAR_SL = 106.0 + TICK

    def test_bull_limit_is_ce(self):
        """V3 bull: limit = CE midpoint = (gap_high + gap_low) / 2."""
        # fill on bar0: low <= CE=101
        future = _make_future((102, 103, 100.5, 102))
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ce_50pct"))
        assert result.entry == pytest.approx(self.BULL_CE)

    def test_bear_limit_is_ce(self):
        """V3 bear: limit = CE midpoint."""
        # fill on bar0: high >= CE=102
        future = _make_future((101, 102.5, 100, 101))
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("ce_50pct"))
        assert result.entry == pytest.approx(self.BEAR_CE)

    def test_v3_sl_same_as_v2_bull(self):
        """V3 bull: SL uses candle-1 far edge (bar_56.low - TICK), same as V2."""
        future = _make_future((102, 103, 100.5, 102))
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ce_50pct"))
        assert result.sl == pytest.approx(self.BULL_SL)

    def test_v3_entry_differs_from_v2_same_window(self):
        """V3 CE entry differs from V2 near-edge entry on the same window.
        V2 limit = bar_58.low = 102; V3 limit = CE = 101.0 (different).
        """
        # Use a window where CE != gap_low
        # gap_low = bar_58.low = 102, gap_high = bar_56.high = 100 (bull)
        # CE = (100 + 102)/2 = 101 != 102
        future = _make_future((102, 103, 100.5, 102))
        r_v2 = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ict_iofed"))
        r_v3 = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ce_50pct"))
        # V2 entry = bar_58.low = 102, V3 entry = CE = 101
        assert r_v2.entry == pytest.approx(102.0)
        assert r_v3.entry == pytest.approx(101.0)
        assert r_v2.entry != pytest.approx(r_v3.entry)

    def test_no_fill(self):
        """V3: no_fill when future never touches CE."""
        # CE = 101.0; future bars stay at 103+
        future = _make_future(
            (103, 104, 102.5, 103),
            (103, 104, 102.5, 103),
        )
        cfg = ExitConfig("ce_50pct", fill_timeout_bars=2)
        result = compute_exit(self.BULL_WINDOW, future, 1, cfg)
        assert result.outcome == "no_fill"
        assert result.filled is False


# ---------------------------------------------------------------------------
# V4 — tradinglab
# ---------------------------------------------------------------------------

class TestTradinglab:
    """V4: limit at bar_58 near-edge (same as V2); SL at bar_57 impulse (tight)."""

    # Bull geometry:
    #   bar_56: high=100, low=93
    #   bar_57: high=108, low=96    SL = bar_57.low - TICK = 95.99
    #   bar_58: high=103, low=101   limit = bar_58.low = 101
    #   swing_high=110
    BULL_WINDOW = _make_window(100, 93, 108, 96, 103, 101, swing_high=110.0)
    BULL_LIMIT = 101.0
    BULL_SL_V4 = 96.0 - TICK    # bar_57.low - TICK = 95.99
    BULL_SL_V2 = 93.0 - TICK    # bar_56.low - TICK (V2/V3 wide)

    # Bear geometry:
    #   bar_56: high=106, low=99
    #   bar_57: high=104, low=92    SL = bar_57.high + TICK = 104.01
    #   bar_58: high=103, low=99    limit = bar_58.high = 103
    BEAR_WINDOW = _make_window(106, 99, 104, 92, 103, 99, swing_low=90.0)
    BEAR_LIMIT = 103.0
    BEAR_SL_V4 = 104.0 + TICK   # bar_57.high + TICK = 104.01
    BEAR_SL_V2 = 106.0 + TICK   # bar_56.high + TICK (wider)

    def test_bull_sl_is_bar57_low(self):
        """REVERT-SENSITIVE: V4 bull SL = bar_57.low - TICK (impulse, NOT bar_56.low)."""
        future = _make_future((102, 103, self.BULL_LIMIT - 0.1, 102))
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("tradinglab"))
        assert result.sl == pytest.approx(self.BULL_SL_V4), (
            "V4 SL must come from bar_57 (impulse candle); got bar_56 value"
        )
        assert result.sl != pytest.approx(self.BULL_SL_V2)

    def test_bear_sl_is_bar57_high(self):
        """REVERT-SENSITIVE: V4 bear SL = bar_57.high + TICK (impulse, NOT bar_56.high)."""
        future = _make_future((101, self.BEAR_LIMIT + 0.1, 100, 101))
        result = compute_exit(self.BEAR_WINDOW, future, 2, ExitConfig("tradinglab"))
        assert result.sl == pytest.approx(self.BEAR_SL_V4), (
            "V4 SL must come from bar_57.high; got bar_56 value"
        )
        assert result.sl != pytest.approx(self.BEAR_SL_V2)

    def test_v4_limit_same_as_v2(self):
        """V4 limit entry = bar_58 near-edge (same as V2, NOT CE)."""
        future = _make_future((102, 103, self.BULL_LIMIT - 0.1, 102))
        r_v2 = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ict_iofed"))
        r_v4 = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("tradinglab"))
        # Both V2 and V4 use bar_58.low as limit
        assert r_v4.entry == pytest.approx(self.BULL_LIMIT)
        assert r_v4.entry == pytest.approx(r_v2.entry)

    def test_v4_limit_differs_from_market_entry(self):
        """REVERT-SENSITIVE: V4 uses limit entry, NOT market (future[0].open)."""
        # Make future[0].open very different from limit to detect if market used
        open_price = 107.0   # far from limit=101
        future = _make_future(
            (open_price, 108, self.BULL_LIMIT - 0.1, 107),  # fills on bar0 (low <= limit)
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("tradinglab"))
        # If market entry: entry = 107. If limit entry: entry = 101.
        assert result.entry == pytest.approx(self.BULL_LIMIT), (
            "V4 must use limit entry (bar_58.low), not future[0].open"
        )
        assert result.entry != pytest.approx(open_price)

    def test_v4_sl_tighter_than_v2(self):
        """V4 SL is tighter (closer to entry) than V2 SL on the same bull window."""
        future = _make_future((102, 103, self.BULL_LIMIT - 0.1, 102))
        r_v2 = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("ict_iofed"))
        r_v4 = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("tradinglab"))
        # V4 SL (95.99) > V2 SL (92.99) for bull: tighter means higher SL
        assert r_v4.sl > r_v2.sl

    def test_v4_tp_hit(self):
        """V4: TP hit after fill => outcome=tp."""
        swing_tp = 110.0
        future = _make_future(
            (102, 103, self.BULL_LIMIT - 0.1, 102),   # bar0: fill
            (105, swing_tp + 1, 104, 105),              # bar1: TP hit
        )
        result = compute_exit(self.BULL_WINDOW, future, 1, ExitConfig("tradinglab"))
        assert result.outcome == "tp"
        assert result.filled is True


# ---------------------------------------------------------------------------
# Swing TP causality
# ---------------------------------------------------------------------------

class TestSwingTpCausality:
    """Swing TP must depend only on window bars 39-58, never on future bars."""

    def test_swing_tp_unaffected_by_future_bars(self):
        """REVERT-SENSITIVE: changing future OHLC bars must NOT change TP level.

        If swing-TP used any future bar, altering future prices would shift TP.
        """
        # Bull window: swing bars 39-55 set via fixture (high=110 default)
        window = _make_window(100, 95, 108, 96, 103, 101, swing_high=110.0)
        cfg = ExitConfig("ict_iofed")

        # Fill on bar0 in both cases
        future_a = _make_future(
            (102, 103, 100.5, 102),   # bar0: fill
            (105, 112, 104, 111),     # bar1: higher high (200 would shift TP if lookahead)
        )
        future_b = _make_future(
            (102, 103, 100.5, 102),   # bar0: fill
            (105, 108, 104, 107),     # bar1: lower high
        )

        result_a = compute_exit(window, future_a, 1, cfg)
        result_b = compute_exit(window, future_b, 1, cfg)

        assert result_a.tp == pytest.approx(result_b.tp), (
            "TP changed when future bars changed — indicates lookahead leak"
        )
        assert result_a.tp == pytest.approx(110.0)

    def test_swing_tp_uses_correct_lookback_window(self):
        """Swing TP = max high over window[39:59] for bull (swing_lookback=20).

        Bars 39-55 have high=swing_high=108. bar_56.high=100, bar_57.high=105,
        bar_58.high=103.  Max high in [39:59] = max(108, 100, 105, 103) = 108.
        Two future bars needed: bar0 fills, bar1 provides walk context.
        """
        window = _make_window(100, 95, 105, 96, 103, 101, swing_high=108.0)
        future = _make_future(
            (102, 103, 100.5, 102),  # bar0: fill (low=100.5 <= limit=101)
            (104, 106, 103, 105),    # bar1: walk — no TP/SL hit
        )
        result = compute_exit(window, future, 1, ExitConfig("ict_iofed"))
        assert result.tp == pytest.approx(108.0)


# ---------------------------------------------------------------------------
# Swing fallback
# ---------------------------------------------------------------------------

class TestSwingFallback:
    """When swing level is not beyond entry, fallback to entry +/- 2R."""

    def test_bull_swing_fallback_when_swing_below_entry(self):
        """Bull: swing_tp <= entry => fallback to entry + 2R, swing_fallback=True.

        Geometry chosen so ALL highs in window[39:59] are below the fill limit:
          bars 39-55: high=98 (swing_high=98)
          bar_56: high=98, low=95
          bar_57: high=98, low=96
          bar_58: high=99, low=101  (limit = bar_58.low = 101)

        swing_tp = max high in [39:59] = max(98x17, 98, 98, 99) = 99 < limit=101 → fallback.
        SL = bar_56.low - TICK = 95 - 0.01 = 94.99
        risk = 101 - 94.99 = 6.01
        fallback TP = 101 + 2 * 6.01 = 113.02
        """
        window = _make_window(98, 95, 98, 96, 99, 101, swing_high=98.0)
        future = _make_future(
            (102, 103, 100.5, 102),  # bar0: fill (low=100.5 <= limit=101)
            (104, 106, 103, 105),    # bar1: walk — no TP/SL hit
        )
        result = compute_exit(window, future, 1, ExitConfig("ict_iofed"))
        assert result.swing_fallback is True
        sl = 95.0 - TICK
        risk = 101.0 - sl
        expected_tp = 101.0 + 2.0 * risk
        assert result.tp == pytest.approx(expected_tp)

    def test_bear_swing_fallback_when_swing_above_entry(self):
        """Bear: swing_tp >= entry => fallback to entry - 2R, swing_fallback=True.

        Geometry chosen so ALL lows in window[39:59] are above the fill limit:
          bars 39-55: low=115 (swing_low=115)
          bar_56: high=116, low=112
          bar_57: high=114, low=114
          bar_58: high=108, low=109  (limit = bar_58.high = 108)

        swing_tp = min low in [39:59] = min(115x17, 112, 114, 109) = 109 > limit=108 → fallback.
        SL = bar_56.high + TICK = 116 + 0.01 = 116.01
        risk = SL - limit = 116.01 - 108 = 8.01
        fallback TP = 108 - 2 * 8.01 = 91.98
        """
        window = _make_window(116, 112, 114, 114, 108, 109, swing_low=115.0)
        future = _make_future(
            (110, 109, 108.5, 109),  # bar0: fill (high=109 >= limit=108)
            (109, 109, 109, 109),    # bar1: walk — no TP/SL hit
        )
        result = compute_exit(window, future, 2, ExitConfig("ict_iofed"))
        assert result.swing_fallback is True
        sl = 116.0 + TICK
        risk = sl - 108.0
        expected_tp = 108.0 - 2.0 * risk
        assert result.tp == pytest.approx(expected_tp)

    def test_no_fallback_when_swing_valid(self):
        """swing_fallback=False when swing level is properly beyond entry.

        Two future bars: bar0 fills, bar1 provides walk context.
        """
        window = _make_window(100, 95, 108, 96, 103, 101, swing_high=115.0)
        future = _make_future(
            (102, 103, 100.5, 102),  # bar0: fill
            (104, 106, 103, 105),    # bar1: walk
        )
        result = compute_exit(window, future, 1, ExitConfig("ict_iofed"))
        assert result.swing_fallback is False
        assert result.tp == pytest.approx(115.0)


# ---------------------------------------------------------------------------
# no_future after fill
# ---------------------------------------------------------------------------

class TestNoFutureAfterFill:
    """Limit variant: filled but no bars remain after fill bar => no_future."""

    def test_no_future_after_fill(self):
        """V2: fill on last available bar => no_future (no walk bars)."""
        window = _make_window(100, 95, 108, 96, 103, 101, swing_high=110.0)
        # Only one bar in future, which also triggers fill — no bars remain for walk
        future = _make_future(
            (102, 103, 100.5, 102),  # bar0: fill (low=100.5 <= limit=101)
        )
        result = compute_exit(window, future, 1, ExitConfig("ict_iofed"))
        assert result.outcome == "no_future"
        assert result.filled is True  # it DID fill
        assert result.entry == pytest.approx(101.0)


# ---------------------------------------------------------------------------
# Entry-price differentiation across variants (same window)
# ---------------------------------------------------------------------------

class TestEntryRuleDiffers:
    """All four variants compute entry from different logic on the same window."""

    # Geometry chosen so V1/V2/V3/V4 all have clearly different entry prices:
    #   bar_56: high=100, low=94
    #   bar_57: high=108, low=97
    #   bar_58: high=103, low=101
    #   V1  market: entry = future[0].open = 105 (arbitrary market price)
    #   V2  limit:  entry = bar_58.low = 101
    #   V3  ce:     entry = CE = (100+101)/2 = 100.5
    #   V4  limit:  entry = bar_58.low = 101 (same as V2)
    WINDOW = _make_window(100, 94, 108, 97, 103, 101, swing_high=115.0)
    V1_OPEN = 105.0
    V2_V4_LIMIT = 101.0   # bar_58.low
    V3_CE = (100.0 + 101.0) / 2  # 100.5

    def _future_fill(self):
        # Low on bar0 <= 101 so V2/V3/V4 all fill immediately
        return _make_future((self.V1_OPEN, 106, 100.5, 105))

    def test_v1_entry_is_market_open(self):
        result = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("fixed_2r"))
        assert result.entry == pytest.approx(self.V1_OPEN)

    def test_v2_entry_is_near_edge(self):
        result = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("ict_iofed"))
        assert result.entry == pytest.approx(self.V2_V4_LIMIT)

    def test_v3_entry_is_ce(self):
        result = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("ce_50pct"))
        assert result.entry == pytest.approx(self.V3_CE)

    def test_v4_entry_is_near_edge(self):
        result = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("tradinglab"))
        assert result.entry == pytest.approx(self.V2_V4_LIMIT)

    def test_all_four_entries_are_distinct_or_matched_correctly(self):
        """V1 != V2 == V4 != V3."""
        r1 = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("fixed_2r"))
        r2 = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("ict_iofed"))
        r3 = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("ce_50pct"))
        r4 = compute_exit(self.WINDOW, self._future_fill(), 1, ExitConfig("tradinglab"))

        assert r1.entry != pytest.approx(r2.entry)
        assert r2.entry == pytest.approx(r4.entry)
        assert r3.entry != pytest.approx(r2.entry)
        assert r3.entry != pytest.approx(r1.entry)


# ---------------------------------------------------------------------------
# ExitConfig validation
# ---------------------------------------------------------------------------

class TestExitConfig:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ExitConfig("bad_strategy")

    def test_valid_strategies_accepted(self):
        for s in ("fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"):
            cfg = ExitConfig(s)
            assert cfg.strategy == s
