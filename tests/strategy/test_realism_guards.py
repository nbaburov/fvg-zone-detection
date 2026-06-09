"""test_realism_guards.py — Tests for the three FVG exit-sim realism guards.

All expected values are hand-computed from synthetic geometry.
No mocking of code under test.  Deterministic, CPU, no network.

Guard 1 — ATR min-stop floor  (_compute_atr, _apply_atr_floor, min_stop_atr_k)
Guard 2 — Cost drag           (slippage_ticks, commission_per_share → after_cost_r)
Guard 3 — Realism reporting   (report_realism=True → 5 extra keys in summarise_trades)

Candle indexing (from exits.py docstring):
  bar_56 = window[56]  candle-1
  bar_57 = window[57]  candle-2 (impulse)
  bar_58 = window[58]  candle-3 (reaction)
  Column order: [open=0, high=1, low=2, close=3, volume=4]
  TICK = 0.01
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.exits import (
    TICK,
    ExitConfig,
    TradeOutcome,
    _compute_atr,
    compute_exit,
)
from src.inspect.outcomes import summarise_trades


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_window(
    bar56_high: float,
    bar56_low: float,
    bar57_high: float,
    bar57_low: float,
    bar58_high: float,
    bar58_low: float,
    base_price: float = 400.0,
    base_tr: float = 1.0,
) -> np.ndarray:
    """Build a (60, 5) window.

    Bars 0–55: each bar has open=close=base_price, high=base_price+base_tr/2,
    low=base_price-base_tr/2, and previous close = base_price, so each TR
    (true high - true low) = base_tr.  Bars 39–55 also follow this pattern.
    Bar 56: candle-1, bar 57: candle-2 (impulse), bar 58: candle-3.
    """
    window = np.zeros((60, 5), dtype=np.float64)
    half = base_tr / 2.0
    for i in range(60):
        o = base_price
        h = base_price + half
        lo = base_price - half
        c = base_price
        window[i] = [o, h, lo, c, 1000.0]
    window[56] = [bar56_low, bar56_high, bar56_low, bar56_high, 1000.0]
    window[57] = [bar57_low, bar57_high, bar57_low, bar57_high, 1000.0]
    window[58] = [bar58_low, bar58_high, bar58_low, bar58_high, 1000.0]
    return window


def _make_future(*bars: tuple[float, float, float, float]) -> np.ndarray:
    arr = np.zeros((len(bars), 5), dtype=np.float64)
    for i, (o, h, lo, c) in enumerate(bars):
        arr[i] = [o, h, lo, c, 1000.0]
    return arr


def _make_trade(
    outcome: str,
    direction: int = 1,
    r_multiple: float = 0.0,
    filled: bool = True,
    entry: float = 400.0,
    sl: float = 399.0,
) -> TradeOutcome:
    return TradeOutcome(
        window_idx=0,
        entry_ts=pd.NaT,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=402.0,
        outcome=outcome,
        exit_idx_in_future=-1,
        exit_ts=None,
        exit_price=entry,
        r_multiple=r_multiple,
        filled=filled,
        swing_fallback=False,
    )


# ---------------------------------------------------------------------------
# Guard 1 — ATR causality and correctness
# ---------------------------------------------------------------------------

class TestComputeAtr:
    """_compute_atr correctness and causality guarantees."""

    def test_known_trs_equal_hand_computed_mean(self):
        """Flat window with uniform TR=2.0 → ATR = 2.0.

        Hand-computed: each of bars 1–59 has high=base+1, low=base-1, and
        prev_close=base (from bar before), so TR=high-low=2.0 for all 59 bars.
        mean(TR) = 2.0.
        """
        base = 400.0
        base_tr = 2.0
        window = _make_window(
            bar56_high=base + 1, bar56_low=base - 1,
            bar57_high=base + 1, bar57_low=base - 1,
            bar58_high=base + 1, bar58_low=base - 1,
            base_price=base,
            base_tr=base_tr,
        )
        # All bars have high=401, low=399, prev_close=400 → TR=2 for all 59 bars
        # Bar 56,57,58 also have high=401, low=399 → TR=2
        atr = _compute_atr(window)
        assert atr == pytest.approx(2.0, abs=1e-9)

    def test_mixed_trs_hand_computed(self):
        """Window with varying TR values: verify mean is correct.

        Bars 0–55: base_tr=1.0 (high=400.5, low=399.5, TR=1.0 for each).
        Bar 56: high=405, low=395 → prev_close=400, TR=max(405,400)-min(395,400)=10.
        Bar 57: high=410, low=395 → prev_close=405 (bar56 close=405), TR=max(410,405)-min(395,405)=15.
        Bar 58: high=402, low=398 → prev_close=410, TR=max(402,410)-min(398,410)=12.

        TR values: 55 bars with TR=1.0 (bars 1..55), then bar56 TR=10, bar57 TR=15, bar58 TR=12.
        Wait — we need exact TR calc. Let me lay it out carefully:
        - bar[i] close = bar[i] high (by _make_window construction: c=high for bars 56-58)
        - bar 56: high=405, low=395, close=405; prev_close=bar[55].close=400; TR=max(405,400)-min(395,400)=405-395=10
        - bar 57: high=410, low=395, close=410; prev_close=405; TR=max(410,405)-min(395,405)=410-395=15
        - bar 58: high=402, low=398, close=402; prev_close=410; TR=max(402,410)-min(398,410)=410-398=12
        - bars 1..55: high=400.5, low=399.5, prev_close=400 (bar before each is same pattern)
          TR=max(400.5,400)-min(399.5,400)=400.5-399.5=1.0 (55 bars)
        - bar 56 is index 56: its TR is computed using bars[1:] highs and bars[:-1] closes.
          highs = window[1:,1], lows = window[1:,2], prev_closes = window[:-1,3]
          indices of highs/lows are 1..59 (59 values).
          index 0 in the TR array = bar index 1 in window.
          index 55 in the TR array = bar index 56 (bar56).
        Total bars contributing: 59.
        Sum = 55*1.0 + 10 + 15 + 12 = 55 + 37 = 92
        Mean = 92/59
        """
        base = 400.0
        window = _make_window(
            bar56_high=405.0, bar56_low=395.0,
            bar57_high=410.0, bar57_low=395.0,
            bar58_high=402.0, bar58_low=398.0,
            base_price=base,
            base_tr=1.0,
        )
        # bar57 close = bar57_high = 410 (set as close in _make_window for bars 56-58)
        # Recompute: bar57 c=bar57_high=410, bar58 prev_close=410
        # TR at bar57 index in array (index 57): prev_close = bar56.close = 405
        # TR at bar58 index in array (index 58): prev_close = bar57.close = 410
        # bar57 TR: true_high=max(410,405)=410, true_low=min(395,405)=395, TR=15
        # bar58 TR: true_high=max(402,410)=410, true_low=min(398,410)=398, TR=12
        # bars 1..55: high=400.5, low=399.5, prev_close=400 → TR=1.0 (55 values)
        # bar56 TR=10, bar57 TR=15, bar58 TR=12
        # bar59: high=400.5, low=399.5, prev_close=bar58.close=402 →
        #   true_high=max(400.5,402)=402, true_low=min(399.5,402)=399.5, TR=2.5
        # Total: 55*1 + 10 + 15 + 12 + 2.5 = 94.5; n=59; mean=94.5/59
        expected = 94.5 / 59.0
        atr = _compute_atr(window)
        assert atr == pytest.approx(expected, rel=1e-9)

    def test_atr_causality_future_bars_irrelevant(self):
        """CAUSAL: changing future bars (after window) never changes ATR.

        _compute_atr only uses window_raw (the 60-bar input).  This test
        verifies that modifying what would be future data (a separate array)
        doesn't affect ATR — proving it is causal and operates only on window_raw.
        """
        window = _make_window(
            bar56_high=405.0, bar56_low=395.0,
            bar57_high=410.0, bar57_low=395.0,
            bar58_high=402.0, bar58_low=398.0,
            base_price=400.0, base_tr=1.0,
        )
        atr_baseline = _compute_atr(window)

        # Modify a copy of window at bar 59 (the boundary) — this is still
        # within window_raw but bar 59 is only a prev_close for nothing in
        # our TR calc (bars 1..59 use prev_closes[0..58] = window[0..58]).
        # The real test: ATR is computed ONLY on window_raw, not on any future
        # data. We confirm by making entirely different future arrays and
        # verifying the same window always gives same ATR.
        future_a = _make_future((400.0, 410.0, 390.0, 400.0))
        future_b = _make_future((300.0, 500.0, 100.0, 300.0))

        result_a = compute_exit(window, future_a, 1, ExitConfig("fixed_2r",
                                                                  min_stop_atr_k=0.1))
        result_b = compute_exit(window, future_b, 1, ExitConfig("fixed_2r",
                                                                  min_stop_atr_k=0.1))
        # Both have same window → same ATR → same SL floor
        assert result_a.sl == pytest.approx(result_b.sl)

        # Also directly confirm _compute_atr is insensitive to future
        atr_again = _compute_atr(window)
        assert atr_again == pytest.approx(atr_baseline)

    def test_flat_window_atr_zero_no_crash(self):
        """Degenerate flat window (all prices equal → TR=0) returns ATR=0.0 and no crash."""
        # Build a fully flat window: every bar identical
        window = np.full((60, 5), 400.0, dtype=np.float64)
        atr = _compute_atr(window)
        assert atr == 0.0

    def test_normalised_window_raises(self):
        """_compute_atr raises ValueError on normalised (max<=1) window."""
        window = np.full((60, 5), 0.5, dtype=np.float64)
        with pytest.raises(ValueError, match="normalised"):
            _compute_atr(window)

    def test_flat_window_floor_skipped_no_crash(self):
        """ATR=0 from flat window → floor is skipped; compute_exit does not crash."""
        window = np.full((60, 5), 400.0, dtype=np.float64)
        # bar56 high=400 for bull SL; entry will be future[0].open
        future = _make_future((401.0, 410.0, 399.0, 405.0))
        # With ATR=0 the floor guard should skip silently
        cfg = ExitConfig("fixed_2r", min_stop_atr_k=0.5)
        result = compute_exit(window, future, 1, cfg)
        # Should not crash and should return a valid outcome
        assert result.outcome in {"tp", "sl", "undecided", "no_future"}


# ---------------------------------------------------------------------------
# Guard 1 — floor widens micro-stop and changes outcome
# ---------------------------------------------------------------------------

class TestAtrFloorChangesOutcome:
    """The ATR floor must be applied BEFORE the walk so it changes the outcome.

    Geometry (bull trade):
      base_price = 400.0, base_tr = 2.0
      → ATR ≈ 2.0 (mean TR over 59 bars, all ~2.0)

      bar_56: high=400.10, low=399.90  (tiny candle)
      bar_57: high=401.00, low=399.50
      bar_58: high=400.50, low=399.80

      V1 bull: entry = future[0].open; SL = bar56.high - TICK = 400.10 - 0.01 = 400.09

      With entry = 401.00 (open of N+2 bar):
        raw_stop_distance = entry - SL = 401.00 - 400.09 = 0.91

      ATR ≈ 2.0.  With k=1.0, floor_dist = 1.0*2.0 = 2.0.
      raw_dist (0.91) < floor_dist (2.0) → SL floored to entry - 2.0 = 399.00.

      Future path:
        bar0: open=401.00, high=401.50, low=400.50, close=401.20 — no hit for tight SL
              but would hit tight SL (400.09) only if low<400.09 (it doesn't here)
        bar1: open=401.20, high=401.80, low=400.50 — no SL hit
        bar2: high=404.00 → TP with tight SL (TP=401 + 2*(401-400.09)=402.82);
              TP with floored SL (TP=401 + 2*(401-399)=405.00) — NOT hit at 404.00

    Wait, we need a path that:
      - With tight SL (400.09): outcome = sl (some bar goes low below 400.09)
      - With floored SL (399.00): outcome = tp (SL not hit, TP hit)

    Revised future path:
        bar0: open=401.00, high=401.50, low=400.05, close=401.0 → low<400.09 → tight SL hit
        bar1: open=401.00, high=405.00, low=400.50 → floored TP hit (TP=405.00)

    With tight SL (400.09): bar0 hits sl (low=400.05 <= 400.09). outcome="sl".
    With floored SL (399.00): bar0 low=400.05 > 399.00 → no SL. bar1 high=405.00 >= 405.00 → TP hit.
    """

    BASE = 400.0
    BASE_TR = 2.0
    # Tiny candle-1: high just above entry
    BAR56_HIGH = 400.10
    BAR56_LOW = 399.90

    def _window(self) -> np.ndarray:
        return _make_window(
            bar56_high=self.BAR56_HIGH, bar56_low=self.BAR56_LOW,
            bar57_high=401.00, bar57_low=399.50,
            bar58_high=400.50, bar58_low=399.80,
            base_price=self.BASE,
            base_tr=self.BASE_TR,
        )

    def _future(self) -> np.ndarray:
        # bar0: low=400.05 hits tight SL (400.09) but not floored SL (399.00)
        # bar1: high=405.00 hits floored TP (entry + 2*floor_risk = 401+2*2 = 405.00)
        return _make_future(
            (401.00, 401.50, 400.05, 401.00),   # bar0
            (401.00, 405.00, 400.50, 404.50),   # bar1
            (404.00, 405.00, 403.50, 404.50),   # bar2 (filler)
        )

    def test_floor_off_tight_sl_gives_sl_outcome(self):
        """Without floor: tight SL (400.09) is hit at bar0. REVERT-SENSITIVE."""
        window = self._window()
        future = self._future()
        cfg = ExitConfig("fixed_2r")  # no floor
        result = compute_exit(window, future, 1, cfg)

        # Verify geometry: entry=401.00, SL=400.10-0.01=400.09
        assert result.entry == pytest.approx(401.00)
        assert result.sl == pytest.approx(self.BAR56_HIGH - TICK)  # 400.09
        # bar0 low=400.05 <= 400.09 → sl
        assert result.outcome == "sl"
        assert result.r_multiple == pytest.approx(-1.0)

    def test_floor_on_wide_sl_flips_to_tp_outcome(self):
        """With floor (k=1.0, ATR≈2.0): SL widens to 399.00, TP flips to tp. REVERT-SENSITIVE.

        This is the critical correctness property: the floor is applied BEFORE
        the walk.  If applied after (or only to R denominator), outcome would
        still be 'sl' here.
        """
        window = self._window()
        future = self._future()
        cfg = ExitConfig("fixed_2r", min_stop_atr_k=1.0)
        result = compute_exit(window, future, 1, cfg)

        # entry=401.00; ATR≈1.947; floor_dist≈1.947; floored SL ≈ 399.05
        # TP = 401 + 2*(1.947) ≈ 404.89; bar1 high=405.0 >= TP → tp hit
        assert result.entry == pytest.approx(401.00)
        # SL is below 400.05 (bar0 low) so SL is NOT hit; exact value is ATR-derived
        assert result.sl < 400.05, f"Floored SL {result.sl} must be below bar0 low (400.05) to avoid SL hit"
        # bar0 low=400.05 > floored SL → not SL
        # bar1 high=405.00 >= TP (401 + 2*ATR) → tp
        assert result.outcome == "tp"
        # r_multiple = tp_rr = 2.0 (V1 fixed_2r always returns 2.0 on TP)
        assert result.r_multiple == pytest.approx(2.0)

    def test_floor_r_multiple_is_realistic(self):
        """Floored trade: r_multiple=2.0 (tp_rr), which is realistic, not astronomical."""
        window = self._window()
        future = self._future()
        cfg = ExitConfig("fixed_2r", min_stop_atr_k=1.0)
        result = compute_exit(window, future, 1, cfg)
        # Should be 2.0 (fixed_2r always returns tp_rr on TP, not raw R)
        assert abs(result.r_multiple) <= 10.0, "r_multiple should be realistic"


class TestAtrFloorNonBinding:
    """Floor must NOT change outcome when raw stop is already wide."""

    def test_wide_stop_floor_nonbinding(self):
        """Normal wide stop (> k*ATR) → floor does not change SL or outcome.

        Geometry:
          base_tr=2.0 → ATR≈2.0; k=0.5 → floor_dist=1.0
          bar56_high=410.0, bull SL = 410.0 - 0.01 = 409.99
          entry = 415.0 (future[0].open)
          raw_dist = 415.0 - 409.99 = 5.01 > floor_dist=1.0 → floor non-binding
        """
        window = _make_window(
            bar56_high=410.0, bar56_low=405.0,
            bar57_high=415.0, bar57_low=408.0,
            bar58_high=413.0, bar58_low=411.0,
            base_price=400.0, base_tr=2.0,
        )
        future = _make_future(
            (415.0, 416.0, 414.5, 415.5),  # bar0: no hit
            (415.5, 430.0, 414.5, 425.0),  # bar1: TP hit
        )

        cfg_no_floor = ExitConfig("fixed_2r")
        cfg_floor = ExitConfig("fixed_2r", min_stop_atr_k=0.5)

        result_no_floor = compute_exit(window, future, 1, cfg_no_floor)
        result_floor = compute_exit(window, future, 1, cfg_floor)

        # SL must be identical
        assert result_no_floor.sl == pytest.approx(result_floor.sl)
        # Outcomes must be identical
        assert result_no_floor.outcome == result_floor.outcome
        assert result_no_floor.r_multiple == pytest.approx(result_floor.r_multiple)


# ---------------------------------------------------------------------------
# Guard 2 — cost drag math
# ---------------------------------------------------------------------------

class TestCostDrag:
    """Cost drag is applied to filled trades; excluded from no_fill/no_future."""

    def _make_filled_trade(
        self,
        outcome: str,
        r_multiple: float,
        entry: float = 400.0,
        sl: float = 399.0,
    ) -> TradeOutcome:
        return _make_trade(
            outcome=outcome,
            r_multiple=r_multiple,
            filled=True,
            entry=entry,
            sl=sl,
        )

    def test_cost_drag_math_tp_trade(self):
        """after_cost_r = r_multiple - (slippage + commission) / risk_dollars. REVERT-SENSITIVE.

        Trade: entry=400.0, sl=399.0, risk_dollars=1.0, r_multiple=2.0.
        slippage_ticks=2 → slippage_cost = 2 * 0.01 * 2 (round-trip) = 0.04
        commission_per_share=0.01 → commission_cost = 0.01 * 2 = 0.02
        total_cost_dollars = 0.06
        cost_drag_r = 0.06 / 1.0 = 0.06
        after_cost_r = 2.0 - 0.06 = 1.94
        """
        trade = self._make_filled_trade("tp", r_multiple=2.0, entry=400.0, sl=399.0)
        cfg = ExitConfig(
            "fixed_2r",
            slippage_ticks=2,
            commission_per_share=0.01,
            report_realism=True,
        )
        summary = summarise_trades([trade], realism_config=cfg)

        # Hand-computed: slippage=2*0.01*2=0.04; commission=0.01*2=0.02; total=0.06/1.0
        expected_after_cost = 2.0 - 0.06
        assert summary["after_cost_total_r"] == pytest.approx(expected_after_cost, abs=1e-9)
        assert summary["after_cost_avg_r"] == pytest.approx(expected_after_cost, abs=1e-9)

    def test_cost_drag_sl_trade(self):
        """Cost drag applies to sl outcomes too (entered = filled).

        Trade: entry=400.0, sl=399.0, risk_dollars=1.0, r_multiple=-1.0.
        cost_drag_r = 0.06 (same as above with slippage_ticks=2, commission=0.01)
        after_cost_r = -1.0 - 0.06 = -1.06
        """
        trade = self._make_filled_trade("sl", r_multiple=-1.0, entry=400.0, sl=399.0)
        cfg = ExitConfig(
            "fixed_2r",
            slippage_ticks=2,
            commission_per_share=0.01,
            report_realism=True,
        )
        summary = summarise_trades([trade], realism_config=cfg)
        assert summary["after_cost_total_r"] == pytest.approx(-1.0 - 0.06, abs=1e-9)

    def test_cost_drag_undecided_trade(self):
        """Cost drag applies to undecided outcomes (entered = filled)."""
        trade = self._make_filled_trade("undecided", r_multiple=0.5, entry=400.0, sl=399.0)
        cfg = ExitConfig(
            "fixed_2r",
            slippage_ticks=1,
            commission_per_share=0.0,
            report_realism=True,
        )
        # cost_drag = 1 * 0.01 * 2 / 1.0 = 0.02
        summary = summarise_trades([trade], realism_config=cfg)
        assert summary["after_cost_total_r"] == pytest.approx(0.5 - 0.02, abs=1e-9)

    def test_cost_not_applied_to_no_fill(self):
        """REVERT-SENSITIVE: cost drag must NOT be applied to no_fill outcomes.

        no_fill.filled=False → not entered → no cost.
        after_cost_total_r must equal only the filled trade's after-cost R.
        """
        filled_trade = self._make_filled_trade("tp", r_multiple=2.0, entry=400.0, sl=399.0)
        no_fill_trade = _make_trade(
            "no_fill", r_multiple=0.0, filled=False, entry=400.0, sl=399.0
        )
        cfg = ExitConfig(
            "fixed_2r",
            slippage_ticks=2,
            commission_per_share=0.01,
            report_realism=True,
        )
        summary = summarise_trades([filled_trade, no_fill_trade], realism_config=cfg)
        # Only the filled trade contributes; cost_drag = 0.06
        expected_after_cost = 2.0 - 0.06
        assert summary["after_cost_total_r"] == pytest.approx(expected_after_cost, abs=1e-9)

    def test_cost_not_applied_to_no_future(self):
        """Cost drag must NOT be applied to no_future outcomes (not entered)."""
        no_future_trade = _make_trade(
            "no_future", r_multiple=0.0, filled=False, entry=400.0, sl=399.0
        )
        cfg = ExitConfig(
            "fixed_2r",
            slippage_ticks=5,
            commission_per_share=0.05,
            report_realism=True,
        )
        summary = summarise_trades([no_future_trade], realism_config=cfg)
        assert summary["after_cost_total_r"] == pytest.approx(0.0, abs=1e-9)

    def test_zero_costs_after_cost_equals_total_r(self):
        """With slippage=0, commission=0: after_cost_total_r == total_r exactly."""
        trades = [
            self._make_filled_trade("tp", r_multiple=2.0),
            self._make_filled_trade("sl", r_multiple=-1.0),
        ]
        cfg = ExitConfig("fixed_2r", slippage_ticks=0, commission_per_share=0.0, report_realism=True)
        summary = summarise_trades(trades, realism_config=cfg)
        assert summary["after_cost_total_r"] == pytest.approx(summary["total_r"], abs=1e-9)


# ---------------------------------------------------------------------------
# Guard 3 — robust realism reporting keys
# ---------------------------------------------------------------------------

class TestRealismReportingKeys:
    """report_realism=True adds exactly the 5 documented keys; False adds none."""

    REALISM_KEYS = {
        "median_r",
        "after_cost_total_r",
        "after_cost_avg_r",
        "n_outlier_r",
        "winsorized_total_r",
    }

    BASE_KEYS = {
        "n_signals", "n_trades", "n_tp", "n_sl", "n_undecided",
        "n_no_future", "n_no_fill", "fill_rate", "win_rate",
        "total_r", "avg_r", "swing_fallback_rate",
    }

    def _trades(self) -> list[TradeOutcome]:
        return [
            _make_trade("tp", r_multiple=2.0, filled=True),
            _make_trade("sl", r_multiple=-1.0, filled=True),
            _make_trade("tp", r_multiple=1.5, filled=True),
        ]

    def test_report_realism_true_adds_all_five_keys(self):
        """report_realism=True → all 5 realism keys present."""
        cfg = ExitConfig("fixed_2r", report_realism=True)
        summary = summarise_trades(self._trades(), realism_config=cfg)
        missing = self.REALISM_KEYS - set(summary.keys())
        assert not missing, f"Missing realism keys: {missing}"

    def test_report_realism_false_adds_no_realism_keys(self):
        """report_realism=False → NONE of the 5 realism keys present (backward-compat)."""
        cfg = ExitConfig("fixed_2r", report_realism=False)
        summary = summarise_trades(self._trades(), realism_config=cfg)
        present = self.REALISM_KEYS & set(summary.keys())
        assert not present, f"Unexpected realism keys when report_realism=False: {present}"

    def test_no_realism_config_adds_no_realism_keys(self):
        """realism_config=None → NONE of the 5 realism keys present."""
        summary = summarise_trades(self._trades(), realism_config=None)
        present = self.REALISM_KEYS & set(summary.keys())
        assert not present, f"Unexpected realism keys with no realism_config: {present}"

    def test_median_r_correct(self):
        """median_r = median of filled r_multiples.

        r_multiples = [2.0, -1.0, 1.5]; sorted = [-1.0, 1.5, 2.0]; median = 1.5
        """
        cfg = ExitConfig("fixed_2r", report_realism=True)
        summary = summarise_trades(self._trades(), realism_config=cfg)
        assert summary["median_r"] == pytest.approx(1.5, abs=1e-9)

    def test_winsorized_clips_at_plus_minus_10(self):
        """winsorized_total_r clips |r|>10 to ±10.

        Trades: r=[12.0, -15.0, 2.0] → clipped: [10.0, -10.0, 2.0] → sum=2.0
        """
        trades = [
            _make_trade("tp", r_multiple=12.0, filled=True),
            _make_trade("sl", r_multiple=-15.0, filled=True),
            _make_trade("tp", r_multiple=2.0, filled=True),
        ]
        cfg = ExitConfig("fixed_2r", report_realism=True)
        summary = summarise_trades(trades, realism_config=cfg)
        assert summary["winsorized_total_r"] == pytest.approx(2.0, abs=1e-9)

    def test_n_outlier_r_counts_abs_r_gt_10(self):
        """n_outlier_r = count of filled trades where |r_multiple| > 10.

        r_multiples = [12.0, -15.0, 2.0, -10.0, 10.1]
        |r|>10: 12.0, 15.0, 10.1 → n_outlier=3
        """
        trades = [
            _make_trade("tp", r_multiple=12.0, filled=True),
            _make_trade("sl", r_multiple=-15.0, filled=True),
            _make_trade("tp", r_multiple=2.0, filled=True),
            _make_trade("sl", r_multiple=-10.0, filled=True),  # exactly 10 → NOT outlier
            _make_trade("tp", r_multiple=10.1, filled=True),   # just over → outlier
        ]
        cfg = ExitConfig("fixed_2r", report_realism=True)
        summary = summarise_trades(trades, realism_config=cfg)
        assert summary["n_outlier_r"] == 3

    def test_after_cost_avg_r_correct(self):
        """after_cost_avg_r = after_cost_total_r / n_trades.

        3 trades, 0 cost: after_cost_total_r = 2.0 + (-1.0) + 1.5 = 2.5
        after_cost_avg_r = 2.5 / 3
        """
        cfg = ExitConfig("fixed_2r", slippage_ticks=0, commission_per_share=0.0, report_realism=True)
        summary = summarise_trades(self._trades(), realism_config=cfg)
        assert summary["after_cost_total_r"] == pytest.approx(2.5, abs=1e-9)
        assert summary["after_cost_avg_r"] == pytest.approx(2.5 / 3, abs=1e-9)

    def test_realism_keys_do_not_alter_base_keys(self):
        """Enabling report_realism must not change any of the 12 base keys."""
        trades = self._trades()
        cfg_off = ExitConfig("fixed_2r", report_realism=False)
        cfg_on = ExitConfig("fixed_2r", report_realism=True)
        summary_off = summarise_trades(trades, realism_config=cfg_off)
        summary_on = summarise_trades(trades, realism_config=cfg_on)
        for key in self.BASE_KEYS:
            v_off = summary_off[key]
            v_on = summary_on[key]
            if isinstance(v_off, float) and not (np.isnan(v_off) and np.isnan(v_on)):
                assert v_off == pytest.approx(v_on), f"Base key '{key}' changed with report_realism=True"
            else:
                assert v_off == v_on, f"Base key '{key}' changed with report_realism=True"


# ---------------------------------------------------------------------------
# Guard 1+2+3 — backward-compat byte-identity (regression anchor)
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """ExitConfig with no guard fields → identical output to pre-guard baseline. REVERT-SENSITIVE.

    Any regression in the default (all-guards-off) path must be caught here.
    """

    WINDOW = _make_window(
        bar56_high=400.10, bar56_low=399.90,
        bar57_high=401.00, bar57_low=399.50,
        bar58_high=400.50, bar58_low=399.80,
        base_price=400.0, base_tr=2.0,
    )

    def test_default_config_no_floor_no_cost_no_report(self):
        """ExitConfig('fixed_2r') default: min_stop_atr_k=None, slippage=0, report=False."""
        cfg = ExitConfig("fixed_2r")
        assert cfg.min_stop_atr_k is None
        assert cfg.slippage_ticks == 0
        assert cfg.commission_per_share == 0.0
        assert cfg.report_realism is False

    def test_default_exit_matches_explicit_guards_off(self):
        """REVERT-SENSITIVE: default ExitConfig and explicit-guards-off ExitConfig produce identical TradeOutcome.

        If any guard runs when it should be disabled, this test catches it.
        """
        future = _make_future(
            (401.00, 401.50, 400.05, 401.00),
            (401.00, 405.00, 400.50, 404.50),
        )
        cfg_default = ExitConfig("fixed_2r")
        cfg_explicit_off = ExitConfig(
            "fixed_2r",
            min_stop_atr_k=None,
            slippage_ticks=0,
            commission_per_share=0.0,
            report_realism=False,
        )
        result_default = compute_exit(self.WINDOW, future, 1, cfg_default)
        result_explicit = compute_exit(self.WINDOW, future, 1, cfg_explicit_off)

        assert result_default.entry == pytest.approx(result_explicit.entry)
        assert result_default.sl == pytest.approx(result_explicit.sl)
        assert result_default.tp == pytest.approx(result_explicit.tp)
        assert result_default.outcome == result_explicit.outcome
        assert result_default.r_multiple == pytest.approx(result_explicit.r_multiple)
        assert result_default.filled == result_explicit.filled

    def test_default_summarise_no_realism_keys(self):
        """REVERT-SENSITIVE: summarise_trades with default ExitConfig has no realism keys."""
        REALISM_KEYS = {
            "median_r", "after_cost_total_r", "after_cost_avg_r",
            "n_outlier_r", "winsorized_total_r",
        }
        trades = [_make_trade("tp", r_multiple=2.0, filled=True)]
        # Default ExitConfig (report_realism=False)
        summary = summarise_trades(trades)
        present = REALISM_KEYS & set(summary.keys())
        assert not present, f"Realism keys leaked into default output: {present}"

    def test_default_total_r_unchanged(self):
        """Default summarise_trades total_r is unchanged vs explicit zero-cost config."""
        trades = [
            _make_trade("tp", r_multiple=2.0, filled=True),
            _make_trade("sl", r_multiple=-1.0, filled=True),
        ]
        summary_default = summarise_trades(trades)
        summary_explicit = summarise_trades(
            trades,
            realism_config=ExitConfig("fixed_2r", slippage_ticks=0, commission_per_share=0.0, report_realism=False),
        )
        assert summary_default["total_r"] == pytest.approx(summary_explicit["total_r"])
        assert summary_default["win_rate"] == pytest.approx(summary_explicit["win_rate"])


# ---------------------------------------------------------------------------
# Guard — __post_init__ validation
# ---------------------------------------------------------------------------

class TestExitConfigValidation:
    """ExitConfig.__post_init__ raises on invalid guard field values."""

    def test_min_stop_atr_k_zero_raises(self):
        """min_stop_atr_k=0 must raise ValueError (0 would collapse the floor)."""
        with pytest.raises(ValueError, match="min_stop_atr_k"):
            ExitConfig("fixed_2r", min_stop_atr_k=0)

    def test_min_stop_atr_k_negative_raises(self):
        """min_stop_atr_k<0 must raise ValueError."""
        with pytest.raises(ValueError, match="min_stop_atr_k"):
            ExitConfig("fixed_2r", min_stop_atr_k=-1.0)

    def test_slippage_ticks_negative_raises(self):
        """slippage_ticks<0 must raise ValueError."""
        with pytest.raises(ValueError, match="slippage_ticks"):
            ExitConfig("fixed_2r", slippage_ticks=-1)

    def test_valid_guard_values_no_raise(self):
        """Valid guard values must not raise."""
        cfg = ExitConfig("fixed_2r", min_stop_atr_k=0.5, slippage_ticks=0,
                         commission_per_share=0.0, report_realism=False)
        assert cfg.min_stop_atr_k == pytest.approx(0.5)

    def test_slippage_zero_valid(self):
        """slippage_ticks=0 is valid (boundary)."""
        cfg = ExitConfig("fixed_2r", slippage_ticks=0)
        assert cfg.slippage_ticks == 0

    def test_commission_per_share_negative_raises(self):
        """commission_per_share<0 must raise ValueError (would silently inflate after_cost_total_r)."""
        with pytest.raises(ValueError, match="commission_per_share"):
            ExitConfig("fixed_2r", commission_per_share=-0.01)

    def test_commission_per_share_zero_valid(self):
        """commission_per_share=0 is valid (boundary)."""
        cfg = ExitConfig("fixed_2r", commission_per_share=0.0)
        assert cfg.commission_per_share == pytest.approx(0.0)
