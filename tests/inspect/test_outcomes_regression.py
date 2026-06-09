"""test_outcomes_regression.py — V1 golden anchor + public API regression tests.

All expected values are hand-computed from synthetic geometry.
No mocking of code under test.

Key invariants tested:
1. ExitConfig("fixed_2r") reproduces expected V1 values (golden anchor).
2. compute_trades_for_model / summarise_trades return documented fields.
3. n_trades = n_filled (not all-inclusive), so no_fill excluded.
4. no_fill trades are NOT counted in win-rate / total_R.
5. New fields (fill_rate, n_signals, swing_fallback_rate) present + correct.
6. Back-compat: Trade alias still importable and is TradeOutcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inspect.outcomes import (
    Trade,
    TradeOutcome,
    ExitConfig,
    compute_trades_for_model,
    summarise_trades,
)
from src.strategy.exits import TICK, compute_exit


# ---------------------------------------------------------------------------
# Helpers (mirrors test_fvg_exits.py pattern)
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
    window = np.zeros((60, 5), dtype=np.float64)
    for i in range(60):
        window[i] = [100.0, swing_high, swing_low, 100.0, 1000.0]
    window[56] = [bar56_low, bar56_high, bar56_low, bar56_high, 1000.0]
    window[57] = [bar57_low, bar57_high, bar57_low, bar57_high, 1000.0]
    window[58] = [bar58_low, bar58_high, bar58_low, bar58_high, 1000.0]
    return window


def _make_future(*bars: tuple[float, float, float, float]) -> np.ndarray:
    arr = np.zeros((len(bars), 5), dtype=np.float64)
    for i, (o, h, l, c) in enumerate(bars):
        arr[i] = [o, h, l, c, 1000.0]
    return arr


# ---------------------------------------------------------------------------
# V1 golden anchor — hand-computed expected values
# ---------------------------------------------------------------------------

class TestV1GoldenAnchor:
    """Verify fixed_2r matches hand-computed expected values.

    Geometry:
      bar_56: high=100, low=95
      future[0].open = 103 (market entry)
      SL = 100 - 0.01 = 99.99
      risk = 103 - 99.99 = 3.01
      TP = 103 + 2 * 3.01 = 109.02
    """

    WINDOW = _make_window(100, 95, 108, 96, 103, 101)
    SL = 100.0 - TICK       # 99.99
    ENTRY = 103.0
    RISK = ENTRY - SL       # 3.01
    TP = ENTRY + 2.0 * RISK # 109.02

    def test_v1_golden_bull_tp(self):
        """V1 bull TP hit: exact entry, sl, tp, r_multiple=2.0."""
        future = _make_future(
            (self.ENTRY, 106, 102, 104),       # bar0: no hit
            (105, self.TP + 0.5, 104, 105),    # bar1: TP
        )
        result = compute_exit(self.WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.entry == pytest.approx(self.ENTRY)
        assert result.sl == pytest.approx(self.SL)
        assert result.tp == pytest.approx(self.TP)
        assert result.outcome == "tp"
        assert result.r_multiple == pytest.approx(2.0)
        assert result.filled is True
        assert result.swing_fallback is False

    def test_v1_golden_bull_sl(self):
        """V1 bull SL hit: r_multiple=-1.0."""
        future = _make_future(
            (self.ENTRY, 104, self.SL - 0.5, 103),  # bar0: SL
        )
        result = compute_exit(self.WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.outcome == "sl"
        assert result.r_multiple == pytest.approx(-1.0)

    def test_v1_golden_bear_sl_anchor(self):
        """V1 bear: SL = bar_56.low + TICK (above gap upper edge)."""
        # bar_56 low=100 => SL = 100.01
        window = _make_window(106, 100, 103, 92, 104, 99)
        future = _make_future((98.0, 99, 97, 98))
        result = compute_exit(window, future, 2, ExitConfig("fixed_2r"))
        assert result.sl == pytest.approx(100.0 + TICK)

    def test_v1_computed_via_outcomes_module(self):
        """compute_exit imported via outcomes.py path yields same result."""
        future = _make_future(
            (self.ENTRY, 106, 102, 104),
            (105, self.TP + 0.5, 104, 105),
        )
        result = compute_exit(self.WINDOW, future, 1, ExitConfig("fixed_2r"))
        assert result.tp == pytest.approx(self.TP)
        assert result.outcome == "tp"


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------

class TestBackCompatAlias:
    def test_trade_alias_is_trade_outcome(self):
        """Trade must be an alias for TradeOutcome — existing importers depend on it."""
        assert Trade is TradeOutcome

    def test_exit_config_importable_from_outcomes(self):
        """ExitConfig importable from outcomes module (re-exported)."""
        cfg = ExitConfig("fixed_2r")
        assert cfg.strategy == "fixed_2r"


# ---------------------------------------------------------------------------
# compute_trades_for_model — API shape + correctness
# ---------------------------------------------------------------------------

class TestComputeTradesForModel:
    """Verify batch API returns correct trades and filters non-positive preds."""

    def _batch(self, tp_rr: float = 2.0):
        """Two windows: pred=1 (bull TP), pred=2 (bear SL), pred=0 (ignored).

        Future bars are generated to guarantee the bull TP is hit at ``tp_rr``.
        bar0: no hit; bar1: TP hit (high well above tp_rr level); bar2: filler.
        """
        # Window 0: bull, TP hit
        w0 = _make_window(100, 95, 108, 96, 103, 101)
        entry0 = 103.0
        sl0 = 100.0 - TICK   # 99.99
        risk0 = entry0 - sl0
        tp0 = entry0 + tp_rr * risk0
        fut0 = _make_future(
            (entry0, 106, 102, 104),          # bar0: no hit
            (105, tp0 + 2.0, 104, 105),       # bar1: TP hit (enough for any tp_rr)
            (105, 107, 104, 106),              # bar2: filler
        )

        # Window 1: bear, SL hit
        w1 = _make_window(106, 100, 103, 92, 104, 99)
        entry1 = 98.0
        sl1 = 100.0 + TICK   # 100.01
        fut1 = _make_future(
            (entry1, sl1 + 0.5, 97, 98),  # SL hit bar0
            (99, 100, 98, 99),            # bar1: filler
            (99, 100, 98, 99),            # bar2: filler
        )

        # Window 2: pred=0, should be ignored
        w2 = _make_window(100, 95, 108, 96, 103, 101)
        fut2 = _make_future(
            (103, 106, 102, 104),
            (103, 106, 102, 104),
            (103, 106, 102, 104),
        )

        windows_raw = np.stack([w0, w1, w2])
        future_ohlcv = np.stack([fut0, fut1, fut2])
        preds = np.array([1, 2, 0], dtype=np.int64)
        future_timestamps = [[], [], []]
        return windows_raw, future_ohlcv, future_timestamps, preds

    def test_only_positive_preds_produce_trades(self):
        """pred=0 windows are skipped; only preds 1 and 2 produce trades."""
        windows_raw, future_ohlcv, future_timestamps, preds = self._batch()
        trades = compute_trades_for_model(
            windows_raw, future_ohlcv, future_timestamps, preds
        )
        assert len(trades) == 2

    def test_trade_directions_match_preds(self):
        windows_raw, future_ohlcv, future_timestamps, preds = self._batch()
        trades = compute_trades_for_model(
            windows_raw, future_ohlcv, future_timestamps, preds
        )
        assert trades[0].direction == 1
        assert trades[1].direction == 2

    def test_trade_outcomes(self):
        """Window 0 => tp, window 1 => sl."""
        windows_raw, future_ohlcv, future_timestamps, preds = self._batch()
        trades = compute_trades_for_model(
            windows_raw, future_ohlcv, future_timestamps, preds
        )
        assert trades[0].outcome == "tp"
        assert trades[1].outcome == "sl"

    def test_exit_config_forwarded(self):
        """ExitConfig passed to compute_trades_for_model is respected.
        Future bars are built for tp_rr=3.0 so the TP is actually hit.
        """
        windows_raw, future_ohlcv, future_timestamps, preds = self._batch(tp_rr=3.0)
        cfg = ExitConfig("fixed_2r", tp_rr=3.0)
        trades = compute_trades_for_model(
            windows_raw, future_ohlcv, future_timestamps, preds,
            exit_config=cfg,
        )
        # TP trade should have r_multiple=3.0 (not 2.0)
        tp_trade = next(t for t in trades if t.outcome == "tp")
        assert tp_trade.r_multiple == pytest.approx(3.0)

    def test_tp_rr_default_forwarded_when_no_exit_config(self):
        """tp_rr kwarg is forwarded to ExitConfig when exit_config=None."""
        windows_raw, future_ohlcv, future_timestamps, preds = self._batch(tp_rr=2.0)
        trades = compute_trades_for_model(
            windows_raw, future_ohlcv, future_timestamps, preds,
            tp_rr=2.0,
        )
        tp_trade = next(t for t in trades if t.outcome == "tp")
        assert tp_trade.r_multiple == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# summarise_trades — field completeness + no_fill exclusion
# ---------------------------------------------------------------------------

class TestSummariseTrades:
    """Verify summarise_trades returns all documented fields with correct values."""

    REQUIRED_FIELDS = {
        "n_signals", "n_trades", "n_tp", "n_sl", "n_undecided",
        "n_no_future", "n_no_fill", "fill_rate", "win_rate",
        "total_r", "avg_r", "swing_fallback_rate",
    }

    def _make_trade(self, outcome: str, direction: int = 1,
                    r_multiple: float = 0.0, filled: bool = True,
                    swing_fallback: bool = False) -> TradeOutcome:
        import pandas as pd
        return TradeOutcome(
            window_idx=0,
            entry_ts=pd.NaT,
            direction=direction,
            entry=100.0,
            sl=99.0,
            tp=102.0,
            outcome=outcome,
            exit_idx_in_future=-1,
            exit_ts=None,
            exit_price=100.0,
            r_multiple=r_multiple,
            filled=filled,
            swing_fallback=swing_fallback,
        )

    def test_all_required_fields_present(self):
        """summarise_trades must return all documented fields."""
        trades = [self._make_trade("tp", r_multiple=2.0)]
        summary = summarise_trades(trades)
        missing = self.REQUIRED_FIELDS - set(summary.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_empty_trades_all_zero(self):
        summary = summarise_trades([])
        assert summary["n_signals"] == 0
        assert summary["n_trades"] == 0
        assert summary["win_rate"] == pytest.approx(0.0)
        assert summary["total_r"] == pytest.approx(0.0)

    def test_win_rate_excludes_no_fill(self):
        """REVERT-SENSITIVE: no_fill must not appear in win_rate denominator."""
        trades = [
            self._make_trade("tp", r_multiple=2.0, filled=True),   # counts
            self._make_trade("sl", r_multiple=-1.0, filled=True),   # counts
            self._make_trade("no_fill", r_multiple=0.0, filled=False),  # excluded
        ]
        summary = summarise_trades(trades)
        # decided = 2 (tp + sl); no_fill not in decided
        assert summary["win_rate"] == pytest.approx(0.5)
        assert summary["n_no_fill"] == 1

    def test_total_r_excludes_no_fill(self):
        """REVERT-SENSITIVE: no_fill r_multiple must not add to total_r."""
        trades = [
            self._make_trade("tp", r_multiple=2.0, filled=True),
            self._make_trade("no_fill", r_multiple=99.0, filled=False),  # should be excluded
        ]
        summary = summarise_trades(trades)
        assert summary["total_r"] == pytest.approx(2.0)

    def test_n_trades_equals_n_filled(self):
        """n_trades counts only filled trades (not no_fill, not no_future)."""
        trades = [
            self._make_trade("tp", r_multiple=2.0, filled=True),
            self._make_trade("sl", r_multiple=-1.0, filled=True),
            self._make_trade("no_fill", r_multiple=0.0, filled=False),
            self._make_trade("no_future", r_multiple=0.0, filled=False),
        ]
        summary = summarise_trades(trades)
        assert summary["n_trades"] == 2
        assert summary["n_signals"] == 4

    def test_fill_rate_computed_correctly(self):
        """fill_rate = n_filled / n_signals."""
        trades = [
            self._make_trade("tp", filled=True),
            self._make_trade("sl", filled=True),
            self._make_trade("no_fill", filled=False),
            self._make_trade("no_fill", filled=False),
        ]
        summary = summarise_trades(trades)
        assert summary["fill_rate"] == pytest.approx(2 / 4)
        assert summary["n_no_fill"] == 2

    def test_swing_fallback_rate(self):
        """swing_fallback_rate = n_swing_fallback / n_trades."""
        trades = [
            self._make_trade("tp", filled=True, swing_fallback=True),
            self._make_trade("tp", filled=True, swing_fallback=False),
            self._make_trade("sl", filled=True, swing_fallback=False),
        ]
        summary = summarise_trades(trades)
        assert summary["swing_fallback_rate"] == pytest.approx(1 / 3)

    def test_avg_r_over_filled_only(self):
        """avg_r = total_r / n_trades (filled only)."""
        trades = [
            self._make_trade("tp", r_multiple=2.0, filled=True),
            self._make_trade("sl", r_multiple=-1.0, filled=True),
            self._make_trade("no_fill", r_multiple=99.0, filled=False),
        ]
        summary = summarise_trades(trades)
        assert summary["avg_r"] == pytest.approx(1.0 / 2)

    def test_v1_n_trades_unchanged_on_full_future_windows(self):
        """V1 fixed_2r with full future windows: n_trades == n_signals (all fill).

        Verifies the semantics change (n_trades=filled not all) does NOT alter
        fixed_2r results where every trade is market-filled.
        """
        w = _make_window(100, 95, 108, 96, 103, 101)
        entry = 103.0
        sl = 100.0 - TICK
        tp = entry + 2.0 * (entry - sl)
        fut = _make_future(
            (entry, 106, 102, 104),
            (105, tp + 1, 104, 105),
        )

        # 3 identical bull predictions
        windows_raw = np.stack([w, w, w])
        future_ohlcv = np.stack([fut, fut, fut])
        preds = np.array([1, 1, 1], dtype=np.int64)
        future_timestamps = [[], [], []]

        trades = compute_trades_for_model(
            windows_raw, future_ohlcv, future_timestamps, preds,
            exit_config=ExitConfig("fixed_2r"),
        )
        summary = summarise_trades(trades)
        assert summary["n_signals"] == 3
        assert summary["n_trades"] == 3  # all market-filled
        assert summary["n_tp"] == 3
        assert summary["win_rate"] == pytest.approx(1.0)
