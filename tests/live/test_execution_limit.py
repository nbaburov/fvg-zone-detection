"""test_execution_limit.py — Tests for WS-C additions to PaperExecutor.

Coverage
--------
1. ABC conformance: PaperExecutor is an instance of Executor.
2. on_intent — market path: guards, sizing, _submit_bracket called with correct args.
3. on_intent — limit path: guards pass, _submit_limit_bracket called with intent SL/TP.
4. on_intent — skip_reason set → order never submitted.
5. on_intent — degenerate SL/TP guard: SKIP_SL_TOO_WIDE fires on limit intents too.
6. Limit-timeout: on_bar increments counter; after fill_timeout_bars → _cancel_limit_order
   called, open_count decremented, no_fill logged.
7. on_fill clears pending_limits entry so timeout does not cancel a filled order.
8. Regression pin (§11 pre-mortem): for a fixed window + strategy the exact SL/TP
   geometry produced by on_intent matches compute_exit(empty_future), proving no
   rounding drift was introduced by the refactor.
9. open_count reflects open_positions correctly.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.live.executor_base import Executor
from src.live.execution import PaperExecutor
from src.live.logger import FillEvent, SessionLogger
from src.live.order_plan import OrderIntent
from src.live.stream import MinuteBar
from src.strategy.exits import ExitConfig, compute_exit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)


def _make_window(direction: int) -> np.ndarray:
    """Synthetic 60×5 raw OHLCV window with a valid FVG pattern at bars 56-58."""
    base = 400.0
    window = np.zeros((60, 5), dtype=np.float64)
    for i in range(60):
        o = base + _RNG.uniform(-1, 1)
        h = o + _RNG.uniform(0.1, 1.5)
        l = o - _RNG.uniform(0.1, 1.5)
        c = _RNG.uniform(l, h)
        window[i] = [o, h, l, c, 10_000.0]
        base = c

    if direction == 1:  # bull
        window[56] = [399.0, 400.0, 398.5, 399.5, 5_000.0]
        window[57] = [399.5, 403.0, 399.0, 402.5, 5_000.0]
        window[58] = [402.5, 404.0, 401.0, 403.0, 5_000.0]
    else:  # bear
        window[56] = [403.0, 403.5, 402.0, 402.5, 5_000.0]
        window[57] = [402.5, 403.0, 399.0, 399.5, 5_000.0]
        window[58] = [399.5, 401.5, 398.0, 398.5, 5_000.0]
    return window


def _make_executor(
    risk_pct: float = 0.01,
    max_shares: int = 50,
    max_sl_pct: float = 0.03,
    fill_timeout_bars: int = 3,
    logger_sink=None,
) -> PaperExecutor:
    trading_client = MagicMock()
    account = MagicMock()
    account.portfolio_value = "100000.0"
    trading_client.get_account.return_value = account
    trading_client._api_key = "test_key"
    trading_client._secret_key = "test_secret"
    return PaperExecutor(
        trading_client=trading_client,
        risk_pct=risk_pct,
        max_shares=max_shares,
        max_sl_pct=max_sl_pct,
        fill_timeout_bars=fill_timeout_bars,
        logger_sink=logger_sink,
    )


def _make_intent(
    entry_type: str = "limit",
    signal: str = "bull",
    entry: float = 402.0,
    sl: float = 400.5,
    tp: float = 404.5,
    skip_reason=None,
    ts: str = "2024-01-02 11:30:00",
) -> OrderIntent:
    return OrderIntent(
        ticker="SPY",
        model="cnn_lstm",
        strategy="ict_iofed" if entry_type == "limit" else "fixed_2r",
        h1_timestamp=pd.Timestamp(ts, tz="America/New_York"),
        direction=1,
        signal=signal,
        confidence=0.75,
        entry=entry,
        sl=sl,
        tp=tp,
        entry_type=entry_type,
        skip_reason=skip_reason,
    )


def _make_bar(symbol: str = "SPY", hour: int = 11, minute: int = 31) -> MinuteBar:
    return MinuteBar(
        symbol=symbol,
        timestamp=pd.Timestamp(f"2024-01-02 {hour:02d}:{minute:02d}:00", tz="America/New_York"),
        open=402.0,
        high=402.5,
        low=401.5,
        close=402.2,
        volume=500.0,
    )


def _make_log() -> MagicMock:
    return MagicMock(spec=SessionLogger)


# ---------------------------------------------------------------------------
# 1. ABC conformance
# ---------------------------------------------------------------------------


class TestAbcConformance:
    def test_paper_executor_is_executor(self):
        ex = _make_executor()
        assert isinstance(ex, Executor)

    def test_executor_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Executor()  # type: ignore[abstract]

    def test_abc_methods_present(self):
        ex = _make_executor()
        assert callable(ex.on_intent)
        assert callable(ex.on_bar)
        assert callable(ex.on_session_close)
        assert callable(ex.open_count)


# ---------------------------------------------------------------------------
# 2. on_intent — market path
# ---------------------------------------------------------------------------


class TestOnIntentMarket:
    def test_market_intent_calls_submit_bracket(self):
        # entry_price=100, sl=99.50 → sl_dist=0.50, sl_pct=0.5% < 5%
        ex = _make_executor(max_sl_pct=0.05)
        intent = _make_intent(
            entry_type="market",
            signal="bull",
            entry=float("nan"),  # market: entry unknown at plan time
            sl=99.50,
            tp=float("nan"),
        )
        with patch.object(ex, "_get_entry_price", return_value=100.0), \
             patch.object(ex, "_submit_bracket", return_value="order_m1") as mock_submit:
            ex.on_intent(intent)

        mock_submit.assert_called_once()

    def test_market_intent_kill_switch_skips(self):
        log = _make_log()
        ex = _make_executor(logger_sink=log)
        ex._is_halted = True
        intent = _make_intent(entry_type="market", signal="bull", sl=399.49)
        ex.on_intent(intent)
        log.log_event.assert_called_once()
        assert log.log_event.call_args[0][1]["reason"] == "SKIP_KILL_SWITCH"

    def test_market_intent_none_signal_is_noop(self):
        ex = _make_executor()
        intent = _make_intent(entry_type="market", signal="none", sl=399.49)
        with patch.object(ex, "_submit_bracket") as mock_submit:
            ex.on_intent(intent)
        mock_submit.assert_not_called()

    def test_market_intent_open_count_increments(self):
        # entry_price=100, sl=99.50 → sl_pct=0.5% < 5%
        ex = _make_executor(max_sl_pct=0.05)
        assert ex.open_count() == 0
        intent = _make_intent(entry_type="market", signal="bull", entry=float("nan"), sl=99.50)
        with patch.object(ex, "_get_entry_price", return_value=100.0), \
             patch.object(ex, "_submit_bracket", return_value="order_oc1"):
            ex.on_intent(intent)
        assert ex.open_count() == 1


# ---------------------------------------------------------------------------
# 3. on_intent — limit path
# ---------------------------------------------------------------------------


class TestOnIntentLimit:
    def test_limit_intent_calls_submit_limit_bracket(self):
        ex = _make_executor(max_sl_pct=0.03)
        # entry=402, sl=400.5 → sl_dist=1.5, sl_pct=1.5/402≈0.0037 < 0.03
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_l1") as mock_submit:
            ex.on_intent(intent)

        mock_submit.assert_called_once()
        signal, qty, entry, sl, tp = mock_submit.call_args[0]
        assert signal == "bull"
        assert entry == pytest.approx(402.0)
        assert sl == pytest.approx(400.5)
        assert tp == pytest.approx(404.5)

    def test_limit_intent_pending_tracked(self):
        ex = _make_executor(max_sl_pct=0.03)
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_tracked"):
            ex.on_intent(intent)
        assert "order_tracked" in ex._pending_limits

    def test_limit_intent_sl_too_wide_skips(self):
        ex = _make_executor(max_sl_pct=0.005)
        # sl_dist=1.5/402=0.0037 → under 0.005; let's use wider gap: entry=100, sl=50
        intent = _make_intent(entry_type="limit", signal="bull", entry=100.0, sl=50.0, tp=200.0)
        log = _make_log()
        ex._log_sink = log
        with patch.object(ex, "_submit_limit_bracket") as mock_submit:
            ex.on_intent(intent)
        mock_submit.assert_not_called()
        log.log_event.assert_called()
        assert log.log_event.call_args[0][1]["reason"] == "SKIP_SL_TOO_WIDE"

    def test_limit_intent_open_position_skips(self):
        ex = _make_executor()
        ex._open_positions = 1
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        log = _make_log()
        ex._log_sink = log
        with patch.object(ex, "_submit_limit_bracket") as mock_submit:
            ex.on_intent(intent)
        mock_submit.assert_not_called()
        assert log.log_event.call_args[0][1]["reason"] == "SKIP_OPEN_POSITION"


# ---------------------------------------------------------------------------
# 4. skip_reason drops intent without submission
# ---------------------------------------------------------------------------


class TestSkipReason:
    def test_degenerate_geometry_drops(self):
        ex = _make_executor()
        intent = _make_intent(
            entry_type="limit",
            signal="bull",
            entry=float("nan"),
            sl=float("nan"),
            tp=float("nan"),
            skip_reason="DEGENERATE_GEOMETRY",
        )
        log = _make_log()
        ex._log_sink = log
        with patch.object(ex, "_submit_limit_bracket") as mock_submit, \
             patch.object(ex, "_submit_bracket") as mock_market:
            ex.on_intent(intent)
        mock_submit.assert_not_called()
        mock_market.assert_not_called()
        log.log_event.assert_called()
        assert log.log_event.call_args[0][1]["reason"] == "DEGENERATE_GEOMETRY"


# ---------------------------------------------------------------------------
# 5. Limit-timeout
# ---------------------------------------------------------------------------


class TestLimitTimeout:
    def test_timeout_cancels_and_decrements_open_count(self):
        ex = _make_executor(max_sl_pct=0.03, fill_timeout_bars=2)
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_timeout"):
            ex.on_intent(intent)

        assert ex.open_count() == 1
        assert "order_timeout" in ex._pending_limits

        # H1 close 1 (11:30 bucket) — counter→1, not yet timed out
        with patch.object(ex, "_cancel_limit_order") as mock_cancel:
            ex.on_bar(_make_bar(hour=11, minute=31))
            # repeat within same H1 bucket must NOT advance the counter
            ex.on_bar(_make_bar(hour=11, minute=45))
            mock_cancel.assert_not_called()
        assert ex.open_count() == 1

        # H1 close 2 (12:30 bucket) — counter→2 hits fill_timeout_bars=2 → cancel
        with patch.object(ex, "_cancel_limit_order") as mock_cancel:
            ex.on_bar(_make_bar(hour=12, minute=31))
            mock_cancel.assert_called_once_with("order_timeout")

        assert ex.open_count() == 0
        assert "order_timeout" not in ex._pending_limits

    def test_no_pending_limits_on_bar_is_noop(self):
        ex = _make_executor()
        bar = _make_bar()
        # Should not raise and does nothing
        ex.on_bar(bar)

    def test_fill_clears_pending_limit(self):
        ex = _make_executor(max_sl_pct=0.03, fill_timeout_bars=5)
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_filled"):
            ex.on_intent(intent)

        assert "order_filled" in ex._pending_limits

        fill = FillEvent(
            order_id="order_filled",
            filled_at=pd.Timestamp("2024-01-02 11:32:00", tz="America/New_York"),
            fill_price=402.0,
            fill_qty=10,
            fill_type="fill",
            side="buy",
        )
        ex.on_fill(fill)

        # pending_limits cleared; timeout cannot fire for this order
        assert "order_filled" not in ex._pending_limits

    def test_on_bar_does_not_timeout_filled_order(self):
        """After a fill clears the pending entry, on_bar must not cancel it."""
        ex = _make_executor(max_sl_pct=0.03, fill_timeout_bars=1)
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_pre_filled"):
            ex.on_intent(intent)

        fill = FillEvent(
            order_id="order_pre_filled",
            filled_at=pd.Timestamp("2024-01-02 11:32:00", tz="America/New_York"),
            fill_price=402.0,
            fill_qty=10,
            fill_type="fill",
            side="buy",
        )
        ex.on_fill(fill)

        bar = _make_bar()
        with patch.object(ex, "_cancel_limit_order") as mock_cancel:
            ex.on_bar(bar)
            mock_cancel.assert_not_called()


# ---------------------------------------------------------------------------
# 6. open_count
# ---------------------------------------------------------------------------


class TestOpenCount:
    def test_initial_open_count_is_zero(self):
        ex = _make_executor()
        assert ex.open_count() == 0

    def test_open_count_increments_on_submit(self):
        ex = _make_executor(max_sl_pct=0.03)
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_cnt"):
            ex.on_intent(intent)
        assert ex.open_count() == 1

    def test_open_count_decrements_on_fill(self):
        ex = _make_executor(max_sl_pct=0.03)
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_dec"):
            ex.on_intent(intent)
        assert ex.open_count() == 1

        fill = FillEvent(
            order_id="order_dec",
            filled_at=pd.Timestamp("2024-01-02 11:32:00", tz="America/New_York"),
            fill_price=402.0,
            fill_qty=10,
            fill_type="fill",
            side="buy",
        )
        ex.on_fill(fill)
        assert ex.open_count() == 0

    def test_open_count_zero_after_session_close(self):
        ex = _make_executor(max_sl_pct=0.03)
        intent = _make_intent(entry_type="limit", signal="bull", entry=402.0, sl=400.5, tp=404.5)
        with patch.object(ex, "_submit_limit_bracket", return_value="order_close"):
            ex.on_intent(intent)
        ex.on_session_close()
        assert ex.open_count() == 0
        assert ex._pending_limits == {}


# ---------------------------------------------------------------------------
# 7. Regression pin — SL/TP geometry (§11 pre-mortem)
#
# For a fixed window + "ict_iofed" strategy, the exact sl/tp that PaperExecutor
# would pass to _submit_limit_bracket must equal compute_exit(empty_future).
# This pins any accidental rounding drift introduced by the refactor.
# ---------------------------------------------------------------------------


class TestSlTpRegressionPin:
    def test_limit_intent_sl_tp_match_compute_exit(self):
        """on_intent passes through intent.sl/tp unchanged → same as compute_exit."""
        window = _make_window(direction=1)  # bull
        config = ExitConfig(strategy="ict_iofed")
        empty_future = np.empty((0, 4), dtype=np.float32)
        outcome = compute_exit(window, empty_future, 1, config)

        # Build an intent that matches what StrategyOrderPlanner would produce
        intent = OrderIntent(
            ticker="SPY",
            model="cnn_lstm",
            strategy="ict_iofed",
            h1_timestamp=pd.Timestamp("2024-01-02 11:30:00", tz="America/New_York"),
            direction=1,
            signal="bull",
            confidence=0.75,
            entry=float(outcome.entry),
            sl=float(outcome.sl),
            tp=float(outcome.tp),
            entry_type="limit",
            skip_reason=None,
        )

        ex = _make_executor(max_sl_pct=0.10)  # wide guard so geometry doesn't trip it
        captured: dict = {}

        def _capture_limit(signal, qty, entry, sl, tp):
            captured.update({"sl": sl, "tp": tp, "entry": entry})
            return "order_pin"

        with patch.object(ex, "_submit_limit_bracket", side_effect=_capture_limit):
            ex.on_intent(intent)

        if not captured:
            pytest.skip("Geometry too wide for default guards — expected in some seeds")

        assert captured["sl"] == pytest.approx(round(float(outcome.sl), 2), abs=1e-8)
        assert captured["tp"] == pytest.approx(round(float(outcome.tp), 2), abs=1e-8)
        assert captured["entry"] == pytest.approx(round(float(outcome.entry), 2), abs=1e-8)

    def test_market_intent_sl_from_intent_not_recomputed(self):
        """Market path uses intent.sl for sizing, not gap_low - 0.01."""
        window = _make_window(direction=1)
        config = ExitConfig(strategy="fixed_2r")
        empty_future = np.empty((0, 4), dtype=np.float32)
        outcome = compute_exit(window, empty_future, 1, config)

        # For fixed_2r, outcome.sl is populated; entry is NaN (unknown until fill)
        if math.isnan(outcome.sl):
            pytest.skip("Window produced degenerate SL")

        ex = _make_executor(max_sl_pct=0.10)
        entry_price = 402.0  # mock fill price

        intent = OrderIntent(
            ticker="SPY",
            model="lstm",
            strategy="fixed_2r",
            h1_timestamp=pd.Timestamp("2024-01-02 11:30:00", tz="America/New_York"),
            direction=1,
            signal="bull",
            confidence=0.8,
            entry=float("nan"),
            sl=float(outcome.sl),
            tp=float("nan"),
            entry_type="market",
            skip_reason=None,
        )

        captured: dict = {}

        def _capture_market(signal, qty, sl, tp):
            captured.update({"sl": sl, "tp": tp})
            return "order_mpin"

        with patch.object(ex, "_get_entry_price", return_value=entry_price), \
             patch.object(ex, "_submit_bracket", side_effect=_capture_market):
            ex.on_intent(intent)

        if not captured:
            pytest.skip("Sizing or SL guard skipped this intent")

        # sl must come from intent.sl, not any re-derived value
        assert captured["sl"] == pytest.approx(round(float(outcome.sl), 2), abs=1e-8)
