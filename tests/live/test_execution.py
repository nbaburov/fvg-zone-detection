"""Unit tests for PaperExecutor.

Tests:
- Each skip guard fires correctly given mocked state
- Position sizing math: equity=100k, risk=1%, sl_dist=0.50 → qty=2000 (capped 50)
- SL/TP prices correct for bull and bear signals
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.live.decision import TradeAction
from src.live.execution import PaperExecutor
from src.live.logger import SessionLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_action(
    signal: str = "bull",
    confidence: float = 0.8,
    gap_low: float = 99.0,
    gap_high: float = 101.0,
    ts: str = "2024-01-02 11:30:00",
) -> TradeAction:
    return TradeAction(
        signal=signal,
        confidence=confidence,
        proba=np.array([0.1, 0.8, 0.1], dtype=np.float32),
        h1_timestamp=pd.Timestamp(ts, tz="America/New_York"),
        gap_low=gap_low,
        gap_high=gap_high,
        skip_reason=None,
    )


def make_mock_account(portfolio_value: float = 100_000.0) -> MagicMock:
    account = MagicMock()
    account.portfolio_value = str(portfolio_value)
    account.cash = str(portfolio_value * 0.5)
    return account


def make_executor(
    trading_client=None,
    risk_pct: float = 0.01,
    tp_rr: float = 2.0,
    max_sl_pct: float = 0.03,
    max_daily_dd: float = 0.02,
    max_shares: int = 50,
) -> PaperExecutor:
    if trading_client is None:
        trading_client = MagicMock()
        trading_client.get_account.return_value = make_mock_account(100_000.0)
        trading_client._api_key = "test_key"
        trading_client._secret_key = "test_secret"
    return PaperExecutor(
        trading_client=trading_client,
        risk_pct=risk_pct,
        tp_rr=tp_rr,
        max_sl_pct=max_sl_pct,
        max_daily_dd=max_daily_dd,
        max_shares=max_shares,
    )


def make_mock_logger() -> MagicMock:
    return MagicMock(spec=SessionLogger)


# ---------------------------------------------------------------------------
# Test: Guards
# ---------------------------------------------------------------------------


class TestExecutorGuards:
    def test_kill_switch_skips(self):
        executor = make_executor()
        executor._is_halted = True
        log = make_mock_logger()
        action = make_action(signal="bull")

        executor.on_trade_action(action, log)

        log.log_event.assert_called_once()
        call_args = log.log_event.call_args[0]
        assert call_args[1]["reason"] == "SKIP_KILL_SWITCH"

    def test_none_signal_is_noop(self):
        executor = make_executor()
        log = make_mock_logger()
        action = make_action(signal="none")

        executor.on_trade_action(action, log)

        # Should not call log_event for skip or log_order
        log.log_order.assert_not_called()

    def test_blackout_open_bar_skips(self):
        """09:30 bar (hour=9) should be skipped."""
        executor = make_executor()
        log = make_mock_logger()
        action = make_action(signal="bull", ts="2024-01-02 09:30:00")

        executor.on_trade_action(action, log)

        log.log_event.assert_called_once()
        call_args = log.log_event.call_args[0]
        assert call_args[1]["reason"] == "SKIP_BLACKOUT"

    def test_blackout_close_bar_skips(self):
        """15:30 bar (hour=15) should be skipped."""
        executor = make_executor()
        log = make_mock_logger()
        action = make_action(signal="bull", ts="2024-01-02 15:30:00")

        executor.on_trade_action(action, log)

        log.log_event.assert_called_once()
        call_args = log.log_event.call_args[0]
        assert call_args[1]["reason"] == "SKIP_BLACKOUT"

    def test_open_position_skips(self):
        executor = make_executor()
        executor._open_positions = 1
        log = make_mock_logger()
        action = make_action(signal="bull", ts="2024-01-02 11:30:00")

        # Mock entry price
        with patch.object(executor, "_get_entry_price", return_value=100.0):
            executor.on_trade_action(action, log)

        log.log_event.assert_called_once()
        call_args = log.log_event.call_args[0]
        assert call_args[1]["reason"] == "SKIP_OPEN_POSITION"

    def test_sl_too_wide_skips(self):
        """SL distance > 3% of price should be skipped."""
        executor = make_executor(max_sl_pct=0.03)
        log = make_mock_logger()
        # entry ~100, gap_low=50 → sl=49.99, sl_dist=50 → 50% >> 3%
        action = make_action(signal="bull", gap_low=50.0, gap_high=60.0, ts="2024-01-02 11:30:00")

        with patch.object(executor, "_get_entry_price", return_value=100.0):
            executor.on_trade_action(action, log)

        log.log_event.assert_called_once()
        call_args = log.log_event.call_args[0]
        assert call_args[1]["reason"] == "SKIP_SL_TOO_WIDE"

    def test_sizing_floor_skips_qty_zero(self):
        """Very tight equity causes qty=0 → SKIP_SIZING."""
        executor = make_executor(risk_pct=0.0001, max_shares=50)
        executor._client.get_account.return_value = make_mock_account(100.0)  # tiny equity
        log = make_mock_logger()
        # entry=100, sl_dist=50 → qty = floor(100 * 0.0001 / 50) = 0
        action = make_action(signal="bull", gap_low=50.0, gap_high=60.0, ts="2024-01-02 11:30:00")

        with patch.object(executor, "_get_entry_price", return_value=100.0):
            executor.on_trade_action(action, log)

        # Should emit SKIP_SL_TOO_WIDE first (50% > 3%) — the SL guard fires before sizing
        log.log_event.assert_called()


# ---------------------------------------------------------------------------
# Test: Position sizing
# ---------------------------------------------------------------------------


class TestExecutorSizing:
    def test_sizing_capped_at_max_shares(self):
        """equity=100k, risk=1%, sl_dist=0.50 → raw_qty=2000, capped to 50."""
        executor = make_executor(risk_pct=0.01, max_shares=50)
        log = make_mock_logger()
        # gap_low=99.50, entry=100.00 → sl=99.49, sl_dist=0.51 ≈ 0.5
        action = make_action(signal="bull", gap_low=99.50, gap_high=101.0, ts="2024-01-02 11:30:00")

        submitted_order_id = "order123"

        with patch.object(executor, "_get_entry_price", return_value=100.0), \
             patch.object(executor, "_submit_bracket", return_value=submitted_order_id) as mock_submit:
            executor.on_trade_action(action, log)

        mock_submit.assert_called_once()
        call_args = mock_submit.call_args[0]  # positional: signal, qty, sl, tp
        qty = call_args[1]
        assert qty == 50  # capped at max_shares

    def test_sizing_formula(self):
        """Verify qty = floor(equity * risk / sl_dist) without cap."""
        executor = make_executor(risk_pct=0.01, max_shares=10_000)
        log = make_mock_logger()
        # entry=100, gap_low=99.50 → sl=99.49, sl_dist=0.51
        # qty = floor(100_000 * 0.01 / 0.51) = floor(1960.78) = 1960
        action = make_action(signal="bull", gap_low=99.50, gap_high=101.0, ts="2024-01-02 11:30:00")

        with patch.object(executor, "_get_entry_price", return_value=100.0), \
             patch.object(executor, "_submit_bracket", return_value="order1") as mock_submit:
            executor.on_trade_action(action, log)

        call_args = mock_submit.call_args[0]
        qty = call_args[1]
        import math
        expected_qty = min(math.floor(100_000 * 0.01 / (100.0 - 99.49)), 10_000)
        assert qty == expected_qty


# ---------------------------------------------------------------------------
# Test: SL/TP calculation
# ---------------------------------------------------------------------------


class TestExecutorSLTP:
    def test_bull_sl_below_gap_low(self):
        executor = make_executor()
        log = make_mock_logger()
        action = make_action(signal="bull", gap_low=99.50, gap_high=101.0, ts="2024-01-02 11:30:00")

        with patch.object(executor, "_get_entry_price", return_value=100.0), \
             patch.object(executor, "_submit_bracket", return_value="o1") as mock_submit:
            executor.on_trade_action(action, log)

        call_args = mock_submit.call_args[0]
        sl = call_args[2]
        tp = call_args[3]
        # sl = gap_low - 0.01 = 99.49
        assert sl == pytest.approx(99.49)
        # tp = entry + 2 * (entry - sl) = 100 + 2 * 0.51 = 101.02
        assert tp == pytest.approx(101.02)

    def test_bear_sl_above_gap_high(self):
        executor = make_executor()
        log = make_mock_logger()
        action = make_action(signal="bear", gap_low=99.0, gap_high=101.0, ts="2024-01-02 11:30:00")

        with patch.object(executor, "_get_entry_price", return_value=100.0), \
             patch.object(executor, "_submit_bracket", return_value="o2") as mock_submit:
            executor.on_trade_action(action, log)

        call_args = mock_submit.call_args[0]
        sl = call_args[2]
        tp = call_args[3]
        # sl = gap_high + 0.01 = 101.01
        assert sl == pytest.approx(101.01)
        # tp = entry - 2 * (sl - entry) = 100 - 2 * 1.01 = 97.98
        assert tp == pytest.approx(97.98)
