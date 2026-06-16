"""test_sim_executor.py — TDD coverage for SimFillExecutor.

Tests
-----
- Market intent fills at next 1-min bar open
- Limit intent fills optimistic (bar.low <= limit for bull)
- Limit intent fills conservative (bar.close <= limit for bull)
- Limit intent does NOT fill when condition not met
- SL intrabar resolution: r_multiple < 0 via real compute_exit
- TP intrabar resolution: r_multiple > 0 via real compute_exit
- open_count tracks pending + open positions
- on_session_close resolves all open positions
- Cross-session reload: FleetState restores open positions; SimFillExecutor
  advances them on new bars
- Idempotent re-apply: resolving same trade twice in FleetState is no-op
  (covered by test_fleet_state; here we confirm executor doesn't double-resolve)
- Skipped intents (skip_reason set) are silently dropped
- Bars for a different symbol are ignored
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.live.fleet_state import FleetState, make_intent_id
from src.live.order_plan import OrderIntent
from src.live.sim_executor import SimFillExecutor
from src.live.stream import MinuteBar
from src.strategy.exits import ExitConfig

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(99)

BASE_TS = pd.Timestamp("2024-03-01 10:30:00", tz="America/New_York")


def _ts(offset_minutes: int = 0) -> pd.Timestamp:
    return BASE_TS + pd.Timedelta(minutes=offset_minutes)


def _make_window(direction: int) -> np.ndarray:
    """60×5 raw OHLCV window with a valid FVG pattern in bars 56/57/58."""
    base = 470.0
    window = np.zeros((60, 5), dtype=np.float64)
    for i in range(60):
        o = base + _RNG.uniform(-0.5, 0.5)
        h = o + _RNG.uniform(0.05, 0.8)
        l = o - _RNG.uniform(0.05, 0.8)
        c = float(_RNG.uniform(l, h))
        window[i] = [o, h, l, c, 5_000.0]
        base = c

    if direction == 1:  # bull FVG
        window[56] = [469.0, 470.0, 468.5, 469.5, 5_000.0]  # high=470
        window[57] = [469.5, 474.0, 469.0, 473.5, 5_000.0]  # impulse up
        window[58] = [473.5, 475.0, 471.0, 474.0, 5_000.0]  # low=471 > 470
    else:  # bear FVG
        window[56] = [473.0, 473.5, 472.0, 472.5, 5_000.0]  # low=472
        window[57] = [472.5, 473.0, 468.0, 468.5, 5_000.0]  # impulse down
        window[58] = [468.5, 471.5, 467.0, 467.5, 5_000.0]  # high=471.5 < 472
    return window


def _market_intent(
    ticker: str = "SPY",
    direction: int = 1,
    sl: float = None,  # if None, computed from window geometry (bar56.high - TICK)
    h1_ts: pd.Timestamp = None,
) -> OrderIntent:
    if h1_ts is None:
        h1_ts = BASE_TS
    # For bull fixed_2r, bar_56.high = 470.0, SL = 469.99 (470.0 - 0.01 TICK)
    actual_sl = sl if sl is not None else 469.99
    return OrderIntent(
        ticker=ticker,
        model="lstm",
        strategy="fixed_2r",
        h1_timestamp=h1_ts,
        direction=direction,
        signal="bull" if direction == 1 else "bear",
        confidence=0.8,
        entry=float("nan"),  # market: entry = next bar open
        sl=actual_sl,
        tp=float("nan"),  # market: tp set after fill
        entry_type="market",
        skip_reason=None,
    )


def _limit_intent(
    limit: float = 471.5,
    ticker: str = "SPY",
    direction: int = 1,
    sl: float = 469.0,
    tp: float = 475.5,
    h1_ts: pd.Timestamp = None,
) -> OrderIntent:
    if h1_ts is None:
        h1_ts = BASE_TS
    return OrderIntent(
        ticker=ticker,
        model="lstm",
        strategy="ict_iofed",
        h1_timestamp=h1_ts,
        direction=direction,
        signal="bull" if direction == 1 else "bear",
        confidence=0.75,
        entry=limit,
        sl=sl,
        tp=tp,
        entry_type="limit",
        skip_reason=None,
    )


def _bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    offset: int = 0,
    symbol: str = "SPY",
) -> MinuteBar:
    return MinuteBar(
        symbol=symbol,
        timestamp=_ts(offset),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=10_000.0,
    )


def _make_executor(
    tmp_path,
    strategy: str = "fixed_2r",
    fill_mode: str = "conservative",
    fleet_id: str = "test-fleet",
    cell_key: str = "SPY:lstm:fixed_2r",
) -> tuple[SimFillExecutor, FleetState]:
    cfg = ExitConfig(strategy=strategy, fill_mode=fill_mode)
    state = FleetState(fleet_id, base_dir=tmp_path)
    exe = SimFillExecutor(realism=cfg, state=state, cell_key=cell_key)
    return exe, state


# ---------------------------------------------------------------------------
# Market fill tests
# ---------------------------------------------------------------------------


def test_market_intent_fills_at_next_bar_open(tmp_path):
    exe, state = _make_executor(tmp_path, strategy="fixed_2r")
    # SL from geometry: bar_56.high=470.0, SL=469.99 (TICK=0.01)
    # Fill bar must have low > 469.99 to avoid immediate SL resolution
    intent = _market_intent()
    window = _make_window(1)

    intent.window_raw = window

    exe.on_intent(intent)
    assert exe.open_count() == 1  # pending

    # Deliver the next 1-min bar — should fill at open; low=470.1 > SL=469.99
    bar = _bar(open_=470.5, high=471.0, low=470.1, close=470.8, offset=1)
    exe.on_bar(bar)

    positions = state.open_positions()
    assert len(positions) == 1
    assert positions[0].fill_price == pytest.approx(470.5)
    assert positions[0].fill_ts == _ts(1)


def test_market_intent_open_count_after_fill(tmp_path):
    exe, state = _make_executor(tmp_path, strategy="fixed_2r")
    intent = _market_intent()
    window = _make_window(1)

    intent.window_raw = window

    exe.on_intent(intent)
    assert exe.open_count() == 1  # pending

    # low=470.1 > SL=469.99 — no resolution on fill bar
    bar = _bar(470.5, 471.0, 470.1, 470.8, offset=1)
    exe.on_bar(bar)
    # pending gone, one open position
    assert exe.open_count() == 1


def test_skipped_intent_is_dropped(tmp_path):
    exe, state = _make_executor(tmp_path, strategy="fixed_2r")
    intent = _market_intent(sl=468.0)
    intent.skip_reason = "DEGENERATE_GEOMETRY"

    intent.window_raw = _make_window(1)

    exe.on_intent(intent)
    assert exe.open_count() == 0


def test_bar_for_wrong_symbol_is_ignored(tmp_path):
    exe, state = _make_executor(tmp_path, strategy="fixed_2r", cell_key="SPY:lstm:fixed_2r")
    intent = _market_intent(ticker="SPY", sl=468.0)
    window = _make_window(1)

    intent.window_raw = window

    exe.on_intent(intent)
    # Bar for QQQ — should not fill our SPY intent
    bar = MinuteBar(symbol="QQQ", timestamp=_ts(1), open=370.0, high=371.0, low=369.5, close=370.5, volume=5000.0)
    exe.on_bar(bar)

    # Still pending
    assert exe.open_count() == 1
    assert len(state.open_positions()) == 0


# ---------------------------------------------------------------------------
# Limit fill tests
# ---------------------------------------------------------------------------


def test_limit_fill_optimistic_bull(tmp_path):
    """Bull limit fills when bar.low <= limit (optimistic)."""
    exe, state = _make_executor(
        tmp_path, strategy="ict_iofed", fill_mode="optimistic",
        cell_key="SPY:lstm:ict_iofed"
    )
    limit = 471.5
    intent = _limit_intent(limit=limit, sl=469.0, tp=475.5)

    intent.window_raw = _make_window(1)

    exe.on_intent(intent)
    assert exe.open_count() == 1

    # Bar whose low just touches the limit
    bar = _bar(open_=472.0, high=472.5, low=471.4, close=472.0, offset=1)
    exe.on_bar(bar)

    positions = state.open_positions()
    assert len(positions) == 1
    assert positions[0].fill_price == pytest.approx(limit)


def test_limit_fill_conservative_bull(tmp_path):
    """Bull limit fills when bar.close <= limit (conservative)."""
    exe, state = _make_executor(
        tmp_path, strategy="ict_iofed", fill_mode="conservative",
        cell_key="SPY:lstm:ict_iofed"
    )
    limit = 471.5
    intent = _limit_intent(limit=limit, sl=469.0, tp=475.5)

    intent.window_raw = _make_window(1)

    exe.on_intent(intent)
    # Bar whose low touches limit but close is ABOVE — should NOT fill conservative
    bar_no_fill = _bar(472.0, 472.5, 471.4, 471.7, offset=1)
    exe.on_bar(bar_no_fill)
    assert len(state.open_positions()) == 0  # not filled

    # Bar whose close is <= limit — should fill
    bar_fill = _bar(472.0, 472.5, 471.0, 471.3, offset=2)
    exe.on_bar(bar_fill)
    positions = state.open_positions()
    assert len(positions) == 1
    assert positions[0].fill_price == pytest.approx(limit)


def test_limit_no_fill_when_condition_not_met(tmp_path):
    """Limit intent stays pending when bar.low > limit (optimistic bull)."""
    exe, state = _make_executor(
        tmp_path, strategy="ict_iofed", fill_mode="optimistic",
        cell_key="SPY:lstm:ict_iofed"
    )
    limit = 471.5
    intent = _limit_intent(limit=limit, sl=469.0, tp=475.5)

    intent.window_raw = _make_window(1)

    exe.on_intent(intent)
    # Bar well above the limit
    bar = _bar(473.0, 474.0, 472.5, 473.5, offset=1)
    exe.on_bar(bar)

    assert len(state.open_positions()) == 0
    assert exe.open_count() == 1  # still pending


# ---------------------------------------------------------------------------
# SL / TP resolution tests (real compute_exit engine)
# ---------------------------------------------------------------------------


def test_sl_resolution_produces_negative_r(tmp_path):
    """Delivering bars that cross SL produces a resolved SimTrade with r < 0.

    Window geometry: bar_56.high=470.0, SL=469.99 (TICK=0.01).
    Fill bar: low=470.1 > SL (no premature resolution).
    SL bar: low=469.5 < SL → resolves.
    """
    exe, state = _make_executor(tmp_path, strategy="fixed_2r", cell_key="SPY:lstm:fixed_2r")
    intent = _market_intent()
    intent.window_raw = _make_window(1)
    exe.on_intent(intent)
    # Fill at 470.5; low=470.1 clears SL=469.99
    exe.on_bar(_bar(470.5, 471.0, 470.1, 470.5, offset=1))
    assert len(state.open_positions()) == 1

    # SL = 469.99 — bar whose low goes through SL
    exe.on_bar(_bar(470.0, 470.2, 469.5, 469.7, offset=2))

    trades = state.resolved_trades()
    assert len(trades) == 1
    assert trades[0].outcome == "sl"
    assert trades[0].r_multiple < 0


def test_tp_resolution_produces_positive_r(tmp_path):
    """Delivering bars that cross TP produces a resolved SimTrade with r > 0.

    Window geometry: bar_56.high=470.0, SL=469.99 (TICK=0.01).
    Fill at 473.0 (bar open):  R = 473.0 - 469.99 = 3.01, TP = 473 + 2*3.01 = 479.02.
    """
    exe, state = _make_executor(tmp_path, strategy="fixed_2r", cell_key="SPY:lstm:fixed_2r")
    intent = _market_intent()
    intent.window_raw = _make_window(1)
    exe.on_intent(intent)
    # Fill at 473.0; low=470.1 clears SL=469.99; high well below TP=479.02
    exe.on_bar(_bar(473.0, 474.0, 470.1, 473.5, offset=1))
    pos = state.open_positions()
    assert len(pos) == 1

    # SL=469.99, fill=473.0, R=3.01, TP≈479.02
    # Deliver a bar well above TP
    exe.on_bar(_bar(480.0, 481.0, 479.5, 480.5, offset=2))

    trades = state.resolved_trades()
    assert len(trades) == 1
    assert trades[0].outcome == "tp"
    assert trades[0].r_multiple > 0


def test_position_stays_open_until_sl_tp(tmp_path):
    """Bars between SL and TP leave the position open.

    SL=469.99, fill=473.0, TP≈479.02.  Use bars entirely in (470.0, 478.5).
    """
    exe, state = _make_executor(tmp_path, strategy="fixed_2r", cell_key="SPY:lstm:fixed_2r")
    intent = _market_intent()
    intent.window_raw = _make_window(1)
    exe.on_intent(intent)
    # Fill at 473.0; first bar: low=470.1 (above SL), high=474.0 (below TP)
    exe.on_bar(_bar(473.0, 474.0, 470.1, 473.5, offset=1))

    # Subsequent bars well between SL and TP
    for i in range(2, 8):
        exe.on_bar(_bar(473.5, 474.0, 470.1, 473.8, offset=i))

    assert len(state.open_positions()) == 1
    assert len(state.resolved_trades()) == 0


# ---------------------------------------------------------------------------
# on_session_close tests
# ---------------------------------------------------------------------------


def test_session_close_resolves_open_positions(tmp_path):
    exe, state = _make_executor(tmp_path, strategy="fixed_2r", cell_key="SPY:lstm:fixed_2r")
    intent = _market_intent()
    intent.window_raw = _make_window(1)
    exe.on_intent(intent)
    # Fill at 473.0; low=470.1 > SL=469.99; high=474.0 < TP≈479.02
    exe.on_bar(_bar(473.0, 474.0, 470.1, 473.5, offset=1))
    assert len(state.open_positions()) == 1

    exe.on_session_close()

    assert len(state.open_positions()) == 0
    trades = state.resolved_trades()
    assert len(trades) == 1
    assert trades[0].outcome == "session_close"


def test_session_close_cancels_pending_fills(tmp_path):
    exe, state = _make_executor(
        tmp_path, strategy="ict_iofed", fill_mode="optimistic",
        cell_key="SPY:lstm:ict_iofed"
    )
    # Submit a limit far from market — won't fill
    intent = _limit_intent(limit=450.0, sl=448.0, tp=456.0)
    intent.window_raw = _make_window(1)
    exe.on_intent(intent)
    assert exe.open_count() == 1

    exe.on_session_close()

    assert exe.open_count() == 0
    assert len(state.resolved_trades()) == 0  # never filled → no ledger entry


# ---------------------------------------------------------------------------
# Cross-session reload
# ---------------------------------------------------------------------------


def test_cross_session_reload_restores_open_positions(tmp_path):
    """Open positions loaded from FleetState are advanced on new bars."""
    cfg = ExitConfig(strategy="fixed_2r", fill_mode="conservative")
    state1 = FleetState("fleet-x", base_dir=tmp_path)
    exe1 = SimFillExecutor(realism=cfg, state=state1, cell_key="SPY:lstm:fixed_2r")

    intent = _market_intent(h1_ts=BASE_TS)
    intent.window_raw = _make_window(1)
    exe1.on_intent(intent)
    # Fill at 473.0; low=470.1 > SL=469.99; high=474.0 < TP≈479.02
    exe1.on_bar(_bar(473.0, 474.0, 470.1, 473.5, offset=1))

    assert len(state1.open_positions()) == 1

    # --- New session ---
    state2 = FleetState("fleet-x", base_dir=tmp_path)
    exe2 = SimFillExecutor(realism=cfg, state=state2, cell_key="SPY:lstm:fixed_2r")

    # Position should be in state2 already (reloaded)
    assert len(state2.open_positions()) == 1

    # Deliver an SL-crossing bar to the new executor (low=469.5 < SL=469.99)
    exe2.on_bar(_bar(470.0, 470.2, 469.5, 469.7, offset=10))

    trades = state2.resolved_trades()
    assert len(trades) == 1
    assert trades[0].outcome == "sl"


# ---------------------------------------------------------------------------
# Idempotency: FleetState-level (confirmed via executor path)
# ---------------------------------------------------------------------------


def test_no_double_resolution_on_reload(tmp_path):
    """A position resolved in session 1 must not appear in session 2 open list."""
    cfg = ExitConfig(strategy="fixed_2r", fill_mode="conservative")
    state1 = FleetState("fleet-y", base_dir=tmp_path)
    exe1 = SimFillExecutor(realism=cfg, state=state1, cell_key="SPY:lstm:fixed_2r")

    intent = _market_intent(h1_ts=BASE_TS)
    intent.window_raw = _make_window(1)
    exe1.on_intent(intent)
    exe1.on_bar(_bar(473.0, 474.0, 470.1, 473.5, offset=1))   # fill; low > SL=469.99
    exe1.on_bar(_bar(470.0, 470.2, 469.5, 469.7, offset=2))   # SL hit (low=469.5 < 469.99)

    assert len(state1.resolved_trades()) == 1
    assert len(state1.open_positions()) == 0

    # Session 2
    state2 = FleetState("fleet-y", base_dir=tmp_path)
    exe2 = SimFillExecutor(realism=cfg, state=state2, cell_key="SPY:lstm:fixed_2r")

    # No open positions to advance — resolved position not reloaded
    assert len(state2.open_positions()) == 0
    # Resolved trades still in ledger
    assert len(state2.resolved_trades()) == 1
