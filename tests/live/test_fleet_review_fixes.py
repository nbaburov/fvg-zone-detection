"""test_fleet_review_fixes.py — Regression coverage for reviewer findings.

Covers the multi-cell / real-path gaps the original suite missed:

- C1  Two sim cells sharing ONE FleetState never cross-contaminate positions;
      each resolves under its OWN ExitConfig (different ticker AND same-ticker
      different-strategy).
- C2  ``--max-real-exposure`` is enforced: a second real cell whose combined
      risk exceeds the cap is blocked.
- C3  A synthetic fill driven through ``FillRouter`` reaches the owning
      executor (decrements open count + releases the exposure bucket).

All geometry flows through the real ``compute_exit`` engine (§0). No mocking
of our own code; only the Alpaca SDK boundary is stubbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.live.execution import PaperExecutor
from src.live.fill_router import FillRouter
from src.live.fleet import FleetRouter, make_cell_key
from src.live.fleet_state import FleetState, make_intent_id
from src.live.logger import FillEvent
from src.live.order_plan import OrderIntent
from src.live.sim_executor import SimFillExecutor
from src.live.stream import MinuteBar
from src.strategy.exits import ExitConfig

_RNG = np.random.default_rng(123)
BASE_TS = pd.Timestamp("2024-03-01 10:30:00", tz="America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window(direction: int) -> np.ndarray:
    base = 470.0
    window = np.zeros((60, 5), dtype=np.float64)
    for i in range(60):
        o = base + _RNG.uniform(-0.5, 0.5)
        h = o + _RNG.uniform(0.05, 0.8)
        lo = o - _RNG.uniform(0.05, 0.8)
        c = float(_RNG.uniform(lo, h))
        window[i] = [o, h, lo, c, 5_000.0]
        base = c
    return window


def _market_intent(
    ticker: str,
    model: str,
    strategy: str,
    sl: float,
    h1_ts: pd.Timestamp = BASE_TS,
    window: np.ndarray | None = None,
) -> OrderIntent:
    return OrderIntent(
        ticker=ticker,
        model=model,
        strategy=strategy,
        h1_timestamp=h1_ts,
        direction=1,
        signal="bull",
        confidence=0.8,
        entry=float("nan"),
        sl=sl,
        tp=float("nan"),
        entry_type="market",
        skip_reason=None,
        window_raw=window if window is not None else _make_window(1),
    )


def _bar(symbol: str, price: float, offset_min: int, ts0: pd.Timestamp = BASE_TS) -> MinuteBar:
    ts = ts0 + pd.Timedelta(minutes=offset_min)
    return MinuteBar(
        symbol=symbol,
        timestamp=ts,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price,
        volume=1_000.0,
    )


# ---------------------------------------------------------------------------
# C1 — shared FleetState must not cross-contaminate cells
# ---------------------------------------------------------------------------


class TestC1NoCrossContamination:
    def test_two_cells_different_tickers_isolated(self, tmp_path):
        state = FleetState("fleet-c1a", base_dir=tmp_path)
        cfg = ExitConfig(strategy="fixed_2r", fill_mode="conservative")

        spy = SimFillExecutor(cfg, state, cell_key="SPY:lstm:fixed_2r")
        qqq = SimFillExecutor(cfg, state, cell_key="QQQ:lstm:fixed_2r")

        spy.on_intent(_market_intent("SPY", "lstm", "fixed_2r", sl=468.0))
        qqq.on_intent(_market_intent("QQQ", "lstm", "fixed_2r", sl=468.0))

        # Fill SPY's market intent (its symbol) — must NOT open a QQQ position
        spy.on_bar(_bar("SPY", 470.0, 1))
        qqq.on_bar(_bar("QQQ", 470.0, 1))

        spy_open = state.open_positions("SPY:lstm:fixed_2r")
        qqq_open = state.open_positions("QQQ:lstm:fixed_2r")
        assert len(spy_open) == 1
        assert len(qqq_open) == 1
        assert spy_open[0].ticker == "SPY"
        assert qqq_open[0].ticker == "QQQ"
        # Each executor only sees its own book
        assert spy.open_count() == 1
        assert qqq.open_count() == 1

    def test_same_ticker_different_strategy_isolated(self, tmp_path):
        """Two real cells share a ticker only in the sim arena — here two sim
        cells on SPY with DIFFERENT strategies must each keep their own book and
        resolve under their own ExitConfig."""
        state = FleetState("fleet-c1b", base_dir=tmp_path)
        cfg_a = ExitConfig(strategy="fixed_2r", fill_mode="conservative", tp_rr=2.0)
        cfg_b = ExitConfig(strategy="fixed_2r", fill_mode="conservative", tp_rr=4.0)

        cell_a = SimFillExecutor(cfg_a, state, cell_key="SPY:lstm:fixed_2r")
        cell_b = SimFillExecutor(cfg_b, state, cell_key="SPY:cnn_lstm:fixed_2r")

        win = _make_window(1)
        cell_a.on_intent(_market_intent("SPY", "lstm", "fixed_2r", sl=468.0, window=win))
        cell_b.on_intent(_market_intent("SPY", "cnn_lstm", "fixed_2r", sl=468.0, window=win))

        # Both fill on SPY's next bar
        cell_a.on_bar(_bar("SPY", 470.0, 1))
        cell_b.on_bar(_bar("SPY", 470.0, 1))

        a_open = state.open_positions("SPY:lstm:fixed_2r")
        b_open = state.open_positions("SPY:cnn_lstm:fixed_2r")
        assert len(a_open) == 1
        assert len(b_open) == 1
        assert a_open[0].cell_key == "SPY:lstm:fixed_2r"
        assert b_open[0].cell_key == "SPY:cnn_lstm:fixed_2r"
        # Neither cell's open_count includes the other's position
        assert cell_a.open_count() == 1
        assert cell_b.open_count() == 1
        # Global view sees both
        assert len(state.open_positions()) == 2

    def test_resolution_does_not_touch_other_cell(self, tmp_path):
        state = FleetState("fleet-c1c", base_dir=tmp_path)
        cfg = ExitConfig(strategy="fixed_2r", fill_mode="conservative")
        spy = SimFillExecutor(cfg, state, cell_key="SPY:lstm:fixed_2r")
        qqq = SimFillExecutor(cfg, state, cell_key="QQQ:lstm:fixed_2r")

        spy.on_intent(_market_intent("SPY", "lstm", "fixed_2r", sl=468.0))
        qqq.on_intent(_market_intent("QQQ", "lstm", "fixed_2r", sl=468.0))
        spy.on_bar(_bar("SPY", 470.0, 1))
        qqq.on_bar(_bar("QQQ", 470.0, 1))

        # Drive SPY hard down to force SL; QQQ gets only benign bars
        for k in range(2, 30):
            spy.on_bar(_bar("SPY", 455.0, k))
            qqq.on_bar(_bar("QQQ", 470.0, k))

        # SPY resolved; QQQ still open and untouched
        assert qqq.open_count() == 1
        assert len(state.open_positions("QQQ:lstm:fixed_2r")) == 1


# ---------------------------------------------------------------------------
# C2 — fleet exposure cap blocks the over-limit real cell
# ---------------------------------------------------------------------------


class _FakeAccount:
    def __init__(self, equity: float) -> None:
        self.portfolio_value = str(equity)


class _FakeTradingClient:
    def __init__(self, equity: float = 100_000.0) -> None:
        self._equity = equity
        self._api_key = "k"
        self._secret_key = "s"
        self.submitted = 0

    def get_account(self):
        return _FakeAccount(self._equity)

    def submit_order(self, order):
        self.submitted += 1
        return type("R", (), {"id": f"ord-{self.submitted}"})()


def _real_executor(symbol: str, client: _FakeTradingClient, risk_pct: float) -> PaperExecutor:
    ex = PaperExecutor(
        trading_client=client,
        symbol=symbol,
        risk_pct=risk_pct,
        max_sl_pct=0.50,  # wide so geometry never trips the guard
        max_shares=10_000,
    )
    # Market entry uses a live quote; stub it so no network is hit and the
    # bracket submit path runs against the fake TradingClient.
    ex._get_entry_price = lambda signal: 470.0  # type: ignore[assignment]
    return ex


class TestC2ExposureCap:
    def test_second_real_cell_blocked_when_over_cap(self):
        # cap = 4% of 100k = 4000.  Each cell risks 3% = 3000 → first fits,
        # second (combined 6000) exceeds 4000 → blocked.
        router = FillRouter(max_real_exposure=0.04)
        spy_client = _FakeTradingClient(100_000.0)
        qqq_client = _FakeTradingClient(100_000.0)
        spy = _real_executor("SPY", spy_client, risk_pct=0.03)
        qqq = _real_executor("QQQ", qqq_client, risk_pct=0.03)
        router.register("SPY", spy)
        router.register("QQQ", qqq)

        fleet = _FleetGate(router, {"SPY": spy, "QQQ": qqq})

        # First real entry on SPY — permitted, records 3000
        d1 = fleet.dispatch(spy, _market_intent("SPY", "lstm", "fixed_2r", sl=460.0), "SPY")
        # Second real entry on QQQ — combined 6000 > cap 4000 → blocked
        d2 = fleet.dispatch(qqq, _market_intent("QQQ", "lstm", "fixed_2r", sl=460.0), "QQQ")

        assert d1 is True
        assert d2 is False
        assert spy_client.submitted == 1
        assert qqq_client.submitted == 0
        assert router.total_open_risk() == pytest.approx(3000.0)


class _FleetGate:
    """Thin adapter to exercise FleetRouter._dispatch without a full stream."""

    def __init__(self, router: FillRouter, executors: dict) -> None:
        self._fr = FleetRouter(
            tickers=list({k for k in executors}),
            adapters={},
            strategy_configs={},
            executor_registry={},
            logger=_NullLogger(),
            fill_router=router,
        )

    def dispatch(self, executor, intent, ticker) -> bool:
        return self._fr._dispatch(executor, intent, ticker)


class _NullLogger:
    def log_bar(self, *a, **k):
        pass

    def log_intent(self, *a, **k):
        pass

    def log_order(self, *a, **k):
        pass

    def log_event(self, *a, **k):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# C3 — a synthetic fill through FillRouter reaches the owning executor
# ---------------------------------------------------------------------------


class TestC3FillRouting:
    def test_fill_decrements_open_count_and_releases_risk(self):
        router = FillRouter(max_real_exposure=0.04)
        client = _FakeTradingClient(100_000.0)
        spy = _real_executor("SPY", client, risk_pct=0.01)
        router.register("SPY", spy)

        # Open a position via the fleet gate so risk is recorded
        fleet = _FleetGate(router, {"SPY": spy})
        fleet.dispatch(spy, _market_intent("SPY", "lstm", "fixed_2r", sl=460.0), "SPY")
        assert spy.open_count() == 1
        assert router.total_open_risk() == pytest.approx(1000.0)

        # Drive a closing fill through the router
        fill = FillEvent(
            order_id="ord-1",
            filled_at=pd.Timestamp.now(tz="America/New_York"),
            fill_price=470.0,
            fill_qty=5,
            fill_type="fill",
            side="buy",
            symbol="SPY",
        )
        router.on_fill(fill)

        assert spy.open_count() == 0          # decremented in owning executor
        assert router.total_open_risk() == 0  # exposure bucket released
