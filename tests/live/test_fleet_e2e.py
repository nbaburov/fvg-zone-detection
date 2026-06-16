"""test_fleet_e2e.py — End-to-end fleet smoke test.

Drives a realistic multi-day 1-min MinuteBar sequence through
FleetRouter.on_bar(...) for a 1-model × 1-ticker × 1-strategy matrix.
Asserts at least one OrderIntent is produced and dispatched.

This test proves that the cross-session fix (LiveWindowBuilder
drop_cross_session=False default) makes signals actually fire end-to-end
through the real builder — not via _on_window_event shortcut.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.inspect.base import ModelAdapter
from src.live.executor_base import Executor
from src.live.fleet import FleetRouter, build_cell_matrix, make_cell_key
from src.live.fleet_state import FleetState
from src.live.logger import FleetSessionLogger
from src.live.order_plan import OrderIntent
from src.live.stream import MinuteBar
from src.strategy.exits import ExitConfig


# ---------------------------------------------------------------------------
# Fake adapter: always predicts bull with confidence 0.9
# ---------------------------------------------------------------------------


class _BullAdapter(ModelAdapter):
    """Always predicts bull with confidence 0.9 — guarantees signals fire."""
    name = "_e2e_bull"

    def __init__(self, checkpoint_dir=None, **kwargs):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 1] = 0.9
        return out


# ---------------------------------------------------------------------------
# Counting executor: records every intent received
# ---------------------------------------------------------------------------


class _IntentCapture(Executor):
    """Records every OrderIntent passed via on_intent."""

    def __init__(self):
        self._intents: list[OrderIntent] = []
        self._bar_count = 0

    def on_intent(self, intent: OrderIntent) -> None:
        self._intents.append(intent)

    def on_bar(self, bar: MinuteBar) -> None:
        self._bar_count += 1

    def on_session_close(self) -> None:
        pass

    def open_count(self) -> int:
        return len(self._intents)

    @property
    def intents(self) -> list[OrderIntent]:
        return list(self._intents)


# ---------------------------------------------------------------------------
# Multi-day 1-min bar generator
# ---------------------------------------------------------------------------

_NY = "America/New_York"


def _make_minute_bars(symbol: str, n_h1_bars: int, base_price: float = 450.0) -> list[MinuteBar]:
    """Generate enough 1-min MinuteBar objects to produce n_h1_bars complete H1 bars.

    Bars are within RTH (09:30–15:59 ET).  We produce 6 RTH H1 slots per day:
    09:30, 10:30, 11:30, 12:30, 13:30, 14:30 (6 H1 bars per day).
    To close an H1 we need 1 bar in the NEXT H1 (triggers the close).

    We generate n_h1_bars + 1 complete-ish H1 slots by seeding the first bar of
    each slot plus one bar in the following slot (enough to close the H1).
    """
    bars: list[MinuteBar] = []
    h1_count = 0
    p = base_price
    rng = np.random.default_rng(seed=42)

    current_day = pd.Timestamp("2026-05-19", tz=_NY)
    while h1_count <= n_h1_bars:
        if current_day.dayofweek >= 5:
            current_day += pd.Timedelta(days=1)
            continue
        for slot in range(6):  # 6 H1 slots: 09:30..14:30
            h1_start = current_day.replace(hour=9, minute=30) + pd.Timedelta(hours=slot)
            # Produce 6 1-min bars for this H1 slot (fills the H1 OHLCV)
            for m in range(6):
                t = h1_start + pd.Timedelta(minutes=m)
                o = p
                hi = o + abs(rng.normal(0, 0.5)) + 0.5
                lo = o - abs(rng.normal(0, 0.3)) - 0.1
                c = o + rng.normal(0, 0.3)
                p = c
                bars.append(MinuteBar(
                    symbol=symbol,
                    timestamp=t,
                    open=float(o),
                    high=float(hi),
                    low=float(lo),
                    close=float(c),
                    volume=float(rng.integers(1000, 10000)),
                    is_update=False,
                ))
            h1_count += 1
            if h1_count > n_h1_bars:
                break
        current_day += pd.Timedelta(days=1)

    return bars


# ---------------------------------------------------------------------------
# E2E smoke test
# ---------------------------------------------------------------------------


class TestFleetE2E:
    def test_at_least_one_intent_produced(self, tmp_path):
        """Driving 65+ H1 bars worth of 1-min data through FleetRouter produces intents.

        This verifies:
        - LiveWindowBuilder.drop_cross_session=False (default) keeps cross-session
          windows, so the builder warms up and produces valid WindowEvents.
        - FleetRouter.on_bar correctly fans out intent to the registered executor.
        - End-to-end: real MinuteBar → FleetRouter → WindowEvent → intent → executor.
        """
        ticker = "SPY"
        model = "_e2e_bull"
        strategy = "fixed_2r"

        # Build the cell matrix and executor registry
        specs = build_cell_matrix([model], [ticker], [strategy], live_subset=[])
        assert len(specs) == 1 and specs[0].executor_type == "sim"

        executor = _IntentCapture()
        registry = {make_cell_key(ticker, model, strategy): executor}

        # Fleet logger (writes to tmp_path)
        fleet_logger = FleetSessionLogger(session_id="e2e_smoke", base_dir=str(tmp_path))

        # Strategy config
        strategy_configs = {strategy: ExitConfig(strategy=strategy)}

        router = FleetRouter(
            tickers=[ticker],
            adapters={model: _BullAdapter()},
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        # Generate enough 1-min bars to warm 60 H1 bars and emit one more
        # (60 complete H1 bars warms the builder; bar 61 triggers first signal)
        bars = _make_minute_bars(ticker, n_h1_bars=62)
        assert len(bars) > 0, "Fixture must produce bars"

        for bar in bars:
            router.on_bar(bar)

        fleet_logger.close()

        # Assert at least one intent was produced and dispatched
        assert len(executor.intents) >= 1, (
            f"Expected at least one OrderIntent after driving 62+ H1 bars of 1-min data. "
            f"Got {len(executor.intents)}. "
            f"Window builder h1_count={router.builders[ticker].h1_count}. "
            f"Check that drop_cross_session=False is the default."
        )

    def test_intent_has_correct_cell_metadata(self, tmp_path):
        """Every produced intent carries correct ticker/model/strategy metadata."""
        ticker = "SPY"
        model = "_e2e_bull"
        strategy = "fixed_2r"

        specs = build_cell_matrix([model], [ticker], [strategy], live_subset=[])
        executor = _IntentCapture()
        registry = {make_cell_key(ticker, model, strategy): executor}

        fleet_logger = FleetSessionLogger(
            session_id="e2e_metadata", base_dir=str(tmp_path)
        )
        router = FleetRouter(
            tickers=[ticker],
            adapters={model: _BullAdapter()},
            strategy_configs={strategy: ExitConfig(strategy=strategy)},
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        bars = _make_minute_bars(ticker, n_h1_bars=62)
        for bar in bars:
            router.on_bar(bar)
        fleet_logger.close()

        if not executor.intents:
            pytest.skip("No intents produced — builder may not have warmed up enough")

        for intent in executor.intents:
            assert intent.ticker == ticker
            assert intent.model == model
            assert intent.strategy == strategy
            assert intent.direction in (1, 2)
            assert intent.signal in ("bull", "bear")

    def test_on_bar_reaches_executor(self, tmp_path):
        """Every 1-min bar is forwarded to the executor via on_bar (ABC contract)."""
        ticker = "SPY"
        model = "_e2e_bull"
        strategy = "fixed_2r"

        executor = _IntentCapture()
        registry = {make_cell_key(ticker, model, strategy): executor}
        fleet_logger = FleetSessionLogger(
            session_id="e2e_on_bar", base_dir=str(tmp_path)
        )
        router = FleetRouter(
            tickers=[ticker],
            adapters={model: _BullAdapter()},
            strategy_configs={strategy: ExitConfig(strategy=strategy)},
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        bars = _make_minute_bars(ticker, n_h1_bars=5)
        n = len(bars)
        for bar in bars:
            router.on_bar(bar)
        fleet_logger.close()

        assert executor._bar_count == n, (
            f"Expected on_bar called {n} times, got {executor._bar_count}"
        )

    def test_below_threshold_no_intents(self, tmp_path):
        """Adapter always predicts 'none' → no intents produced even after warm-up."""

        class _NoneAdapter(ModelAdapter):
            name = "_e2e_none"
            def __init__(self, **kwargs): pass
            def predict_proba(self, w):
                n = w.shape[0]
                out = np.zeros((n, 3), dtype=np.float32)
                out[:, 0] = 0.99
                return out

        ticker = "SPY"
        model = "_e2e_none"
        strategy = "fixed_2r"

        executor = _IntentCapture()
        registry = {make_cell_key(ticker, model, strategy): executor}
        fleet_logger = FleetSessionLogger(
            session_id="e2e_noint", base_dir=str(tmp_path)
        )
        router = FleetRouter(
            tickers=[ticker],
            adapters={model: _NoneAdapter()},
            strategy_configs={strategy: ExitConfig(strategy=strategy)},
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        bars = _make_minute_bars(ticker, n_h1_bars=62)
        for bar in bars:
            router.on_bar(bar)
        fleet_logger.close()

        assert len(executor.intents) == 0, (
            f"Expected no intents from a none-predicting adapter, got {len(executor.intents)}"
        )
