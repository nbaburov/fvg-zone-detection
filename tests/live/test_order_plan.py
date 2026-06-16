"""test_order_plan.py — TDD tests for OrderIntent + StrategyOrderPlanner.

Coverage
--------
- All 4 strategies × both directions: intent.entry/sl/tp match a direct
  compute_exit(empty future) call (real engine, no mocking).
- entry_type == "market" for fixed_2r; "limit" for all others.
- Degenerate geometry (NaN entry from compute_exit) → skip_reason="DEGENERATE_GEOMETRY".
- Metadata fields (ticker, model, strategy, h1_timestamp, direction, signal,
  confidence) are forwarded verbatim.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.strategy.exits import ExitConfig, compute_exit
from src.live.order_plan import OrderIntent, StrategyOrderPlanner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_window(direction: int) -> np.ndarray:
    """Build a synthetic 60×5 raw OHLCV window that forms a valid FVG pattern.

    Prices are in dollar range (≈ 400) so ATR guard tests don't trip the
    normalised-window assertion in _compute_atr.
    """
    base = 400.0
    window = np.zeros((60, 5), dtype=np.float64)

    # Fill with random plausible H1 bars
    for i in range(60):
        o = base + _RNG.uniform(-1, 1)
        h = o + _RNG.uniform(0.1, 1.5)
        l = o - _RNG.uniform(0.1, 1.5)
        c = _RNG.uniform(l, h)
        window[i] = [o, h, l, c, 10_000.0]
        base = c

    # Stamp a valid 3-candle FVG pattern in bars 56/57/58.
    # For bull:  bar_56.high < bar_58.low  (gap exists).
    # For bear: bar_56.low > bar_58.high  (gap exists).
    if direction == 1:  # bull
        window[56] = [399.0, 400.0, 398.5, 399.5, 5_000.0]   # bar_56: high=400
        window[57] = [399.5, 403.0, 399.0, 402.5, 5_000.0]   # bar_57: impulse up
        window[58] = [402.5, 404.0, 401.0, 403.0, 5_000.0]   # bar_58: low=401 > bar56.high=400
    else:  # bear
        window[56] = [403.0, 403.5, 402.0, 402.5, 5_000.0]   # bar_56: low=402
        window[57] = [402.5, 403.0, 399.0, 399.5, 5_000.0]   # bar_57: impulse down
        window[58] = [399.5, 401.5, 398.0, 398.5, 5_000.0]   # bar_58: high=401.5 < bar56.low=402

    return window


_EMPTY_FUTURE = np.empty((0, 4), dtype=np.float32)

_STRATEGIES = ["fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"]
_DIRECTIONS = [1, 2]

_TS = pd.Timestamp("2025-06-10 10:00:00", tz="America/New_York")


# ---------------------------------------------------------------------------
# Parametrised: entry/sl/tp match compute_exit, entry_type correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", _STRATEGIES)
@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_intent_matches_compute_exit(strategy: str, direction: int) -> None:
    """intent.entry/sl/tp must equal a direct compute_exit(empty future) call."""
    window = _make_window(direction)
    config = ExitConfig(strategy=strategy)
    planner = StrategyOrderPlanner(config)

    intent = planner.plan(
        window,
        direction,
        _TS,
        ticker="SPY",
        model="lstm",
        signal="bull" if direction == 1 else "bear",
        confidence=0.72,
    )

    reference = compute_exit(window, _EMPTY_FUTURE, direction, config)

    assert intent.entry == pytest.approx(reference.entry, nan_ok=True), (
        f"entry mismatch for {strategy} dir={direction}"
    )
    assert intent.sl == pytest.approx(reference.sl, nan_ok=True), (
        f"sl mismatch for {strategy} dir={direction}"
    )
    assert intent.tp == pytest.approx(reference.tp, nan_ok=True), (
        f"tp mismatch for {strategy} dir={direction}"
    )


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_entry_type(strategy: str) -> None:
    """fixed_2r → market; all others → limit."""
    window = _make_window(1)
    config = ExitConfig(strategy=strategy)
    planner = StrategyOrderPlanner(config)
    intent = planner.plan(window, 1, _TS, ticker="SPY", model="xgb",
                          signal="bull", confidence=0.6)
    expected = "market" if strategy == "fixed_2r" else "limit"
    assert intent.entry_type == expected, (
        f"entry_type={intent.entry_type!r} for strategy={strategy!r}"
    )


# ---------------------------------------------------------------------------
# Metadata forwarding
# ---------------------------------------------------------------------------

def test_metadata_forwarded() -> None:
    """ticker, model, strategy, h1_timestamp, direction, signal, confidence
    must all be forwarded verbatim into the intent."""
    window = _make_window(2)
    config = ExitConfig(strategy="tradinglab")
    planner = StrategyOrderPlanner(config)
    ts = pd.Timestamp("2025-06-11 14:00:00", tz="America/New_York")

    intent = planner.plan(
        window, 2, ts,
        ticker="QQQ", model="cnn_lstm", signal="bear", confidence=0.88,
    )

    assert intent.ticker == "QQQ"
    assert intent.model == "cnn_lstm"
    assert intent.strategy == "tradinglab"
    assert intent.h1_timestamp == ts
    assert intent.direction == 2
    assert intent.signal == "bear"
    assert intent.confidence == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Degenerate geometry → skip_reason
# ---------------------------------------------------------------------------

def _make_degenerate_window() -> np.ndarray:
    """Window where bar_56, bar_57 and bar_58 are all NaN.

    This corrupts SL geometry for all strategies:
      - fixed_2r:  sl = bar_56[1] - TICK → NaN.
      - limit variants: sl from bar_56 or bar_57; entry (limit) from bar_58 → NaN.

    The planner must set skip_reason="DEGENERATE_GEOMETRY" for all strategies.
    """
    window = _make_window(1)
    window[56, :] = np.nan
    window[57, :] = np.nan
    window[58, :] = np.nan
    return window


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_degenerate_geometry_sets_skip_reason(strategy: str) -> None:
    """When compute_exit produces NaN SL (all strategies) → DEGENERATE_GEOMETRY."""
    window = _make_degenerate_window()
    config = ExitConfig(strategy=strategy)
    planner = StrategyOrderPlanner(config)

    intent = planner.plan(
        window, 1, _TS,
        ticker="SPY", model="lstm", signal="bull", confidence=0.5,
    )

    assert intent.skip_reason == "DEGENERATE_GEOMETRY", (
        f"Expected DEGENERATE_GEOMETRY for {strategy}, got {intent.skip_reason!r}"
    )


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_no_skip_reason_on_valid_window(strategy: str) -> None:
    """Valid window must produce skip_reason=None for all strategies.

    For fixed_2r the entry is NaN (it is the N+2 bar open, unknown at
    planning time), but the SL is always defined from bar_56 coordinates.
    The planner must NOT flag this as degenerate — the executor resolves the
    entry from the market fill.
    """
    window = _make_window(1)
    config = ExitConfig(strategy=strategy)
    planner = StrategyOrderPlanner(config)
    intent = planner.plan(window, 1, _TS, ticker="SPY", model="lstm",
                          signal="bull", confidence=0.7)
    assert intent.skip_reason is None


# ---------------------------------------------------------------------------
# OrderIntent is a dataclass (smoke)
# ---------------------------------------------------------------------------

def test_order_intent_is_dataclass() -> None:
    """OrderIntent can be constructed directly with keyword args."""
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    intent = OrderIntent(
        ticker="SPY",
        model="lstm",
        strategy="fixed_2r",
        h1_timestamp=ts,
        direction=1,
        signal="bull",
        confidence=0.65,
        entry=400.0,
        sl=398.5,
        tp=403.0,
        entry_type="market",
        skip_reason=None,
    )
    assert intent.ticker == "SPY"
    assert intent.entry_type == "market"
    assert intent.skip_reason is None
