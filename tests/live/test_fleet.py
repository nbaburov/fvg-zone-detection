"""test_fleet.py — Integration tests for FleetRouter.

Uses:
- Fake adapters (tiny stubs returning fixed probas) — no torch, no XGB
- Stub Alpaca at the SDK boundary — never imported here
- Real LiveWindowBuilder, SimFillExecutor, StrategyOrderPlanner, ExitConfig
  (no mocking our own infrastructure)

Tests assert:
1. Synthetic multi-ticker 1-min bar stream → one intent per
   (ticker, model, strategy) per valid H1 close, dispatched to the
   correct executor.
2. Legality rejects (dup real ticker, out-of-matrix, empty set) raise
   ValueError before any connect.
3. FleetRouter dispatches only to cells whose ticker matches the bar.
4. SimFillExecutor executor gets on_bar called every minute bar.
5. No signal (below threshold) → no intent produced.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.inspect.base import ModelAdapter
from src.live.fleet import FleetRouter, build_cell_matrix, make_cell_key
from src.live.fleet_state import FleetState
from src.live.logger import FleetSessionLogger
from src.live.sim_executor import SimFillExecutor
from src.live.stream import MinuteBar
from src.strategy.exits import ExitConfig


# -----------------------------------------------------------------------
# Fake adapter — returns fixed probas, no file I/O
# -----------------------------------------------------------------------


class _BullAdapter(ModelAdapter):
    """Always predicts bull with confidence 0.9."""
    name = "_test_bull"

    def __init__(self, checkpoint_dir=None, **kwargs):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 1] = 0.9   # bull
        return out


class _NoSignalAdapter(ModelAdapter):
    """Always predicts 'none' (low confidence)."""
    name = "_test_nosignal"

    def __init__(self, checkpoint_dir=None, **kwargs):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 0] = 0.99   # none class dominant
        return out


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_strategy_configs(strategies: list[str], fill_mode: str = "optimistic") -> dict[str, ExitConfig]:
    return {s: ExitConfig(strategy=s, fill_mode=fill_mode) for s in strategies}


def _make_fleet_logger(tmp_path: Path, session: str = "test_session") -> FleetSessionLogger:
    return FleetSessionLogger(session_id=session, base_dir=str(tmp_path / "logs" / "paper"))


def _make_fleet_state(tmp_path: Path, fleet_id: str = "test_fleet") -> FleetState:
    return FleetState(fleet_id=fleet_id, base_dir=tmp_path / "logs")


def _make_registry(
    specs,
    strategy_configs: dict[str, ExitConfig],
    fleet_state: FleetState,
) -> dict:
    from src.live.sim_executor import SimFillExecutor

    registry = {}
    for spec in specs:
        key = make_cell_key(spec.ticker, spec.model, spec.strategy)
        cfg = strategy_configs[spec.strategy]
        executor = SimFillExecutor(realism=cfg, state=fleet_state, cell_key=key)
        registry[key] = executor
    return registry


def _make_spy_bar(
    minute_offset: int,
    base_price: float = 450.0,
    ticker: str = "SPY",
    base_ts: Optional[pd.Timestamp] = None,
) -> MinuteBar:
    """Generate a 1-min bar at 09:30 + minute_offset for the given ticker."""
    if base_ts is None:
        base_ts = pd.Timestamp("2024-01-03 09:30:00", tz="America/New_York")
    ts = base_ts + pd.Timedelta(minutes=minute_offset)
    return MinuteBar(
        symbol=ticker,
        timestamp=ts,
        open=base_price,
        high=base_price + 0.5,
        low=base_price - 0.5,
        close=base_price,
        volume=1000.0,
    )


def _build_warmup_bars(n_h1: int = 62, ticker: str = "SPY") -> list[MinuteBar]:
    """Build 1-min RTH bars spanning exactly n_h1 complete H1 closes.

    Uses a single continuous day so there are no overnight gaps in the H1
    buffer (the live window builder's cross-session gap check fires when
    consecutive H1 bars are > 90 min apart, which would always skip windows
    spanning multiple calendar days).

    To keep all bars within RTH (09:30–15:59), timestamps are kept at
    09:30–15:29 (6 H1 bars per repeat).  For test purposes the timestamp
    date is the same; we bump minutes continuously so the builder accumulates
    unique bars.  The same-session constraint means tests that require a valid
    window event should use ``_make_valid_window_event`` instead of this helper.
    """
    bars: list[MinuteBar] = []
    price = 450.0
    base = pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York")
    for m in range(n_h1 * 60):  # n_h1 × 60 consecutive minutes from 09:30
        ts = base + pd.Timedelta(minutes=m)
        # Keep time within RTH window by clamping: use a fake but RTH-valid ts
        # We just use consecutive minutes from 09:30; only a fraction will be RTH
        bars.append(MinuteBar(
            symbol=ticker,
            timestamp=ts,
            open=price,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=500.0,
        ))
    return bars


def _make_valid_window_event(
    ticker: str = "SPY",
    price: float = 450.0,
) -> "WindowEvent":
    """Build a synthetic valid WindowEvent (no skip_reason).

    Bypasses the live builder's RTH/gap constraints — used in integration
    tests that need to verify fleet dispatch without depending on the builder.
    """
    from src.live.window_builder import H1Bar, WindowEvent
    from src.data.normalize import normalise_window

    ts = pd.Timestamp("2024-06-10 14:30:00", tz="America/New_York")
    raw = np.full((60, 5), price, dtype=np.float64)
    raw[:, 1] = price + 0.5   # high
    raw[:, 2] = price - 0.5   # low
    raw[:, 4] = 1000.0         # volume
    normalised = normalise_window(raw)
    h1 = H1Bar(
        timestamp=ts,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price,
        volume=60_000.0,
    )
    return WindowEvent(
        h1_timestamp=ts,
        h1_bar=h1,
        window=normalised,
        raw_window=raw,
        skip_reason=None,
    )


# -----------------------------------------------------------------------
# Test: legality rejects BEFORE connect (pure matrix builder)
# -----------------------------------------------------------------------


class TestLegalityRejects:
    def test_dup_real_ticker_raises(self):
        with pytest.raises(ValueError, match="Duplicate real ticker"):
            build_cell_matrix(
                models=["lstm", "cnn_lstm"],
                tickers=["SPY"],
                strategies=["fixed_2r"],
                live_subset=["lstm:SPY:fixed_2r", "cnn_lstm:SPY:fixed_2r"],
            )

    def test_out_of_matrix_raises(self):
        with pytest.raises(ValueError, match="not in the resolved cell matrix"):
            build_cell_matrix(
                models=["lstm"],
                tickers=["SPY"],
                strategies=["fixed_2r"],
                live_subset=["lstm:QQQ:fixed_2r"],
            )

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="models"):
            build_cell_matrix([], ["SPY"], ["fixed_2r"], [])

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers"):
            build_cell_matrix(["lstm"], [], ["fixed_2r"], [])

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="strategies"):
            build_cell_matrix(["lstm"], ["SPY"], [], [])


# -----------------------------------------------------------------------
# Test: FleetRouter basic routing — one ticker, one model, one strategy
# -----------------------------------------------------------------------


class TestFleetRouterSingleCell:
    def test_no_intent_during_warmup(self, tmp_path):
        """No intents should be dispatched while builder is still in WARMUP."""
        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        # Only 10 bars — far fewer than the 60 needed for warmup
        for i in range(10):
            router.on_bar(_make_spy_bar(i))

        # No intents should have been dispatched — executor has 0 open
        exec_key = make_cell_key("SPY", "_test_bull", "fixed_2r")
        assert registry[exec_key].open_count() == 0

        fleet_logger.close()

    def test_intent_dispatched_on_valid_window_event(self, tmp_path):
        """A valid WindowEvent fed directly → bull signal → one pending intent."""
        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        # Inject a valid window event directly (bypasses builder warm-up)
        event = _make_valid_window_event(ticker="SPY")
        router._on_window_event("SPY", event)

        exec_key = make_cell_key("SPY", "_test_bull", "fixed_2r")
        # Market intent goes to pending (needs next bar to fill)
        assert registry[exec_key].open_count() >= 1

        fleet_logger.close()

    def test_nosignal_adapter_no_intent(self, tmp_path):
        """A below-threshold adapter must not produce intents."""
        adapters = {"_test_nosignal": _NoSignalAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_nosignal"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        event = _make_valid_window_event(ticker="SPY")
        router._on_window_event("SPY", event)

        exec_key = make_cell_key("SPY", "_test_nosignal", "fixed_2r")
        assert registry[exec_key].open_count() == 0

        fleet_logger.close()


# -----------------------------------------------------------------------
# Test: multi-ticker routing
# -----------------------------------------------------------------------


class TestMultiTickerRouting:
    def test_bar_routes_only_to_correct_ticker(self, tmp_path):
        """A SPY window event must only dispatch to SPY cells, not QQQ cells."""
        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY", "QQQ"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY", "QQQ"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        # Fire a valid event only for SPY
        router._on_window_event("SPY", _make_valid_window_event(ticker="SPY"))

        spy_key = make_cell_key("SPY", "_test_bull", "fixed_2r")
        qqq_key = make_cell_key("QQQ", "_test_bull", "fixed_2r")

        # SPY cell should have a pending intent; QQQ cell should be untouched
        assert registry[spy_key].open_count() >= 1
        assert registry[qqq_key].open_count() == 0

        fleet_logger.close()

    def test_both_tickers_produce_intents(self, tmp_path):
        """Window events for both tickers → both cells receive intents."""
        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY", "QQQ"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY", "QQQ"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        router._on_window_event("SPY", _make_valid_window_event(ticker="SPY"))
        router._on_window_event("QQQ", _make_valid_window_event(ticker="QQQ"))

        spy_key = make_cell_key("SPY", "_test_bull", "fixed_2r")
        qqq_key = make_cell_key("QQQ", "_test_bull", "fixed_2r")
        assert registry[spy_key].open_count() >= 1
        assert registry[qqq_key].open_count() >= 1

        fleet_logger.close()


# -----------------------------------------------------------------------
# Test: cartesian fan-out — N strategies → N intents per H1 close
# -----------------------------------------------------------------------


class TestCartesianFanOut:
    def test_two_strategies_produce_two_intents(self, tmp_path):
        """One model × one ticker × two strategies → both cells receive intents."""
        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r", "ce_50pct"]
        strategy_configs = _make_strategy_configs(strategies, fill_mode="conservative")
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        router._on_window_event("SPY", _make_valid_window_event(ticker="SPY"))

        key_fixed = make_cell_key("SPY", "_test_bull", "fixed_2r")
        key_ce = make_cell_key("SPY", "_test_bull", "ce_50pct")

        # Both cells should have received intents (fixed_2r = pending market fill;
        # ce_50pct = pending limit fill)
        assert registry[key_fixed].open_count() >= 1
        # ce_50pct intent is pending-fill (counted in open_count)
        assert registry[key_ce].open_count() >= 1

        fleet_logger.close()

    def test_two_models_each_dispatch_independently(self, tmp_path):
        """Two adapters on the same ticker → each cell receives its own intents."""
        adapters = {
            "_test_bull": _BullAdapter(),
            "_test_nosignal": _NoSignalAdapter(),
        }
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull", "_test_nosignal"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        router._on_window_event("SPY", _make_valid_window_event(ticker="SPY"))

        bull_key = make_cell_key("SPY", "_test_bull", "fixed_2r")
        nosig_key = make_cell_key("SPY", "_test_nosignal", "fixed_2r")

        assert registry[bull_key].open_count() >= 1
        assert registry[nosig_key].open_count() == 0

        fleet_logger.close()


# -----------------------------------------------------------------------
# Test: on_bar fed to every executor (not only ticker-matched ones)
# -----------------------------------------------------------------------


class TestOnBarFanOut:
    def test_on_bar_called_on_all_executors(self, tmp_path):
        """Every executor.on_bar must be called for every 1-min bar (uniform ABC contract)."""
        # Use mock executors to track on_bar calls
        from src.live.executor_base import Executor

        class _CountingExecutor(Executor):
            def __init__(self):
                self.bar_count = 0
                self.intent_count = 0

            def on_intent(self, intent):
                self.intent_count += 1

            def on_bar(self, bar):
                self.bar_count += 1

            def on_session_close(self):
                pass

            def open_count(self):
                return 0

        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_logger = _make_fleet_logger(tmp_path)

        exec_spy = _CountingExecutor()
        exec_qqq = _CountingExecutor()
        registry = {
            make_cell_key("SPY", "_test_bull", "fixed_2r"): exec_spy,
            make_cell_key("QQQ", "_test_bull", "fixed_2r"): exec_qqq,
        }

        router = FleetRouter(
            tickers=["SPY", "QQQ"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        # Send 5 SPY bars
        for i in range(5):
            router.on_bar(_make_spy_bar(i, ticker="SPY"))

        # Both executors should have received all 5 bars
        assert exec_spy.bar_count == 5
        assert exec_qqq.bar_count == 5

        fleet_logger.close()


# -----------------------------------------------------------------------
# Test: real-vs-sim tagging passes through to registry
# -----------------------------------------------------------------------


class TestRealSimTagging:
    def test_specs_real_xor_sim(self):
        """Cell specs must be real XOR sim — never both."""
        specs = build_cell_matrix(
            models=["lstm", "cnn_lstm"],
            tickers=["SPY", "QQQ"],
            strategies=["fixed_2r", "tradinglab"],
            live_subset=["lstm:SPY:fixed_2r"],
        )
        real = [s for s in specs if s.executor_type == "real"]
        sim = [s for s in specs if s.executor_type == "sim"]
        # Every cell is exactly one type
        assert len(real) + len(sim) == len(specs)
        real_keys = {s.key for s in real}
        sim_keys = {s.key for s in sim}
        assert real_keys.isdisjoint(sim_keys)

    def test_real_cell_identified_correctly(self):
        specs = build_cell_matrix(
            models=["lstm"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=["lstm:SPY:fixed_2r"],
        )
        assert specs[0].executor_type == "real"

    def test_sim_cell_when_not_in_live_subset(self):
        specs = build_cell_matrix(
            models=["lstm", "cnn_lstm"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=["lstm:SPY:fixed_2r"],
        )
        # cnn_lstm:SPY:fixed_2r not in live subset → sim
        sim_spec = next(s for s in specs if s.model == "cnn_lstm")
        assert sim_spec.executor_type == "sim"


# ---------------------------------------------------------------------------
# WS-7: Multi-TF fleet routing tests
# ---------------------------------------------------------------------------


class TestFleetMultiTFRouting:
    """FleetRouter routes bars to builders keyed by (ticker, tf); cells use cell TF."""

    def test_per_cell_tf_resolved_from_specs(self, tmp_path):
        """CellSpec.tf is stamped from model_tfs dict; missing model → 'h1'."""
        from src.live.fleet import build_cell_matrix

        specs = build_cell_matrix(
            models=["_test_bull"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"_test_bull": "5m"},
        )
        assert len(specs) == 1
        assert specs[0].tf == "5m"

    def test_missing_model_tf_defaults_to_h1(self, tmp_path):
        """model_tfs=None → all cells default to tf='h1'."""
        from src.live.fleet import build_cell_matrix

        specs = build_cell_matrix(
            models=["_test_bull"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs=None,
        )
        assert specs[0].tf == "h1"

    def test_builders_keyed_by_ticker_tf(self, tmp_path):
        """FleetRouter creates builders keyed (ticker, tf_token)."""
        from src.live.fleet import build_cell_matrix, FleetRouter

        specs = build_cell_matrix(
            models=["_test_bull"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"_test_bull": "h1"},
        )
        strategy_configs = _make_strategy_configs(["fixed_2r"])
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters={"_test_bull": _BullAdapter()},
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            specs=specs,
        )

        assert ("SPY", "h1") in router.builders
        fleet_logger.close()

    def test_mixed_tf_fleet_has_two_builders_per_ticker(self, tmp_path):
        """A fleet with 5m + 15m models on same ticker creates two builders."""
        from src.live.fleet import build_cell_matrix, FleetRouter

        class _Bull5m(_BullAdapter):
            name = "_bull5m"

        class _Bull15m(_BullAdapter):
            name = "_bull15m"

        specs = build_cell_matrix(
            models=["_bull5m", "_bull15m"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"_bull5m": "5m", "_bull15m": "15m"},
        )
        strategy_configs = _make_strategy_configs(["fixed_2r"])
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters={"_bull5m": _Bull5m(), "_bull15m": _Bull15m()},
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            specs=specs,
        )

        assert ("SPY", "5m") in router.builders
        assert ("SPY", "15m") in router.builders
        assert ("SPY", "h1") not in router.builders
        fleet_logger.close()

    def test_5m_builder_emits_at_5m_boundary_not_h1(self, tmp_path):
        """M5 builder fires every 5 min; SPY-5m cell gets intent before H1 would fire."""
        from src.live.fleet import build_cell_matrix, FleetRouter
        from src.live.stream import MinuteBar

        class _BullM5(_BullAdapter):
            name = "_bullm5"

        # Build a 60-bar M5 warm-up (60 × 5 = 300 1-min bars) + 1 trigger bar
        specs = build_cell_matrix(
            models=["_bullm5"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"_bullm5": "5m"},
        )
        strategy_configs = _make_strategy_configs(["fixed_2r"])
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters={"_bullm5": _BullM5()},
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            specs=specs,
        )

        # Feed 60 complete M5 bars (300 min from 09:30) + trigger next bar
        base = pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York")
        for m in range(300):
            ts = base + pd.Timedelta(minutes=m)
            router.on_bar(MinuteBar(
                symbol="SPY", timestamp=ts,
                open=450.0, high=450.5, low=449.5, close=450.0, volume=1000.0,
            ))

        # Trigger 61st M5 close
        trigger_ts = base + pd.Timedelta(minutes=300)
        exec_key = make_cell_key("SPY", "_bullm5", "fixed_2r")
        open_before = registry[exec_key].open_count()
        router.on_bar(MinuteBar(
            symbol="SPY", timestamp=trigger_ts,
            open=450.0, high=450.5, low=449.5, close=450.0, volume=1000.0,
        ))
        # Should have dispatched at least one intent (the 61st M5 close fired)
        assert registry[exec_key].open_count() > open_before

        fleet_logger.close()

    def test_list_cells_shows_tf_column(self, tmp_path):
        """format_cell_list includes TF column."""
        from src.live.fleet import build_cell_matrix, format_cell_list

        specs = build_cell_matrix(
            models=["_test_bull"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"_test_bull": "5m"},
        )
        output = format_cell_list(specs)
        assert "TF" in output
        assert "5m" in output

    def test_resolve_tf_from_meta_missing_file_defaults_h1(self):
        """resolve_tf_from_meta with nonexistent path returns 'h1'."""
        from src.live.fleet import resolve_tf_from_meta
        assert resolve_tf_from_meta("/nonexistent/path/model.pt") == "h1"

    def test_resolve_tf_from_meta_none_returns_h1(self):
        """resolve_tf_from_meta with None returns 'h1'."""
        from src.live.fleet import resolve_tf_from_meta
        assert resolve_tf_from_meta(None) == "h1"

    def test_resolve_tf_from_meta_reads_token(self, tmp_path):
        """resolve_tf_from_meta reads 'timeframe' key from meta.json."""
        import json
        from src.live.fleet import resolve_tf_from_meta

        meta_path = tmp_path / "model.meta.json"
        meta_path.write_text(json.dumps({"arch": "lstm", "timeframe": "15m"}))
        result = resolve_tf_from_meta(tmp_path / "model.pt")
        assert result == "15m"

    def test_resolve_tf_from_meta_invalid_token_defaults_h1(self, tmp_path):
        """resolve_tf_from_meta with unknown token in meta falls back to 'h1'."""
        import json
        from src.live.fleet import resolve_tf_from_meta

        meta_path = tmp_path / "model.meta.json"
        meta_path.write_text(json.dumps({"arch": "lstm", "timeframe": "badtoken"}))
        result = resolve_tf_from_meta(tmp_path / "model.pt")
        assert result == "h1"


# -----------------------------------------------------------------------
# Test: log_intent wiring and intents.jsonl population
# -----------------------------------------------------------------------


class TestIntentLogging:
    """Verify that FleetRouter calls log_intent for every actionable intent
    and that intents.jsonl is non-empty after dispatch (regression guard for
    the empty-intents bug where the router created trades but never logged
    them, breaking FleetReplayer).
    """

    def test_log_intent_called_on_valid_signal(self, tmp_path):
        """log_intent is called exactly once per (model, strategy) cell when
        the adapter fires a bull signal on a valid window event."""
        from unittest.mock import patch

        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r", "tradinglab"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        call_log: list = []
        original_log_intent = fleet_logger.log_intent

        def _spy(intent):
            call_log.append(intent)
            original_log_intent(intent)

        fleet_logger.log_intent = _spy  # type: ignore[method-assign]

        event = _make_valid_window_event(ticker="SPY")
        router._on_window_event("SPY", event)

        # One cell per strategy (fixed_2r + tradinglab) × 1 model = 2 intents
        assert len(call_log) == 2, (
            f"Expected 2 log_intent calls (one per strategy cell), got {len(call_log)}. "
            "This indicates the router is not calling log_intent."
        )

        fleet_logger.close()

    def test_intents_jsonl_non_empty_after_dispatch(self, tmp_path):
        """Root intents.jsonl must contain >=1 line after a bull window event."""
        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        event = _make_valid_window_event(ticker="SPY")
        router._on_window_event("SPY", event)
        fleet_logger.close()

        # Locate the intents.jsonl written by FleetSessionLogger
        intents_path = Path(tmp_path) / "logs" / "paper" / "test_session" / "intents.jsonl"
        assert intents_path.exists(), f"intents.jsonl not found at {intents_path}"
        lines = [l for l in intents_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 1, "intents.jsonl is empty — log_intent was not called or did not write"

    def test_intent_schema_matches_replayer_fields(self, tmp_path):
        """Every field FleetReplayer reads is present in a logged intent record."""
        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        event = _make_valid_window_event(ticker="SPY")
        router._on_window_event("SPY", event)
        fleet_logger.close()

        import json as _json
        intents_path = Path(tmp_path) / "logs" / "paper" / "test_session" / "intents.jsonl"
        record = _json.loads(intents_path.read_text(encoding="utf-8").splitlines()[0])

        # Fields that FleetReplayer._resolve_intent and run() read directly
        required = {
            "ticker", "model", "strategy", "h1_timestamp",
            "direction", "signal", "confidence",
            "entry", "sl", "tp", "entry_type", "skip_reason",
        }
        missing = required - record.keys()
        assert not missing, f"Intent record missing fields: {missing}"

        # window_raw must NOT be present (excluded per WS-G spec)
        assert "window_raw" not in record, "window_raw must not be serialised into intents.jsonl"

    def test_no_signal_produces_no_intent(self, tmp_path):
        """Below-threshold adapter must not write any intents."""
        adapters = {"_test_nosignal": _NoSignalAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies)
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_nosignal"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        event = _make_valid_window_event(ticker="SPY")
        router._on_window_event("SPY", event)
        fleet_logger.close()

        intents_path = Path(tmp_path) / "logs" / "paper" / "test_session" / "intents.jsonl"
        if intents_path.exists():
            lines = [l for l in intents_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) == 0, "No-signal adapter must not write any intents"

    def test_e2e_logged_intents_resolve_in_fleet_replayer(self, tmp_path):
        """End-to-end: FleetRouter logs intents → FleetReplayer reads them and
        returns a FleetReplayResult with n_intents >= 1.

        This closes the exact loop the empty-intents bug broke: trades resolved
        in SimFillExecutor but FleetReplayer saw 0 intents and returned empty cells.
        """
        import json as _json
        from src.live.fleet_replay import FleetReplayer

        adapters = {"_test_bull": _BullAdapter()}
        strategies = ["fixed_2r"]
        strategy_configs = _make_strategy_configs(strategies, fill_mode="optimistic")
        fleet_state = _make_fleet_state(tmp_path)
        fleet_logger = _make_fleet_logger(tmp_path)

        specs = build_cell_matrix(["_test_bull"], ["SPY"], strategies, [])
        registry = _make_registry(specs, strategy_configs, fleet_state)

        router = FleetRouter(
            tickers=["SPY"],
            adapters=adapters,
            strategy_configs=strategy_configs,
            executor_registry=registry,
            logger=fleet_logger,
            threshold=0.5,
        )

        # Fire a bull window event — should log 1 intent
        event = _make_valid_window_event(ticker="SPY", price=450.0)
        router._on_window_event("SPY", event)

        # Write some 1-min bars to bars_1m_fleet.jsonl so replayer has H1 data
        # to resolve the intent against (need bars AFTER the intent h1_timestamp)
        session_root = Path(tmp_path) / "logs" / "paper" / "test_session"
        bars_path = session_root / "bars_1m_fleet.jsonl"
        intent_ts = event.h1_timestamp  # 2024-06-10 14:30 ET
        # Write 120 1-min bars starting at 14:31 (forward bars for resolution)
        with bars_path.open("a", encoding="utf-8") as fh:
            for m in range(120):
                bar_ts = intent_ts + pd.Timedelta(minutes=m + 1)
                fh.write(_json.dumps({
                    "symbol": "SPY",
                    "timestamp": bar_ts.isoformat(),
                    "open": 450.0,
                    "high": 452.0,
                    "low": 449.0,
                    "close": 451.0,
                    "volume": 1000.0,
                    "is_update": False,
                    "source": "live",
                }) + "\n")

        fleet_logger.close()

        # Replay
        replayer = FleetReplayer(
            session_dir=session_root,
            exit_configs=strategy_configs,
            fleet_id="test_fleet",
        )
        result = replayer.run()

        assert result.cells, "FleetReplayer returned no cells — intents.jsonl was not read"
        total_intents = sum(c.n_intents for c in result.cells)
        assert total_intents >= 1, (
            f"Expected >= 1 intent across all cells, got {total_intents}. "
            "intents.jsonl was likely empty (the bug)."
        )
