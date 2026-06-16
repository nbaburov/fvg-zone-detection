"""execution.py — Bracket order submit + position tracking + kill switch.

PaperExecutor validates safety guards, sizes positions, submits bracket orders
via Alpaca TradingClient, tracks fills, and enforces the daily drawdown kill switch.

WS-C: PaperExecutor now implements ``Executor`` ABC.
- ``on_intent(intent)`` is the new primary entry point (used by FleetRouter).
  Entry/SL/TP geometry comes from the pre-computed ``OrderIntent`` (produced by
  ``StrategyOrderPlanner`` → ``compute_exit``); no price math is duplicated here.
  Market intents (entry_type="market") use the existing market-bracket path.
  Limit intents (entry_type="limit") submit an Alpaca LIMIT entry with attached
  bracket; unfilled after ``fill_timeout_bars`` H1 closes → cancel + log no_fill.
- ``on_trade_action(action, log)`` is kept as a backward-compatible shim for the
  single-model paper_trade.py script and existing tests.  Its SL/TP inline math
  is preserved exactly so test_execution.py passes unmodified.
- ``on_fill`` is NOT on the Executor ABC (sim has no exchange fills); it is
  exposed here for FillRouter (WS-E).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.live.decision import TradeAction
from src.live.executor_base import Executor
from src.live.logger import FillEvent, SessionLogger
from src.live.order_plan import OrderIntent
from src.live.stream import MinuteBar
from src.data.timeframe import H1, Timeframe, blackout_bars
from src.live.window_builder import _h1_boundary_for, _session_bar_index
from src.strategy.exits import market_tp_from_entry

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient
    from src.live.logger import FleetSessionLogger

logger = logging.getLogger(__name__)

_BLACKOUT_HOURS = {9, 15}  # legacy H1 literal — kept for on_trade_action shim only


def _is_blackout(ts: pd.Timestamp, tf: Timeframe) -> bool:
    """Return True if *ts* falls within the session-open or session-close blackout.

    For a given timeframe, the first and last ``blackout_bars(tf)`` bars of the
    RTH session are blacked out.  The number of complete RTH bars is
    ``tf.bars_per_rth_day``; valid indices are 0 … bars_per_rth_day-1.

    H1 (blackout=1, bars_per_rth_day=7):
        index 0 (09:30) and index 6 (15:30) are blacked out → hours {9, 15}.
        Byte-identical to the old ``_BLACKOUT_HOURS`` check.
    M15 (blackout=2, bars_per_rth_day=26):
        indices 0–1 (09:30, 09:45) and 24–25 (15:00, 15:15) are blacked out.
    M5  (blackout=6, bars_per_rth_day=78):
        indices 0–5 and 72–77 are blacked out.
    """
    n = tf.bars_per_rth_day
    k = blackout_bars(tf)
    idx = _session_bar_index(ts, tf)
    return idx < k or idx >= n - k


class PaperExecutor(Executor):
    """Submits bracket orders to Alpaca paper account.

    Parameters
    ----------
    trading_client : TradingClient
        Alpaca TradingClient configured for paper trading.
    risk_pct : float
        Fraction of equity to risk per trade. Default 0.01 (1%).
    tp_rr : float
        Take-profit R-multiple. Default 2.0 (2R).
    max_sl_pct : float
        Skip if SL distance exceeds this % of entry price. Default 0.03.
    max_daily_dd : float
        Kill switch: halt if daily drawdown >= this fraction. Default 0.02.
    max_shares : int
        Hard cap on position size. Default 50.
    symbol : str
        Symbol to trade. Default "SPY".
    fill_timeout_bars : int
        Number of H1 closes to wait before cancelling an unfilled limit order.
        Default 3.
    """

    def __init__(
        self,
        trading_client: "TradingClient",
        risk_pct: float = 0.01,
        tp_rr: float = 2.0,
        max_sl_pct: float = 0.03,
        max_daily_dd: float = 0.02,
        max_shares: int = 50,
        symbol: str = "SPY",
        fill_timeout_bars: int = 3,
        logger_sink: Optional["SessionLogger | FleetSessionLogger"] = None,
        timeframe: Timeframe = H1,
    ) -> None:
        self._tf = timeframe
        self._client = trading_client
        self._risk_pct = risk_pct
        self._tp_rr = tp_rr
        self._max_sl_pct = max_sl_pct
        self._max_daily_dd = max_daily_dd
        self._max_shares = max_shares
        self._symbol = symbol
        self._fill_timeout_bars = fill_timeout_bars
        # Held logger (set at construction) so the fleet path satisfies the
        # uniform ``Executor.on_intent(intent)`` ABC while still logging orders
        # (M2).  May be a SessionLogger (single-model) or a FleetSessionLogger.
        self._log_sink = logger_sink

        self._is_halted = False
        self._open_positions: int = 0
        self._session_start_equity: Optional[float] = None

        # Pending limit orders: order_id -> bars_waited counter
        self._pending_limits: dict[str, int] = {}
        self._last_h1_boundary: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Executor ABC
    # ------------------------------------------------------------------

    def on_intent(self, intent: OrderIntent) -> None:
        """Process an ``OrderIntent`` from ``StrategyOrderPlanner``.

        Matches the uniform ``Executor.on_intent(intent)`` ABC (M2): the
        logger is the instance ``_log_sink`` set at construction, not a
        per-call argument.  This is how the fleet drives both real and sim
        cells through the same contract.

        Drops intents with a ``skip_reason`` (logs them).
        Market intents (fixed_2r) use the market-bracket path.
        Limit intents use the limit-bracket path.

        Parameters
        ----------
        intent : OrderIntent
            Fully-resolved intent from ``StrategyOrderPlanner``.
        """
        log = self._log_sink
        # Drop degenerate intents
        if intent.skip_reason is not None:
            msg = {"reason": intent.skip_reason, "ts": str(intent.h1_timestamp)}
            if log:
                log.log_event("SKIP", msg)
            logger.info("SKIP intent skip_reason=%s ts=%s", intent.skip_reason, intent.h1_timestamp)
            return

        # Guard 1: kill switch
        if self._is_halted:
            msg = {"reason": "SKIP_KILL_SWITCH", "ts": str(intent.h1_timestamp)}
            if log:
                log.log_event("SKIP", msg)
            return

        # Guard 2: no-op signal
        if intent.signal == "none":
            return

        # Guard 3: blackout window (session-relative, TF-aware)
        if _is_blackout(intent.h1_timestamp, self._tf):
            msg = {"reason": "SKIP_BLACKOUT", "ts": str(intent.h1_timestamp)}
            if log:
                log.log_event("SKIP", msg)
            logger.info("SKIP_BLACKOUT at %s", intent.h1_timestamp)
            return

        # Guard 4: open position
        if self._open_positions > 0:
            msg = {"reason": "SKIP_OPEN_POSITION", "ts": str(intent.h1_timestamp)}
            if log:
                log.log_event("SKIP", msg)
            logger.info("SKIP_OPEN_POSITION at %s", intent.h1_timestamp)
            return

        # Fetch account equity
        account = self._client.get_account()
        equity = float(account.portfolio_value)
        if self._session_start_equity is None:
            self._session_start_equity = equity

        # Determine entry price for sizing
        if intent.entry_type == "market":
            # For market intents, entry is not known until fill; fetch quote
            entry_price = self._get_entry_price(intent.signal)
            if entry_price is None:
                if log:
                    log.log_event("SKIP", {"reason": "SKIP_NO_QUOTE", "ts": str(intent.h1_timestamp)})
                return
            sl_price = round(intent.sl, 2)
            sl_dist = abs(entry_price - sl_price)
            # TP geometry at fill time goes through the single exits.py helper
            # (§0 keystone) — no exit arithmetic inline here (H2).
            tp_price = round(
                market_tp_from_entry(
                    entry=entry_price,
                    sl=sl_price,
                    tp_rr=self._tp_rr,
                    direction=intent.direction,
                ),
                2,
            )
        else:
            # Limit entry: entry/sl/tp all come from the intent
            entry_price = intent.entry
            sl_price = round(intent.sl, 2)
            tp_price = round(intent.tp, 2)
            sl_dist = abs(entry_price - sl_price)

        # Guard 5: SL too wide
        sl_pct = sl_dist / entry_price if entry_price > 0 else 1.0
        if sl_pct > self._max_sl_pct:
            msg = {"reason": "SKIP_SL_TOO_WIDE", "sl_pct": sl_pct, "ts": str(intent.h1_timestamp)}
            if log:
                log.log_event("SKIP", msg)
            logger.info("SKIP_SL_TOO_WIDE sl_pct=%.4f", sl_pct)
            return

        # Guard 6: position sizing
        if sl_dist <= 0:
            if log:
                log.log_event("SKIP", {"reason": "SKIP_SIZING", "ts": str(intent.h1_timestamp)})
            return
        qty = min(int(math.floor(equity * self._risk_pct / sl_dist)), self._max_shares)
        if qty < 1:
            if log:
                log.log_event("SKIP", {"reason": "SKIP_SIZING", "ts": str(intent.h1_timestamp)})
            logger.info("SKIP_SIZING qty=0")
            return

        # Submit order — branch on entry type
        if intent.entry_type == "market":
            order_id = self._submit_bracket(intent.signal, qty, sl_price, tp_price)
        else:
            order_id = self._submit_limit_bracket(intent.signal, qty, entry_price, sl_price, tp_price)

        if order_id is None:
            return

        self._open_positions += 1
        if intent.entry_type == "limit":
            self._pending_limits[order_id] = 0

        if log is not None:
            # Log the OrderIntent directly (L3): it duck-types the only field
            # log_order reads (``.signal``).  FleetSessionLogger.log_order needs
            # the cell coordinates, which the intent carries.
            self._log_order(log, order_id, intent, qty, sl_price, tp_price)

        logger.info(
            "ORDER submitted: %s %s qty=%d sl=%.2f tp=%.2f id=%s entry_type=%s",
            intent.signal, self._symbol, qty, sl_price, tp_price, order_id, intent.entry_type,
        )

    def on_bar(self, bar: MinuteBar) -> None:
        """Advance limit-order timeout counters on each H1 boundary crossing.

        Accepts every 1-min bar (uniform `Executor.on_bar` contract shared
        with `SimFillExecutor`). The timeout counter advances once per *H1
        boundary crossing*, detected internally from the bar timestamp, so
        ``fill_timeout_bars`` counts H1 closes regardless of feed cadence.
        Orders exceeding the timeout are cancelled and logged ``no_fill``.
        """
        if not self._pending_limits:
            return

        boundary = _h1_boundary_for(bar.timestamp)
        if self._last_h1_boundary is not None and boundary <= self._last_h1_boundary:
            return  # same (or earlier) H1 — no new close to count
        self._last_h1_boundary = boundary

        timed_out = []
        for order_id in list(self._pending_limits):
            self._pending_limits[order_id] += 1
            if self._pending_limits[order_id] >= self._fill_timeout_bars:
                timed_out.append(order_id)

        for order_id in timed_out:
            self._cancel_limit_order(order_id)
            del self._pending_limits[order_id]
            self._open_positions = max(0, self._open_positions - 1)
            logger.info("LIMIT_TIMEOUT order_id=%s → no_fill", order_id)

    def open_count(self) -> int:
        """Return the number of currently open positions."""
        return self._open_positions

    # ------------------------------------------------------------------
    # Backward-compatible shim (used by paper_trade.py + test_execution.py)
    # ------------------------------------------------------------------

    def on_trade_action(self, action: TradeAction, log: SessionLogger) -> None:
        """Validate guards, size position, submit bracket order.

        This is a backward-compatible shim for the single-model
        ``paper_trade.py`` CLI and the existing ``test_execution.py`` suite.
        The SL/TP inline math (gap_low - 0.01 / gap_high + 0.01) is
        preserved exactly to keep old tests green.

        New code should use ``on_intent`` with a pre-planned ``OrderIntent``
        from ``StrategyOrderPlanner``.
        """
        # Guard 1: kill switch
        if self._is_halted:
            log.log_event("SKIP", {"reason": "SKIP_KILL_SWITCH", "ts": str(action.h1_timestamp)})
            return

        # Guard 2: no-op signal (already logged by decision layer)
        if action.signal == "none":
            return

        # Guard 3: blackout window
        if action.h1_timestamp.hour in _BLACKOUT_HOURS:
            log.log_event("SKIP", {"reason": "SKIP_BLACKOUT", "ts": str(action.h1_timestamp)})
            logger.info("SKIP_BLACKOUT at %s", action.h1_timestamp)
            return

        # Guard 4: open position
        if self._open_positions > 0:
            log.log_event("SKIP", {"reason": "SKIP_OPEN_POSITION", "ts": str(action.h1_timestamp)})
            logger.info("SKIP_OPEN_POSITION at %s", action.h1_timestamp)
            return

        # Fetch current account state
        account = self._client.get_account()
        equity = float(account.portfolio_value)
        if self._session_start_equity is None:
            self._session_start_equity = equity

        # Fetch latest quote for entry price estimate
        entry_price = self._get_entry_price(action.signal)
        if entry_price is None:
            log.log_event("SKIP", {"reason": "SKIP_NO_QUOTE", "ts": str(action.h1_timestamp)})
            return

        # Compute SL/TP — legacy inline math (preserved for shim correctness)
        if action.signal == "bull":
            sl_price = round(action.gap_low - 0.01, 2)
            sl_dist = abs(entry_price - sl_price)
            tp_price = round(entry_price + self._tp_rr * sl_dist, 2)
        else:
            sl_price = round(action.gap_high + 0.01, 2)
            sl_dist = abs(sl_price - entry_price)
            tp_price = round(entry_price - self._tp_rr * sl_dist, 2)

        # Guard 5: SL too wide
        sl_pct = sl_dist / entry_price if entry_price > 0 else 1.0
        if sl_pct > self._max_sl_pct:
            log.log_event("SKIP", {
                "reason": "SKIP_SL_TOO_WIDE",
                "sl_pct": sl_pct,
                "ts": str(action.h1_timestamp),
            })
            logger.info("SKIP_SL_TOO_WIDE sl_pct=%.4f", sl_pct)
            return

        # Guard 6: position sizing
        if sl_dist <= 0:
            log.log_event("SKIP", {"reason": "SKIP_SIZING", "ts": str(action.h1_timestamp)})
            return
        qty = min(int(math.floor(equity * self._risk_pct / sl_dist)), self._max_shares)
        if qty < 1:
            log.log_event("SKIP", {"reason": "SKIP_SIZING", "ts": str(action.h1_timestamp)})
            logger.info("SKIP_SIZING qty=0")
            return

        # Submit bracket order
        order_id = self._submit_bracket(action.signal, qty, sl_price, tp_price)
        if order_id is None:
            return

        self._open_positions += 1
        log.log_order(order_id, action, qty, sl_price, tp_price, self._symbol)
        logger.info(
            "ORDER submitted: %s %s qty=%d sl=%.2f tp=%.2f id=%s",
            action.signal, self._symbol, qty, sl_price, tp_price, order_id,
        )

    def on_fill_event(self, fill: FillEvent) -> None:
        """Update position tracker and check kill switch.

        Called by ``FillRouter`` (WS-E).  Not on the Executor ABC.
        """
        if fill.fill_type in ("fill",):
            self._open_positions = max(0, self._open_positions - 1)
            # If this was a pending limit order, remove from timeout tracker
            self._pending_limits.pop(fill.order_id, None)

        account = self._client.get_account()
        current_equity = float(account.portfolio_value)

        if self._session_start_equity is not None and self._session_start_equity > 0:
            dd = (self._session_start_equity - current_equity) / self._session_start_equity
            if dd >= self._max_daily_dd:
                logger.error("KILL_SWITCH_TRIGGERED drawdown=%.4f", dd)
                self._trigger_kill_switch()

    # on_fill is an alias so FillRouter can call executor.on_fill(fill_event)
    def on_fill(self, fill: FillEvent) -> None:
        """Alias for ``on_fill_event`` — used by ``FillRouter`` (WS-E)."""
        self.on_fill_event(fill)

    def on_session_close(self) -> None:
        """Cancel all open orders and flatten positions."""
        try:
            self._client.cancel_orders()
            logger.info("All orders cancelled at session close.")
        except Exception as exc:
            logger.warning("Could not cancel orders at session close: %s", exc)

        try:
            self._client.close_all_positions(cancel_orders=True)
            logger.info("All positions flattened at session close.")
        except Exception as exc:
            logger.warning("Could not flatten positions at session close: %s", exc)

        self._open_positions = 0
        self._pending_limits.clear()

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    @property
    def risk_pct(self) -> float:
        """Per-trade risk fraction (used by the fleet exposure gate)."""
        return self._risk_pct

    def current_equity(self) -> Optional[float]:
        """Fetch the account portfolio value, or ``None`` if unavailable.

        Used by ``FleetRouter`` to size the fleet-level exposure gate
        (``FillRouter.can_enter``) before a real submit.
        """
        try:
            account = self._client.get_account()
            return float(account.portfolio_value)
        except Exception as exc:  # pragma: no cover - network/SDK guard
            logger.warning("current_equity: could not fetch account: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_order(
        self,
        log: "SessionLogger | FleetSessionLogger",
        order_id: str,
        intent: OrderIntent,
        qty: int,
        sl_price: float,
        tp_price: float,
    ) -> None:
        """Log a submitted order through either logger flavour.

        ``FleetSessionLogger.log_order`` needs the cell coordinates
        (ticker/model/strategy); the single-model ``SessionLogger`` does not.
        We branch on the presence of the fleet signature so real fleet cells
        log via ``FleetSessionLogger`` (M2).
        """
        from src.live.logger import FleetSessionLogger

        if isinstance(log, FleetSessionLogger):
            log.log_order(
                intent.ticker,
                intent.model,
                intent.strategy,
                order_id,
                intent,
                qty,
                sl_price,
                tp_price,
            )
        else:
            log.log_order(order_id, intent, qty, sl_price, tp_price, self._symbol)

    def _get_entry_price(self, signal: str) -> Optional[float]:
        """Fetch current ask (buy) or bid (sell) from Alpaca snapshot."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            # Re-use client credentials from trading client if available
            data_client = StockHistoricalDataClient(
                api_key=self._client._api_key,
                secret_key=self._client._secret_key,
            )
            req = StockLatestQuoteRequest(symbol_or_symbols=self._symbol)
            quote = data_client.get_stock_latest_quote(req)[self._symbol]
            if signal == "bull":
                return float(quote.ask_price) if quote.ask_price else float(quote.bid_price)
            else:
                return float(quote.bid_price) if quote.bid_price else float(quote.ask_price)
        except Exception as exc:
            logger.warning("Could not fetch entry price quote: %s", exc)
            return None

    def _submit_bracket(
        self,
        signal: str,
        qty: int,
        sl_price: float,
        tp_price: float,
    ) -> Optional[str]:
        """Submit bracket market order. Returns order_id or None on failure."""
        try:
            from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
            from alpaca.trading.requests import (
                MarketOrderRequest,
                StopLossRequest,
                TakeProfitRequest,
            )

            order = MarketOrderRequest(
                symbol=self._symbol,
                qty=qty,
                side=OrderSide.BUY if signal == "bull" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=sl_price),
                take_profit=TakeProfitRequest(limit_price=tp_price),
            )
            result = self._client.submit_order(order)
            return str(result.id)
        except Exception as exc:
            logger.error("Order submission failed: %s", exc)
            return None

    def _submit_limit_bracket(
        self,
        signal: str,
        qty: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
    ) -> Optional[str]:
        """Submit bracket limit-entry order. Returns order_id or None on failure."""
        try:
            from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
            from alpaca.trading.requests import (
                LimitOrderRequest,
                StopLossRequest,
                TakeProfitRequest,
            )

            order = LimitOrderRequest(
                symbol=self._symbol,
                qty=qty,
                side=OrderSide.BUY if signal == "bull" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=entry_price,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=sl_price),
                take_profit=TakeProfitRequest(limit_price=tp_price),
            )
            result = self._client.submit_order(order)
            return str(result.id)
        except Exception as exc:
            logger.error("Limit order submission failed: %s", exc)
            return None

    def _cancel_limit_order(self, order_id: str) -> None:
        """Cancel a single limit order by id."""
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Cancelled limit order %s (timeout)", order_id)
        except Exception as exc:
            logger.warning("Could not cancel limit order %s: %s", order_id, exc)

    def _trigger_kill_switch(self) -> None:
        self._is_halted = True
        try:
            self._client.cancel_orders()
            self._client.close_all_positions(cancel_orders=True)
        except Exception as exc:
            logger.error("Kill switch cleanup failed: %s", exc)
        logger.error("Kill switch active. Session halted.")
