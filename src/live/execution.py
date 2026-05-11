"""execution.py — Bracket order submit + position tracking + kill switch.

PaperExecutor validates safety guards, sizes positions, submits bracket orders
via Alpaca TradingClient, tracks fills, and enforces the daily drawdown kill switch.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.live.decision import TradeAction
from src.live.logger import FillEvent, SessionLogger

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)

_BLACKOUT_HOURS = {9, 15}  # 09:30 bar (hour=9) and 15:30 bar (hour=15)


class PaperExecutor:
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
    ) -> None:
        self._client = trading_client
        self._risk_pct = risk_pct
        self._tp_rr = tp_rr
        self._max_sl_pct = max_sl_pct
        self._max_daily_dd = max_daily_dd
        self._max_shares = max_shares
        self._symbol = symbol

        self._is_halted = False
        self._open_positions: int = 0
        self._session_start_equity: Optional[float] = None

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    def on_trade_action(self, action: TradeAction, log: SessionLogger) -> None:
        """Validate guards, size position, submit bracket order."""
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

        # Compute SL/TP
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
        """Update position tracker and check kill switch."""
        if fill.fill_type in ("fill",):
            self._open_positions = max(0, self._open_positions - 1)

        account = self._client.get_account()
        current_equity = float(account.portfolio_value)

        if self._session_start_equity is not None and self._session_start_equity > 0:
            dd = (self._session_start_equity - current_equity) / self._session_start_equity
            if dd >= self._max_daily_dd:
                logger.error("KILL_SWITCH_TRIGGERED drawdown=%.4f", dd)
                self._trigger_kill_switch()

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _trigger_kill_switch(self) -> None:
        self._is_halted = True
        try:
            self._client.cancel_orders()
            self._client.close_all_positions(cancel_orders=True)
        except Exception as exc:
            logger.error("Kill switch cleanup failed: %s", exc)
        logger.error("Kill switch active. Session halted.")
