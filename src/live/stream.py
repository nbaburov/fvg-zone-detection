"""stream.py — Alpaca WebSocket 1-min bar subscriber.

Wraps alpaca-py StockDataStream. Delivers MinuteBar events to caller via
an on_bar callback. Handles reconnect with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BACKOFF_DELAYS = [1, 2, 4, 8, 16, 60]


@dataclass
class MinuteBar:
    """A single 1-minute OHLCV bar from Alpaca."""

    symbol: str
    timestamp: pd.Timestamp  # tz=America/New_York
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_update: bool = False


class AlpacaBarStream:
    """Streams 1-minute bars for a single symbol from Alpaca WebSocket.

    Parameters
    ----------
    api_key : str
    secret_key : str
    symbol : str
        Ticker to subscribe to (e.g. "SPY").
    on_bar : Callable[[MinuteBar], None]
        Called on each new or updated bar.
    on_reconnect : Optional[Callable[[], None]]
        Called after a successful reconnect so the window builder can trigger
        REST gap-fill.
    paper : bool
        Use paper feed URL (default True).
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        on_bar: Callable[[MinuteBar], None],
        on_reconnect: Optional[Callable[[], None]] = None,
        paper: bool = True,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._symbol = symbol
        self._on_bar = on_bar
        self._on_reconnect = on_reconnect
        self._paper = paper
        self._stop_event: Optional[asyncio.Event] = None

    def _build_stream(self):
        """Build a new StockDataStream instance."""
        from alpaca.data.live import StockDataStream

        return StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed="iex",
        )

    def _bar_to_minutebar(self, bar, is_update: bool = False) -> MinuteBar:
        """Convert alpaca-py Bar object to MinuteBar dataclass."""
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert("America/New_York")
        return MinuteBar(
            symbol=bar.symbol,
            timestamp=ts,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            is_update=is_update,
        )

    async def start(self) -> None:
        """Connect and stream bars. Blocks until stop() is called or fatal error."""
        self._stop_event = asyncio.Event()
        consecutive_failures = 0

        while not self._stop_event.is_set():
            stream = self._build_stream()

            async def bar_handler(bar):
                mb = self._bar_to_minutebar(bar, is_update=False)
                self._on_bar(mb)

            async def updated_bar_handler(bar):
                mb = self._bar_to_minutebar(bar, is_update=True)
                self._on_bar(mb)

            stream.subscribe_bars(bar_handler, self._symbol)
            # updated_bars is not a standard alpaca-py event — handle gracefully
            try:
                stream.subscribe_updated_bars(updated_bar_handler, self._symbol)
            except AttributeError:
                logger.debug("subscribe_updated_bars not available on this alpaca-py version")

            try:
                logger.info("Connecting to Alpaca WebSocket for %s", self._symbol)
                await stream._run_forever()
                consecutive_failures = 0
                logger.info("Stream ended cleanly.")
                break
            except Exception as exc:
                consecutive_failures += 1
                logger.warning(
                    "Stream error #%d: %s", consecutive_failures, exc
                )
                if consecutive_failures > len(_BACKOFF_DELAYS):
                    logger.error("STREAM_FATAL: exceeded max reconnect attempts. Halting.")
                    raise RuntimeError("STREAM_FATAL") from exc

                delay = _BACKOFF_DELAYS[min(consecutive_failures - 1, len(_BACKOFF_DELAYS) - 1)]
                logger.info("Reconnecting in %ds...", delay)
                await asyncio.sleep(delay)

                if self._on_reconnect is not None:
                    try:
                        self._on_reconnect()
                    except Exception as cb_exc:
                        logger.warning("on_reconnect callback raised: %s", cb_exc)
                consecutive_failures = 0  # reset after successful reconnect call

    async def stop(self) -> None:
        """Signal the stream to stop after current iteration."""
        if self._stop_event is not None:
            self._stop_event.set()
