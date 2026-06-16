"""backfill.py — REST-based warm-up for LiveWindowBuilder.

Fetches historical 1-min RTH bars from Alpaca StockHistoricalDataClient and
feeds them through builder.gap_fill() so the H1 buffer is populated before the
live WebSocket stream starts.

Generalises the inline _backfill_from_rest in scripts/paper_trade.py:
- Configurable warmup_days (vs today-only).
- Works without a live trading account (uses data-only client).
- Optional logger for replay self-containment (§4.10 of the live-fleet plan).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

# Lazy SDK imports are done inside warm() so this module can be imported
# in test environments where alpaca-py may be mocked at the boundary.
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.data.timeframe import H1, Timeframe
from src.data.timeframe import warmup_days as _warmup_days_for
from src.live.stream import MinuteBar
from src.live.window_builder import LiveWindowBuilder

logger = logging.getLogger(__name__)


@runtime_checkable
class _BarLogger(Protocol):
    """Minimal protocol for a logger that supports bar-level tagging.

    Any object with a log_bar(bar, source) method satisfies this protocol.
    Standard Python loggers do not implement log_bar — callers must pass
    an object that does if they want per-bar source tagging.
    """

    def log_bar(self, bar: MinuteBar, source: str) -> None:  # pragma: no cover
        ...


class RestBackfiller:
    """Fetches historical 1-min bars via Alpaca REST and warms a LiveWindowBuilder.

    Parameters
    ----------
    api_key : str
        Alpaca API key (data-only key is sufficient).
    secret_key : str
        Alpaca secret key.
    feed : str
        Alpaca data feed ("iex" or "sip"). Default "iex" (free tier).
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: str = "iex",
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._feed = feed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm(
        self,
        symbol: str,
        builder: LiveWindowBuilder,
        warmup_days: Optional[int] = None,
        logger: Optional[Any] = None,
        tf: Timeframe = H1,
        window_size: int = 60,
    ) -> int:
        """Fetch warm-up bars and feed them into builder.

        The number of calendar days is derived from ``warmup_days(tf, window_size)``
        unless ``warmup_days`` is explicitly supplied as an override.

        After fetching, asserts that ``builder.h1_count >= window_size``.
        The assert fires only on success (skipped if the fetch failed).

        Parameters
        ----------
        symbol : str
            Ticker to fetch (e.g. "SPY").
        builder : LiveWindowBuilder
            Accumulator to warm up.
        warmup_days : int, optional
            Override for how many calendar days to request.  When None (default),
            the value is derived as ``ceil(window_size / tf.bars_per_rth_day) + 1``.
        logger : optional
            Any object with a ``log_bar(bar, source)`` method.  When supplied,
            each fetched bar is logged with ``source="backfill"`` for replay
            self-containment (plan §4.10).  Standard Python loggers are silently
            ignored (they lack ``log_bar``).
        tf : Timeframe
            Timeframe for warmup-depth computation.  Default H1.
        window_size : int
            Sliding window size; used for warmup-depth math.  Default 60.

        Returns
        -------
        int
            ``builder.h1_count`` after warm-up (0 if fetch failed or no bars).
        """
        days = warmup_days if warmup_days is not None else _warmup_days_for(tf, window_size)
        try:
            count = self._fetch_and_feed(symbol, builder, days, logger)
        except Exception as exc:
            module_logger = logging.getLogger(__name__)
            module_logger.warning(
                "REST backfill failed for %s (proceeding without warm-up): %s",
                symbol,
                exc,
            )
            return builder.h1_count

        if count < window_size:
            logging.getLogger(__name__).warning(
                "RestBackfiller.warm: %s warm-up returned only %d bars "
                "(need %d for %s window_size=%d). "
                "Consider increasing warmup_days or check data availability.",
                symbol, count, window_size, tf.token, window_size,
            )
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _session_start(now: pd.Timestamp, warmup_sessions: int) -> pd.Timestamp:
        """Return the 09:30 ET start timestamp ``warmup_sessions`` NYSE sessions back.

        ``warmup_sessions`` is a count of trading sessions (the unit returned by
        :func:`src.data.timeframe.warmup_days`), not calendar days. This walks the
        NYSE session calendar so weekends and holidays are skipped rather than
        eating into the warm-up depth. Falls back to a calendar-day heuristic
        (``warmup_sessions * 2 + 4``) if the calendar lookup is unavailable.
        """
        anchor_kwargs = dict(hour=9, minute=30, second=0, microsecond=0)
        try:
            import exchange_calendars as xcals

            cal = xcals.get_calendar("XNYS")
            # +1 so we span warmup_sessions *complete* prior sessions plus today.
            count = max(2, warmup_sessions + 1)
            window = cal.sessions_window(
                cal.date_to_session(now.normalize().tz_localize(None), direction="previous"),
                -count,
            )
            start_date = pd.Timestamp(window[0])
            return start_date.tz_localize("America/New_York").replace(**anchor_kwargs)
        except Exception:
            calendar_days = warmup_sessions * 2 + 4
            return (now - pd.Timedelta(days=calendar_days)).replace(**anchor_kwargs)

    def _fetch_and_feed(
        self,
        symbol: str,
        builder: LiveWindowBuilder,
        warmup_days: int,
        bar_logger: Optional[Any],
    ) -> int:
        """Core fetch-and-feed logic (raises on error — caller wraps in try/except)."""
        client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        now = pd.Timestamp.now(tz="America/New_York")
        # warmup_days is a SESSION count (see src/data/timeframe.warmup_days),
        # NOT a calendar-day count. Subtracting it directly as calendar days
        # starves the buffer across weekends/holidays (e.g. M5 needs 2 sessions
        # but a Sat/Sun span yields <60 bars → builder stays in WARMUP forever).
        # Convert sessions → a calendar start date via the NYSE calendar.
        start = self._session_start(now, warmup_days)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start.isoformat(),
            end=now.isoformat(),
            adjustment="raw",
            feed=self._feed,
        )

        raw_bars = client.get_stock_bars(req)[symbol]

        if raw_bars is None or len(raw_bars) == 0:
            logging.getLogger(__name__).info(
                "No historical 1-min bars available for %s backfill.", symbol
            )
            return builder.h1_count

        minute_bars: list[MinuteBar] = []
        for bar in raw_bars:
            # Alpaca historical bars carry a python datetime; coerce to a pandas
            # Timestamp so tz_localize/tz_convert are available (live WS bars are
            # already pandas Timestamps — pd.Timestamp(...) is idempotent there).
            ts = pd.Timestamp(bar.timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert("America/New_York")
            mb = MinuteBar(
                symbol=symbol,
                timestamp=ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
            minute_bars.append(mb)
            if isinstance(bar_logger, _BarLogger):
                bar_logger.log_bar(mb, source="backfill")

        logging.getLogger(__name__).info(
            "Backfilling %d 1-min bars for %s warm-up (%d days).",
            len(minute_bars),
            symbol,
            warmup_days,
        )
        builder.gap_fill(minute_bars)
        logging.getLogger(__name__).info(
            "Warm-up complete for %s. H1 buffer: %d bars.", symbol, builder.h1_count
        )
        return builder.h1_count
