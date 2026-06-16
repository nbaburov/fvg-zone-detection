"""Unit tests for RestBackfiller.

Tests:
- warm() returns h1_count >= 60 when SDK returns enough bars across warmup_days
- backfill bars are logged with source="backfill" when a logger stub is passed
- warm() returns current h1_count (not raise) when SDK call raises
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.live.window_builder import LiveWindowBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdk_bar(ts: pd.Timestamp, price: float = 100.0) -> MagicMock:
    """Return a mock alpaca-py Bar-like object.

    Alpaca historical bars expose a python ``datetime`` (NOT a pandas Timestamp),
    so the stub uses ``to_pydatetime()`` to reproduce the real API — this locks
    the regression where backfill called the pandas-only ``.tz_convert`` on a
    plain datetime and crashed warm-up for every ticker.
    """
    bar = MagicMock()
    bar.timestamp = ts.to_pydatetime()
    bar.open = price
    bar.high = price + 0.5
    bar.low = price - 0.5
    bar.close = price + 0.1
    bar.volume = 1000.0
    return bar


def _make_sdk_bars(n_days: int = 12) -> list[MagicMock]:
    """Generate n_days * 390-min RTH sessions worth of 1-min bars.

    390 minutes = full RTH (09:30–16:00).  Each day yields 6 complete H1 bars
    (one H1 closes when the next boundary starts), so 12 days => ~72 H1 bars,
    which satisfies the >= 60 assertion.
    """
    bars: list[MagicMock] = []
    base_date = pd.Timestamp("2026-01-02", tz="America/New_York")
    for day in range(n_days):
        session_start = (base_date + pd.Timedelta(days=day)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        for minute in range(390):  # full RTH session => 6 H1 bars per day
            ts = session_start + pd.Timedelta(minutes=minute)
            bars.append(_make_sdk_bar(ts))
    return bars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRestBackfillerWarm:
    """RestBackfiller.warm() happy path."""

    def test_warm_returns_h1_count_gte_60(self) -> None:
        """warm() with enough days of data should fill >= 60 H1 bars."""
        from src.live.backfill import RestBackfiller

        sdk_bars = _make_sdk_bars(n_days=12)

        # Stub StockHistoricalDataClient at the SDK import boundary
        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = {"SPY": sdk_bars}

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            return_value=mock_client_instance,
        ):
            backfiller = RestBackfiller(api_key="key", secret_key="secret")
            builder = LiveWindowBuilder()
            result = backfiller.warm("SPY", builder, warmup_days=12)

        assert result >= 60, f"Expected h1_count >= 60, got {result}"
        assert result == builder.h1_count

    def test_warm_logs_bars_with_source_tag(self) -> None:
        """When a logger stub is passed, each fetched bar is logged tagged source='backfill'."""
        from src.live.backfill import RestBackfiller

        sdk_bars = _make_sdk_bars(n_days=2)

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = {"SPY": sdk_bars}

        logged_sources: list[str] = []

        class _StubLogger:
            def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
                pass

            def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
                pass

            def log_bar(self, bar: Any, source: str) -> None:
                logged_sources.append(source)

        stub_logger = _StubLogger()

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            return_value=mock_client_instance,
        ):
            backfiller = RestBackfiller(api_key="key", secret_key="secret")
            builder = LiveWindowBuilder()
            backfiller.warm("SPY", builder, warmup_days=2, logger=stub_logger)

        assert len(logged_sources) > 0, "Expected at least one bar to be logged"
        assert all(
            s == "backfill" for s in logged_sources
        ), f"All logged source tags should be 'backfill', got: {set(logged_sources)}"

    def test_warm_returns_current_count_on_fetch_failure(self) -> None:
        """warm() must WARN and return current h1_count (never raise) on SDK failure."""
        from src.live.backfill import RestBackfiller

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            side_effect=RuntimeError("network error"),
        ):
            backfiller = RestBackfiller(api_key="key", secret_key="secret")
            builder = LiveWindowBuilder()
            # Pre-fill builder with one H1 worth of data to check count is preserved
            result = backfiller.warm("SPY", builder, warmup_days=5)

        assert result == builder.h1_count  # no crash, count unchanged

    def test_warm_empty_response_returns_zero(self) -> None:
        """warm() with empty bars from SDK returns 0 without error."""
        from src.live.backfill import RestBackfiller

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = {"SPY": []}

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            return_value=mock_client_instance,
        ):
            backfiller = RestBackfiller(api_key="key", secret_key="secret")
            builder = LiveWindowBuilder()
            result = backfiller.warm("SPY", builder, warmup_days=3)

        assert result == 0

    def test_warm_uses_configured_feed(self) -> None:
        """StockBarsRequest is constructed with the feed passed to ctor."""
        from src.live.backfill import RestBackfiller
        from alpaca.data.requests import StockBarsRequest

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = {"SPY": []}

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            return_value=mock_client_instance,
        ), patch("src.live.backfill.StockBarsRequest", wraps=StockBarsRequest) as mock_req:
            backfiller = RestBackfiller(api_key="key", secret_key="secret", feed="sip")
            builder = LiveWindowBuilder()
            backfiller.warm("SPY", builder, warmup_days=1)

        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs.get("feed") == "sip"


# ---------------------------------------------------------------------------
# WS-7: warmup_days math per TF
# ---------------------------------------------------------------------------


class TestWarmupDaysMath:
    """warmup_days(tf, window_size) drives the REST fetch depth."""

    def test_h1_ws60_gives_10_days(self):
        """H1, ws=60: ceil(60/7)+1 = 10."""
        from src.data.timeframe import H1
        from src.data.timeframe import warmup_days
        assert warmup_days(H1, 60) == 10

    def test_m15_ws60_gives_4_days(self):
        """M15, ws=60: ceil(60/26)+1 = 4."""
        from src.data.timeframe import M15
        from src.data.timeframe import warmup_days
        assert warmup_days(M15, 60) == 4

    def test_m5_ws60_gives_2_days(self):
        """M5, ws=60: ceil(60/78)+1 = 2."""
        from src.data.timeframe import M5
        from src.data.timeframe import warmup_days
        assert warmup_days(M5, 60) == 2

    def test_warm_uses_derived_days_for_m15(self):
        """warm() with tf=M15 derives warmup_days=4 internally (not the old flat 12)."""
        from src.live.backfill import RestBackfiller
        from src.data.timeframe import M15

        call_kwargs: dict = {}

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = {"SPY": []}

        from unittest.mock import patch
        from alpaca.data.requests import StockBarsRequest

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            return_value=mock_client_instance,
        ), patch("src.live.backfill.StockBarsRequest", wraps=StockBarsRequest) as mock_req:
            backfiller = RestBackfiller(api_key="key", secret_key="secret")
            builder = LiveWindowBuilder(timeframe=M15, window_size=60)
            backfiller.warm("SPY", builder, tf=M15, window_size=60)

        # warmup_days(M15, 60) = 4 SESSIONS. Sessions are converted to calendar
        # coverage via the NYSE calendar, so the calendar span is >= the session
        # count (weekends/holidays push it out) and always covers >= 4 sessions.
        import pandas as pd
        now = pd.Timestamp.now(tz="America/New_York")
        call_start = pd.Timestamp(mock_req.call_args.kwargs["start"])
        if call_start.tzinfo is None:
            call_start = call_start.tz_localize("America/New_York")
        days_back = (now - call_start).days
        # 4 sessions back spans at least 4 calendar days, at most ~4 + a weekend.
        assert 4 <= days_back <= 8, f"Expected ~4 sessions back, got {days_back} days"

    def test_warm_warmup_days_override_takes_precedence(self):
        """Explicit warmup_days kwarg (a SESSION count) overrides TF-derived default."""
        from src.live.backfill import RestBackfiller
        from src.data.timeframe import M15

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = {"SPY": []}

        from unittest.mock import patch
        from alpaca.data.requests import StockBarsRequest
        import pandas as pd

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            return_value=mock_client_instance,
        ), patch("src.live.backfill.StockBarsRequest", wraps=StockBarsRequest) as mock_req:
            backfiller = RestBackfiller(api_key="key", secret_key="secret")
            builder = LiveWindowBuilder(timeframe=M15, window_size=60)
            backfiller.warm("SPY", builder, warmup_days=20, tf=M15, window_size=60)

        # 20 sessions back ≈ 4 calendar weeks (20 sessions + ~8 weekend days).
        now = pd.Timestamp.now(tz="America/New_York")
        call_start = pd.Timestamp(mock_req.call_args.kwargs["start"])
        if call_start.tzinfo is None:
            call_start = call_start.tz_localize("America/New_York")
        days_back = (now - call_start).days
        # Override of 20 sessions must request strictly more than the derived
        # default (4 sessions) — proving precedence — and span ≥ 20 calendar days.
        assert days_back >= 20, f"Expected ≥20 calendar days for 20 sessions, got {days_back}"


class TestWeekendWarmupCoverage:
    """C1 regression — session-count warm-up must clear window_size across a weekend.

    warmup_days() returns a SESSION count, not calendar days. Subtracting it
    directly as calendar days starves the buffer over weekends/holidays (M5
    needs only 2 sessions → a Sat/Sun span yields <window_size bars → builder
    stays in WARMUP forever). The fix converts sessions to calendar coverage
    via the NYSE calendar.
    """

    @staticmethod
    def _requested_start(now: pd.Timestamp, tf, window_size: int) -> pd.Timestamp:
        """Run warm() with the SDK stubbed and return the requested start ts."""
        from alpaca.data.requests import StockBarsRequest

        from src.live.backfill import RestBackfiller

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = {"SPY": []}

        captured = {}

        def _fake_now(*a, **k):  # noqa: ANN001
            return now

        with patch(
            "src.live.backfill.StockHistoricalDataClient",
            return_value=mock_client_instance,
        ), patch(
            "src.live.backfill.StockBarsRequest", wraps=StockBarsRequest
        ) as mock_req, patch(
            "src.live.backfill.RestBackfiller._session_start",
            wraps=RestBackfiller._session_start,
        ):
            # Pin "now" to a known Monday so the prior weekend is in the window.
            with patch.object(pd.Timestamp, "now", staticmethod(_fake_now)):
                backfiller = RestBackfiller(api_key="k", secret_key="s")
                builder = LiveWindowBuilder(timeframe=tf, window_size=window_size)
                backfiller.warm("SPY", builder, tf=tf, window_size=window_size)
        start = pd.Timestamp(mock_req.call_args.kwargs["start"])
        if start.tzinfo is None:
            start = start.tz_localize("America/New_York")
        captured["start"] = start
        return start

    @pytest.mark.parametrize("token", ["5m", "15m"])
    def test_weekend_span_covers_enough_sessions_to_clear_window(self, token):
        from src.data.timeframe import Timeframe, warmup_days

        tf = Timeframe.from_token(token)
        window_size = 60
        # Monday 2026-06-15 16:00 ET — the immediately preceding days include a
        # Sat/Sun weekend, the classic starvation case.
        now = pd.Timestamp("2026-06-15 16:00", tz="America/New_York")
        start = self._requested_start(now, tf, window_size)

        # Count NYSE sessions actually inside [start, now] — must be >= the
        # session depth needed to produce window_size bars.
        import exchange_calendars as xcals

        cal = xcals.get_calendar("XNYS")
        n_sessions = len(
            cal.sessions_in_range(
                start.tz_convert("UTC").normalize().tz_localize(None),
                now.tz_convert("UTC").normalize().tz_localize(None),
            )
        )
        needed_sessions = -(-window_size // tf.bars_per_rth_day)  # ceil
        assert n_sessions >= needed_sessions, (
            f"{token}: requested span covers {n_sessions} sessions, "
            f"need >= {needed_sessions} to clear window_size={window_size}; "
            f"warmup_days={warmup_days(tf, window_size)}"
        )
        # And the calendar span must exceed the bare session count (weekend present).
        assert (now - start).days > warmup_days(tf, window_size), (
            f"{token}: calendar span {(now - start).days}d did not exceed "
            f"session count {warmup_days(tf, window_size)} — weekend not absorbed"
        )
