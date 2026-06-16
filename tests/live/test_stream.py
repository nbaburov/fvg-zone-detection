"""Tests for AlpacaBarStream — multi-symbol subscribe + str coercion.

Stubs the Alpaca SDK at the boundary so no real connection is needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.live.stream import AlpacaBarStream, MinuteBar


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _make_alpaca_bar(symbol: str, price: float = 100.0):
    """Return a minimal fake alpaca Bar object."""
    bar = MagicMock()
    bar.symbol = symbol
    bar.timestamp = pd.Timestamp("2024-01-02 10:01:00", tz="UTC")
    bar.open = price
    bar.high = price + 0.5
    bar.low = price - 0.5
    bar.close = price
    bar.volume = 1000.0
    return bar


class FakeStream:
    """Stub StockDataStream that records subscribe_bars calls and immediately ends."""

    def __init__(self, *args, **kwargs):
        self.subscribed_symbols: list[str] = []
        self._bar_handler = None

    def subscribe_bars(self, handler, *symbols):
        self._bar_handler = handler
        self.subscribed_symbols.extend(symbols)

    def subscribe_updated_bars(self, handler, *symbols):
        pass  # optional, not tested here

    async def _run_forever(self):
        # Deliver one bar per subscribed symbol then return (stream ends cleanly)
        for sym in self.subscribed_symbols:
            await self._bar_handler(_make_alpaca_bar(sym))


# ---------------------------------------------------------------------------
# Str-coercion tests
# ---------------------------------------------------------------------------

class TestStrCoercion:
    def test_single_str_coerced_to_list(self):
        received: list[MinuteBar] = []
        stream = AlpacaBarStream(
            api_key="k",
            secret_key="s",
            symbols="SPY",
            on_bar=received.append,
        )
        assert stream._symbols == ["SPY"]
        assert stream._symbol == "SPY"  # legacy attr preserved

    def test_list_preserved(self):
        stream = AlpacaBarStream(
            api_key="k",
            secret_key="s",
            symbols=["SPY", "QQQ", "IWM"],
            on_bar=lambda b: None,
        )
        assert stream._symbols == ["SPY", "QQQ", "IWM"]
        assert stream._symbol == "SPY"  # first element


# ---------------------------------------------------------------------------
# Multi-symbol subscribe tests
# ---------------------------------------------------------------------------

class TestMultiSymbolSubscribe:
    def _run_stream(self, symbols, received):
        """Run the stream with a FakeStream backend; return subscribed symbols."""
        fake = FakeStream()

        with patch.object(AlpacaBarStream, "_build_stream", return_value=fake):
            stream = AlpacaBarStream(
                api_key="k",
                secret_key="s",
                symbols=symbols,
                on_bar=received.append,
            )
            asyncio.run(stream.start())

        return fake.subscribed_symbols

    def test_single_symbol_subscribes_one(self):
        received: list[MinuteBar] = []
        subs = self._run_stream("SPY", received)
        assert subs == ["SPY"]
        assert len(received) == 1
        assert received[0].symbol == "SPY"

    def test_multi_symbol_subscribes_all(self):
        received: list[MinuteBar] = []
        subs = self._run_stream(["SPY", "QQQ"], received)
        assert set(subs) == {"SPY", "QQQ"}
        symbols_received = {b.symbol for b in received}
        assert symbols_received == {"SPY", "QQQ"}

    def test_bars_routed_by_symbol(self):
        """Each bar carries the correct symbol tag from the fake bar objects."""
        received: list[MinuteBar] = []
        self._run_stream(["SPY", "QQQ", "IWM"], received)
        assert {b.symbol for b in received} == {"SPY", "QQQ", "IWM"}

    def test_all_symbols_passed_as_separate_args_to_subscribe_bars(self):
        """subscribe_bars must receive *symbols as separate positional args."""
        captured_args: list[tuple] = []
        orig_subscribe = FakeStream.subscribe_bars

        def recording_subscribe(self_fake, handler, *symbols):
            captured_args.append(symbols)
            orig_subscribe(self_fake, handler, *symbols)

        fake = FakeStream()
        fake.subscribe_bars = lambda handler, *syms: (
            captured_args.append(syms) or orig_subscribe(fake, handler, *syms)
        )

        with patch.object(AlpacaBarStream, "_build_stream", return_value=fake):
            stream = AlpacaBarStream(
                api_key="k",
                secret_key="s",
                symbols=["SPY", "QQQ"],
                on_bar=lambda b: None,
            )
            asyncio.run(stream.start())

        assert len(captured_args) == 1
        assert set(captured_args[0]) == {"SPY", "QQQ"}


# ---------------------------------------------------------------------------
# Regression: Alpaca live/historical bars carry a python datetime, NOT a
# pandas Timestamp.  _bar_to_minutebar must coerce before tz_convert, else
# every live bar throws AttributeError and the fleet processes zero bars.
# ---------------------------------------------------------------------------

class TestBarTimestampCoercion:
    def _stream(self):
        return AlpacaBarStream(
            api_key="k", secret_key="s", symbols=["SPY"], on_bar=lambda b: None,
        )

    def test_python_datetime_timestamp_coerced(self):
        import datetime as _dt

        bar = MagicMock()
        bar.symbol = "SPY"
        bar.timestamp = _dt.datetime(2024, 1, 2, 15, 1, tzinfo=_dt.timezone.utc)
        bar.open = bar.high = bar.low = bar.close = 100.0
        bar.volume = 1000.0

        mb = self._stream()._bar_to_minutebar(bar)
        assert isinstance(mb, MinuteBar)
        assert str(mb.timestamp.tz) == "America/New_York"
        assert mb.timestamp.hour == 10  # 15:01 UTC → 10:01 ET

    def test_naive_python_datetime_assumed_utc(self):
        import datetime as _dt

        bar = MagicMock()
        bar.symbol = "SPY"
        bar.timestamp = _dt.datetime(2024, 1, 2, 15, 1)  # naive
        bar.open = bar.high = bar.low = bar.close = 100.0
        bar.volume = 1000.0

        mb = self._stream()._bar_to_minutebar(bar)
        assert mb.timestamp.hour == 10
