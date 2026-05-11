"""Tests for download.py — all Alpaca calls mocked, no real API calls."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minute_bars(
    date_str: str = "2020-01-02",
    n_rth: int = 390,
    include_zero_volume: bool = False,
    include_ohlc_violation: bool = False,
    tz_name: str = "America/New_York",
) -> pd.DataFrame:
    """
    Produce a synthetic minute bar DataFrame in the America/New_York timezone.
    n_rth bars starting at 09:30 ET.
    """
    start = pd.Timestamp(f"{date_str} 09:30", tz=tz_name)
    idx = pd.date_range(start, periods=n_rth, freq="1min")
    rng = np.random.default_rng(42)
    prices = 400.0 + rng.normal(0, 0.1, size=n_rth).cumsum()
    prices = np.abs(prices) + 350.0
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.01, 0.2, n_rth),
            "low": prices - rng.uniform(0.01, 0.2, n_rth),
            "close": prices + rng.normal(0, 0.05, n_rth),
            "volume": rng.integers(1000, 5000, n_rth).astype(float),
        },
        index=idx,
    )
    # Ensure OHLC integrity on base data
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    if include_zero_volume:
        df.iloc[100, df.columns.get_loc("volume")] = 0.0

    if include_ohlc_violation:
        # Make high < max(open, close) at row 50
        df.iloc[50, df.columns.get_loc("high")] = (
            min(df.iloc[50]["open"], df.iloc[50]["close"]) - 0.01
        )

    return df


def _make_half_day_bars(date_str: str = "2020-11-27") -> pd.DataFrame:
    """Black Friday: RTH 09:30–13:00 ET (210 minutes)."""
    return _make_minute_bars(date_str=date_str, n_rth=210)


# ---------------------------------------------------------------------------
# Tests: schema and basic contract
# ---------------------------------------------------------------------------

@pytest.fixture
def full_day_minute_bars():
    return _make_minute_bars("2020-01-02", n_rth=390)


@pytest.fixture
def half_day_minute_bars():
    return _make_half_day_bars("2020-11-27")


@pytest.fixture
def two_days_minute_bars():
    """Two full RTH days."""
    d1 = _make_minute_bars("2020-01-02", n_rth=390)
    d2 = _make_minute_bars("2020-01-03", n_rth=390)
    return pd.concat([d1, d2])


@patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
@patch("src.data.download.StockHistoricalDataClient")
def test_output_columns(MockClient, two_days_minute_bars):
    from src.data.download import download_spy_h1

    instance = MockClient.return_value
    bar_set = MagicMock()
    bar_set.df = two_days_minute_bars
    instance.get_stock_bars.return_value = bar_set

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "spy_minute.parquet")
        result = download_spy_h1(start="2020-01-02", end="2020-01-03", use_cache=False, cache_path=cache)

    required = {"open", "high", "low", "close", "volume", "session_type"}
    assert required.issubset(set(result.columns)), f"Missing columns: {required - set(result.columns)}"


@patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
@patch("src.data.download.StockHistoricalDataClient")
def test_index_is_new_york_timezone(MockClient, two_days_minute_bars):
    from src.data.download import download_spy_h1

    instance = MockClient.return_value
    bar_set = MagicMock()
    bar_set.df = two_days_minute_bars
    instance.get_stock_bars.return_value = bar_set

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "spy_minute.parquet")
        result = download_spy_h1(start="2020-01-02", end="2020-01-03", use_cache=False, cache_path=cache)

    assert result.index.tz is not None
    assert "New_York" in str(result.index.tz) or "America" in str(result.index.tz)


@patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
@patch("src.data.download.StockHistoricalDataClient")
def test_no_after_hours_bars(MockClient, two_days_minute_bars):
    """No bar timestamp outside 09:30–15:59 ET should appear in H1 output."""
    from src.data.download import download_spy_h1

    # Add an after-hours minute bar at 16:30 ET
    ah_idx = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-02 16:30", tz="America/New_York")]
    )
    ah_bar = pd.DataFrame(
        {"open": [400.0], "high": [401.0], "low": [399.0], "close": [400.5], "volume": [500.0]},
        index=ah_idx,
    )
    bars_with_ah = pd.concat([two_days_minute_bars, ah_bar]).sort_index()

    instance = MockClient.return_value
    bar_set = MagicMock()
    bar_set.df = bars_with_ah
    instance.get_stock_bars.return_value = bar_set

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "spy_minute.parquet")
        result = download_spy_h1(start="2020-01-02", end="2020-01-03", use_cache=False, cache_path=cache)

    for ts in result.index:
        hour = ts.hour
        minute = ts.minute
        assert (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16, (
            f"Bar at {ts} is outside RTH 09:30–15:59"
        )


@patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
@patch("src.data.download.StockHistoricalDataClient")
def test_half_day_tagged_as_half(MockClient):
    """Black Friday session bars tagged with session_type='half'."""
    from src.data.download import download_spy_h1

    bars = _make_half_day_bars("2020-11-27")
    instance = MockClient.return_value
    bar_set = MagicMock()
    bar_set.df = bars
    instance.get_stock_bars.return_value = bar_set

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "spy_minute.parquet")
        result = download_spy_h1(start="2020-11-27", end="2020-11-27", use_cache=False, cache_path=cache)

    # All bars on 2020-11-27 should be tagged 'half'
    half_day_bars = result[result.index.date == pd.Timestamp("2020-11-27").date()]
    # Assert bars exist before checking their tag — silent pass if empty would hide filter bugs
    assert len(half_day_bars) > 0, "Expected H1 bars on 2020-11-27 (Black Friday) but got none"
    assert (half_day_bars["session_type"] == "half").all(), (
        f"Expected all 2020-11-27 bars tagged 'half', got: {half_day_bars['session_type'].unique()}"
    )


@patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
@patch("src.data.download.StockHistoricalDataClient")
def test_zero_volume_bar_dropped(MockClient, two_days_minute_bars):
    from src.data.download import download_spy_h1
    from src.data.download import _validate_ohlc

    # Inject zero-volume bar
    bars = two_days_minute_bars.copy()
    bars.iloc[5, bars.columns.get_loc("volume")] = 0.0

    instance = MockClient.return_value
    bar_set = MagicMock()
    bar_set.df = bars
    instance.get_stock_bars.return_value = bar_set

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "spy_minute.parquet")
        result = download_spy_h1(start="2020-01-02", end="2020-01-03", use_cache=False, cache_path=cache)

    # All H1 bars should have volume > 0 (zero-vol minute bars are dropped before resample)
    assert (result["volume"] > 0).all()


@patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
@patch("src.data.download.StockHistoricalDataClient")
def test_ohlc_violation_bar_dropped(MockClient):
    from src.data.download import download_spy_h1

    bars = _make_minute_bars("2020-01-02", n_rth=390, include_ohlc_violation=True)

    instance = MockClient.return_value
    bar_set = MagicMock()
    bar_set.df = bars
    instance.get_stock_bars.return_value = bar_set

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "spy_minute.parquet")
        result = download_spy_h1(start="2020-01-02", end="2020-01-02", use_cache=False, cache_path=cache)

    # high >= max(open, close) must hold for all H1 bars
    assert (result["high"] >= result[["open", "close"]].max(axis=1)).all()
    assert (result["low"] <= result[["open", "close"]].min(axis=1)).all()


@patch("src.data.download.load_dotenv")  # prevent .env file from setting vars
def test_environment_error_when_no_creds(mock_dotenv):
    """EnvironmentError raised when API keys are absent."""
    env_without_keys = {k: v for k, v in os.environ.items()
                        if k not in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
    with patch.dict(os.environ, env_without_keys, clear=True):
        from src.data.download import download_spy_h1
        with pytest.raises(EnvironmentError):
            download_spy_h1(use_cache=False)


@patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
@patch("src.data.download.StockHistoricalDataClient")
def test_cache_hit_skips_api(MockClient, two_days_minute_bars):
    """When cache parquet exists and use_cache=True, Alpaca SDK is never called."""
    from src.data.download import download_spy_h1, _resample_minute_to_h1, _tag_session_type, _validate_ohlc

    instance = MockClient.return_value

    # Build a small H1 parquet to serve as cache
    h1 = _resample_minute_to_h1(two_days_minute_bars)
    h1 = _validate_ohlc(h1)
    h1 = _tag_session_type(h1)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "spy_minute.parquet")
        two_days_minute_bars.to_parquet(cache)

        result = download_spy_h1(start="2020-01-02", end="2020-01-03", use_cache=True, cache_path=cache)

    instance.get_stock_bars.assert_not_called()
    assert len(result) > 0
