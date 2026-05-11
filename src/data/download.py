"""Alpaca minute-bar download and H1 resample for SPY."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_MIN_MULTIYEAR_BAR_COUNT: int = 5000


@functools.lru_cache(maxsize=1)
def _get_nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def _sanity_check_bar_count(h1: pd.DataFrame, start: str, end: str | None) -> None:
    """
    Raise ValueError if a multi-year pull returned suspiciously few H1 bars.
    Applies when date range > 365 days. Single-day or short pulls are exempt.
    """
    if end is None:
        return
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    if (end_dt - start_dt).days > 365 and len(h1) < _MIN_MULTIYEAR_BAR_COUNT:
        raise ValueError(
            f"Sanity check failed: only {len(h1)} H1 bars for "
            f"{start} to {end}. Expected >= {_MIN_MULTIYEAR_BAR_COUNT} for a multi-year pull."
        )


def _get_credentials() -> tuple[str, str]:
    """Load and validate Alpaca credentials from environment."""
    load_dotenv()  # Load .env at call time, not import time
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. "
            "Copy .env.example to .env and fill in your paper account credentials."
        )
    return key, secret


def _resample_minute_to_h1(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute bars to H1 with 09:30-anchored bars.
    Uses closed='left', label='left' — each bar represents the hour starting at the label time.
    """
    # Ensure tz-aware index in NY time
    if minute_df.index.tz is None:
        minute_df.index = minute_df.index.tz_localize("UTC").tz_convert("America/New_York")
    elif str(minute_df.index.tz) != "America/New_York":
        minute_df.index = minute_df.index.tz_convert("America/New_York")

    # Filter to RTH: 09:30–15:59 inclusive
    minute_df = minute_df.between_time("09:30", "15:59")

    # Resample to 1H anchored at 09:30 using offset
    h1 = minute_df.resample("1h", closed="left", label="left", offset="30min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop empty bars (can occur if no minute bars in that hour)
    h1 = h1.dropna(subset=["open"])

    # Deduplicate on index (guard against Alpaca pagination edge case)
    n_before = len(h1)
    h1 = h1[~h1.index.duplicated(keep="first")]
    n_dropped = n_before - len(h1)
    if n_dropped:
        logger.warning("Dropped %d duplicate H1 index rows after resample", n_dropped)

    return h1


def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows violating OHLC integrity rules:
      1. low <= min(open, close)
      2. high >= max(open, close)
      3. low <= high
      4. all OHLC > 0
      5. volume > 0
    Logs count of dropped rows. Does not raise.
    """
    n_before = len(df)

    mask = (
        (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df["high"])
        & (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] > 0)
    )

    df_clean = df[mask].copy()
    n_dropped = n_before - len(df_clean)
    if n_dropped:
        logger.warning("_validate_ohlc: dropped %d rows with integrity violations", n_dropped)

    return df_clean


def _tag_session_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session_type column ('full' or 'half') using exchange_calendars NYSE schedule.
    Half-day sessions identified by early_close on NYSE calendar.
    """
    nyse = _get_nyse()
    df = df.copy()

    # Build set of early-close (half-day) dates
    # exchange_calendars returns a DatetimeIndex for early_closes
    try:
        early_closes = set(nyse.early_closes.date)
    except AttributeError:
        early_closes = set()
        logger.warning("Could not determine early-close dates from NYSE calendar")

    def _session_type(dt: pd.Timestamp) -> str:
        return "half" if dt.date() in early_closes else "full"

    df["session_type"] = pd.Categorical(
        df.index.map(_session_type),
        categories=["full", "half"],
    )
    return df


def download_spy_h1(
    start: str = "2018-01-01",
    end: str | None = None,
    use_cache: bool = True,
    cache_path: str = "data/raw/spy_minute.parquet",
) -> pd.DataFrame:
    """
    Download SPY 1-minute bars from Alpaca, resample to H1, and return cleaned DataFrame.

    Returns H1 DataFrame with DatetimeIndex (America/New_York, bar open time).
    Columns: open, high, low, close, volume (float64), session_type (Categorical: 'full'|'half').
    Index frequency: not guaranteed regular (gaps for holidays/weekends — expected).

    Raises: EnvironmentError if ALPACA_API_KEY or ALPACA_SECRET_KEY not set.
    Raises: ValueError if returned bar count < 5000 (sanity check for full multi-year pull).
    """
    cache = Path(cache_path)

    # Cache hit: load minute bars from parquet, skip API call
    if use_cache and cache.exists():
        logger.info("Cache hit: loading minute bars from %s", cache)
        minute_df = pd.read_parquet(cache)
        h1 = _resample_minute_to_h1(minute_df)
        h1 = _validate_ohlc(h1)
        h1 = _tag_session_type(h1)
        _sanity_check_bar_count(h1, start, end)
        return h1

    # Validate credentials before making any API call
    key, secret = _get_credentials()

    if end is None:
        end = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")

    logger.info("Downloading SPY minute bars from Alpaca: %s to %s", start, end)

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    bar_set = client.get_stock_bars(request)
    minute_df = bar_set.df

    # If multi-index (symbol, timestamp), drop symbol level
    if isinstance(minute_df.index, pd.MultiIndex):
        minute_df = minute_df.xs("SPY", level=0)

    # Ensure tz-aware UTC → NY
    if minute_df.index.tz is None:
        minute_df.index = minute_df.index.tz_localize("UTC").tz_convert("America/New_York")
    elif str(minute_df.index.tz) != "America/New_York":
        minute_df.index = minute_df.index.tz_convert("America/New_York")

    # Rename columns to lowercase if needed
    minute_df.columns = [c.lower() for c in minute_df.columns]

    # Cache raw minute bars
    cache.parent.mkdir(parents=True, exist_ok=True)
    minute_df.to_parquet(cache)
    logger.info("Cached %d minute bars to %s", len(minute_df), cache)

    # Pipeline
    h1 = _resample_minute_to_h1(minute_df)
    h1 = _validate_ohlc(h1)
    h1 = _tag_session_type(h1)

    _sanity_check_bar_count(h1, start, end)

    return h1
