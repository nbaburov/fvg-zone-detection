"""Alpaca minute-bar download and H1 resample for equity symbols."""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from src.data.timeframe import H1, Timeframe

logger = logging.getLogger(__name__)

# Baseline threshold tuned for H1 multi-year pulls (7 bars/day × ~252 days/yr × 3 yrs ≈ 5300).
_MIN_MULTIYEAR_BAR_COUNT_H1: int = 5000


@functools.lru_cache(maxsize=1)
def _get_nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def _sanity_check_bar_count(
    bars: pd.DataFrame,
    start: str,
    end: str | None,
    tf: Timeframe = H1,
) -> None:
    """Raise ValueError if a multi-year pull returned suspiciously few bars.

    The threshold is scaled from the H1 baseline by the ratio of bars_per_rth_day,
    so a 5-minute pull (78 bars/day) gets a proportionally higher threshold than H1
    (7 bars/day) and is not falsely rejected.

    Applies when date range > 365 days. Single-day or short pulls are exempt.
    """
    if end is None:
        return
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    if (end_dt - start_dt).days > 365:
        # Scale threshold proportionally to bars_per_rth_day ratio.
        scale = tf.bars_per_rth_day / H1.bars_per_rth_day
        min_bars = int(_MIN_MULTIYEAR_BAR_COUNT_H1 * scale)
        if len(bars) < min_bars:
            raise ValueError(
                f"Sanity check failed: only {len(bars)} {tf.token.upper()} bars for "
                f"{start} to {end}. Expected >= {min_bars} for a multi-year pull."
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


def _resample_minute(minute_df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """Resample minute bars to *tf* with 09:30-anchored bars.

    Uses ``closed='left', label='left'`` — each bar represents the period starting at the
    label time.  RTH filter (09:30–15:59 ET) is applied before resampling so that
    pre/post-market minutes never pollute a bar.

    The resample anchor offset (``tf.rth_offset``) shifts the period boundaries so the
    first bar of every session lands exactly on 09:30 ET regardless of timeframe.

    Args:
        minute_df: DataFrame with a DatetimeIndex of 1-minute bars.
        tf: Timeframe descriptor (e.g. ``H1``, ``M15``, ``M5``).

    Returns:
        Resampled DataFrame with columns open/high/low/close/volume.
    """
    # Ensure tz-aware index in NY time
    if minute_df.index.tz is None:
        minute_df.index = minute_df.index.tz_localize("UTC").tz_convert("America/New_York")
    elif str(minute_df.index.tz) != "America/New_York":
        minute_df.index = minute_df.index.tz_convert("America/New_York")

    # Filter to RTH: 09:30–15:59 inclusive
    minute_df = minute_df.between_time("09:30", "15:59")

    # Resample to tf anchored at 09:30 using rth_offset
    resampled = minute_df.resample(
        tf.pandas_rule, closed="left", label="left", offset=tf.rth_offset
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop empty bars (can occur if no minute bars in that period)
    resampled = resampled.dropna(subset=["open"])

    # Deduplicate on index (guard against Alpaca pagination edge case)
    n_before = len(resampled)
    resampled = resampled[~resampled.index.duplicated(keep="first")]
    n_dropped = n_before - len(resampled)
    if n_dropped:
        logger.warning(
            "Dropped %d duplicate %s index rows after resample",
            n_dropped,
            tf.token.upper(),
        )

    return resampled


# Backward-compat alias — kept because tests/data/test_download.py imports this name directly.
_resample_minute_to_h1 = lambda df: _resample_minute(df, H1)  # noqa: E731


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


def download_bars(
    symbol: str,
    tf: Timeframe = H1,
    start: str = "2018-01-01",
    end: str | None = "2025-12-31",
    use_cache: bool = True,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Download 1-minute bars for *symbol* from Alpaca, resample to *tf*, and return a cleaned DataFrame.

    RTH filter: 09:30–15:59 ET. Resample: closed='left', label='left', offset=tf.rth_offset.
    Validation: _validate_ohlc (zero-volume + OHLC-integrity), adjustment='raw'.

    Returns a DataFrame with DatetimeIndex (America/New_York, bar open time).
    Columns: open, high, low, close, volume (float64), session_type (Categorical: 'full'|'half').
    Index frequency: not guaranteed regular (gaps for holidays/weekends — expected).
    No symbol column is added here; that is the caller's responsibility.

    Args:
        symbol:     Ticker string, e.g. "SPY", "QQQ".
        tf:         Timeframe descriptor. Defaults to H1 (backward-compat).
        start:      ISO date string for the start of the requested range.
        end:        ISO date string for the end of the requested range, or None for today.
        use_cache:  When True, read from / write to a local parquet cache to avoid repeat API calls.
                    The cache always stores raw 1-minute bars; the TF resample is applied on load,
                    so a single cache file supports multiple TF resamples.
        cache_path: Path to the minute-bar parquet cache. Defaults to
                    ``data/raw/<symbol_lower>_minute.parquet``.

    Raises:
        EnvironmentError: if ALPACA_API_KEY or ALPACA_SECRET_KEY not set.
        ValueError:       if returned bar count is too low for a multi-year pull (sanity check,
                          threshold scaled by tf.bars_per_rth_day).
    """
    symbol_upper = symbol.upper()

    if cache_path is None:
        cache_path = f"data/raw/{symbol.lower()}_minute.parquet"

    cache = Path(cache_path)

    # Cache hit: load minute bars from parquet, skip API call
    if use_cache and cache.exists():
        logger.info("Cache hit: loading minute bars from %s", cache)
        minute_df = pd.read_parquet(cache)
        # Guard against silent truncation when caller asks for a start earlier than cache covers.
        cache_start = minute_df.index.min()
        if cache_start.tz is not None:
            cache_start = cache_start.tz_convert("UTC").tz_localize(None)
        requested_start = pd.Timestamp(start)
        if cache_start.date() > requested_start.date():
            logger.warning(
                "Cache starts %s but caller requested %s. Re-downloading.",
                cache_start.date(), requested_start.date(),
            )
        else:
            bars = _resample_minute(minute_df, tf)
            bars = _validate_ohlc(bars)
            bars = _tag_session_type(bars)
            _sanity_check_bar_count(bars, start, end, tf)
            return bars

    # Validate credentials before making any API call
    key, secret = _get_credentials()

    if end is None:
        end = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")

    logger.info("Downloading %s minute bars from Alpaca: %s to %s", symbol_upper, start, end)

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbol_upper,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        adjustment="raw",
    )
    bar_set = client.get_stock_bars(request)
    minute_df = bar_set.df

    # If multi-index (symbol, timestamp), drop symbol level
    if isinstance(minute_df.index, pd.MultiIndex):
        minute_df = minute_df.xs(symbol_upper, level=0)

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
    bars = _resample_minute(minute_df, tf)
    bars = _validate_ohlc(bars)
    bars = _tag_session_type(bars)

    _sanity_check_bar_count(bars, start, end, tf)

    return bars


def download_h1(
    symbol: str,
    start: str = "2018-01-01",
    end: str | None = "2025-12-31",
    use_cache: bool = True,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Thin shim: download and resample to H1. Output is byte-identical to pre-WS-2 behavior.

    All existing callers continue to work without modification.
    See ``download_bars`` for full documentation.
    """
    return download_bars(
        symbol, tf=H1, start=start, end=end, use_cache=use_cache, cache_path=cache_path
    )


def load_class_weights(
    data_dir: Path | str,
    token: str = "h1",
    scope: str = "spy",
) -> list[float]:
    """Load persisted class weights from *data_dir*.

    Path resolution:
        ``<data_dir>/class_weights_{scope}_{token}.json``

    Args:
        data_dir: Directory containing the class_weights JSON file.
        token:    Timeframe token (``"h1"``, ``"5m"``, ``"15m"``).  Defaults to ``"h1"``.
        scope:    Dataset scope (``"spy"`` or ``"multisym"``).  Defaults to ``"spy"``.

    Returns:
        List of floats [w_class0, w_class1, w_class2].
    """
    data_dir = Path(data_dir)
    path = data_dir / f"class_weights_{scope}_{token}.json"
    with path.open() as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [float(w) for w in data]
    return [float(data[str(i)]) for i in range(len(data))]


def download_spy_h1(
    start: str = "2018-01-01",
    end: str | None = "2025-12-31",
    use_cache: bool = True,
    cache_path: str = "data/raw/spy_minute.parquet",
) -> pd.DataFrame:
    """
    Thin wrapper around download_h1 for SPY. Preserved for backward compatibility.

    All existing callers and tests continue to work without modification.
    See download_h1 for full documentation.
    """
    return download_h1("SPY", start=start, end=end, use_cache=use_cache, cache_path=cache_path)
