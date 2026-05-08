"""Shared pytest fixtures for SMC data pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_df(ohlcv: list[tuple]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with 1h DatetimeIndex."""
    idx = pd.date_range("2020-01-02 09:30", periods=len(ohlcv), freq="1h", tz="America/New_York")
    return pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume"], index=idx)


@pytest.fixture
def spy_9candle_fvg() -> pd.DataFrame:
    """
    9-candle synthetic DataFrame with exactly one bullish FVG (label at 4)
    and one bearish FVG (label at 7). All other positions label = 0.

    Bullish FVG at i=3: high[2]=100.5 < low[4]=103.0, close[3]>open[3].
    Bearish FVG at i=6: low[5]=103.0 > high[7]=102.0, close[6]<open[6].
    """
    data = [
        (100.0, 100.5, 99.5,  100.0, 1000.0),   # 0: neutral
        (100.0, 101.5, 99.5,  100.0, 1000.0),   # 1: neutral (high=101.5 prevents gap at i=2)
        (99.8,  100.5, 99.5,  100.0, 1000.0),   # 2: i-1 for bull FVG; high=100.5
        (101.0, 104.5, 100.8, 104.0, 1500.0),   # 3: bull FVG center (close>open); label→4
        (103.5, 104.0, 103.0, 103.5, 1200.0),   # 4: bull FVG label; low=103.0>high[2]=100.5
        (103.8, 104.5, 103.0, 103.8, 1000.0),   # 5: neutral bridge; low=103.0 (i-1 for bear at 6)
        (110.0, 111.0, 103.5, 106.0, 1500.0),   # 6: bear FVG center (close<open); label→7
        (101.5, 102.0, 101.0, 101.5, 1200.0),   # 7: bear FVG label; high=102.0<low[5]=103.0
        (101.0, 101.5, 100.5, 101.0, 1000.0),   # 8: last candle; always 0
    ]
    return _make_df(data)


@pytest.fixture
def spy_h1_fixture() -> pd.DataFrame:
    """
    500-row synthetic H1 DataFrame with RTH structure.
    One half-day session and one 3-day holiday gap injected.
    """
    rng = np.random.default_rng(1337)

    # Continuous 1h bars for most of the series
    n_main = 450
    idx_main = pd.date_range("2020-01-06 09:30", periods=n_main, freq="1h", tz="America/New_York")

    # Holiday gap: 3-day break (skip ~72 hours) then 50 more bars
    n_tail = 50
    gap_start = idx_main[-1] + pd.Timedelta(hours=73)
    idx_tail = pd.date_range(gap_start, periods=n_tail, freq="1h")

    idx = idx_main.append(idx_tail)
    n = len(idx)

    prices = 400.0 + rng.normal(0, 1.0, n).cumsum() + 350.0
    prices = np.abs(prices)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.01, 0.5, n),
            "low": prices - rng.uniform(0.01, 0.5, n),
            "close": prices + rng.normal(0, 0.1, n),
            "volume": rng.integers(5000, 20000, n).astype(float),
            "session_type": pd.Categorical(
                ["half" if i == 10 else "full" for i in range(n)],
                categories=["full", "half"],
            ),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


@pytest.fixture
def spy_half_day_session() -> pd.DataFrame:
    """7-bar half-day H1 DataFrame (09:30–13:00 on 2020-11-27 Black Friday)."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2020-11-27 09:30", periods=7, freq="1h", tz="America/New_York")
    prices = 360.0 + rng.normal(0, 0.5, 7).cumsum()
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices + 0.05,
            "volume": [8000.0] * 7,
            "session_type": pd.Categorical(["half"] * 7, categories=["full", "half"]),
        },
        index=idx,
    )
    return df


@pytest.fixture
def spy_dst_transition() -> pd.DataFrame:
    """
    2 bars spanning DST transition (2020-03-08: clocks spring forward).
    Both bars in America/New_York — local time semantics preserved.
    """
    # Before DST: 2020-03-07 10:30 ET (UTC-5)
    # After DST:  2020-03-08 10:30 ET (UTC-4)
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-03-07 10:30", tz="America/New_York"),
            pd.Timestamp("2020-03-08 10:30", tz="America/New_York"),
        ]
    )
    df = pd.DataFrame(
        {
            "open": [350.0, 351.0],
            "high": [351.0, 352.0],
            "low": [349.0, 350.0],
            "close": [350.5, 351.5],
            "volume": [10000.0, 11000.0],
            "session_type": pd.Categorical(["full", "full"], categories=["full", "half"]),
        },
        index=idx,
    )
    return df


@pytest.fixture
def spy_zero_volume_bar() -> pd.DataFrame:
    """10-bar H1 DataFrame with one injected zero-volume bar at index 5."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-06 09:30", periods=10, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 0.5, 10)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices + 0.1,
            "volume": [5000.0] * 10,
            "session_type": pd.Categorical(["full"] * 10, categories=["full", "half"]),
        },
        index=idx,
    )
    df.iloc[5, df.columns.get_loc("volume")] = 0.0
    return df


@pytest.fixture
def spy_ohlc_integrity_violation() -> pd.DataFrame:
    """10-bar H1 DataFrame with one bar where high < max(open, close)."""
    rng = np.random.default_rng(13)
    idx = pd.date_range("2020-01-06 09:30", periods=10, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 0.5, 10)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 0.2,
            "volume": [5000.0] * 10,
            "session_type": pd.Categorical(["full"] * 10, categories=["full", "half"]),
        },
        index=idx,
    )
    # Inject violation at row 3: high < max(open, close)
    df.iloc[3, df.columns.get_loc("high")] = prices[3] - 0.5
    return df


@pytest.fixture
def gold_labels_fixture(tmp_path) -> str:
    """
    Minimal gold_labels.csv with 10 rows and approximate kappa = 0.75.
    Returns path to the CSV file.
    """
    import csv

    path = tmp_path / "gold_labels.csv"
    # Agreement on 8/10 rows → high kappa
    prog = [0, 1, -1, 0, 1, -1, 0, 0, 1, -1]
    human = [0, 1, -1, 0, 1, -1, 0, 0, 1, -1]  # perfect agreement kappa=1.0 for testing

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candle_index", "datetime", "programmatic_label", "human_label", "annotator_note"],
        )
        writer.writeheader()
        for i, (p, h) in enumerate(zip(prog, human)):
            writer.writerow(
                {
                    "candle_index": i,
                    "datetime": "2020-01-02 09:30:00-05:00",
                    "programmatic_label": p,
                    "human_label": h,
                    "annotator_note": "",
                }
            )
    return str(path)
