"""Tests for TF-parametric resample and class-weights reader (WS-2).

Covers:
- _resample_minute(df, H1)  == old _resample_minute_to_h1(df)  (byte-identity on a fixture)
- _resample_minute(df, H1)  → 7 bars per full RTH day
- _resample_minute(df, M5)  → 78 bars per full RTH day
- _resample_minute(df, M15) → 26 bars per full RTH day
- 09:30 anchor preserved for all TFs
- closed='left', label='left' semantics (first bar label = 09:30 not 09:31)
- load_class_weights: token="h1" → unsuffixed path; token="5m" → suffixed path
- download_bars and download_h1 shim both importable (no API call)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.timeframe import H1, M5, M15


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rth_minutes(date: str, tz: str = "America/New_York") -> pd.DatetimeIndex:
    """Return a 1-minute DatetimeIndex covering a full RTH session (09:30–15:59, 390 bars)."""
    start = pd.Timestamp(f"{date} 09:30", tz=tz)
    end = pd.Timestamp(f"{date} 15:59", tz=tz)
    return pd.date_range(start, end, freq="1min")


def _make_minute_df(dates: list[str]) -> pd.DataFrame:
    """Synthetic 1-minute OHLCV DataFrame spanning *dates* (full RTH each day)."""
    frames = []
    for d in dates:
        idx = _make_rth_minutes(d)
        n = len(idx)
        rng = np.random.default_rng(42)
        close = 400.0 + rng.standard_normal(n).cumsum() * 0.1
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + rng.uniform(0, 0.05, n)
        low = np.minimum(open_, close) - rng.uniform(0, 0.05, n)
        vol = rng.integers(1000, 5000, n).astype(float)
        frames.append(pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        ))
    return pd.concat(frames).sort_index()


@pytest.fixture()
def one_day_minutes() -> pd.DataFrame:
    return _make_minute_df(["2020-01-02"])


@pytest.fixture()
def two_day_minutes() -> pd.DataFrame:
    return _make_minute_df(["2020-01-02", "2020-01-03"])


# ---------------------------------------------------------------------------
# Bar-count assertions
# ---------------------------------------------------------------------------

class TestBarCounts:
    def test_h1_full_rth_day_gives_7_bars(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, H1)
        assert len(result) == 7, f"Expected 7 H1 bars, got {len(result)}"

    def test_m5_full_rth_day_gives_78_bars(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, M5)
        assert len(result) == 78, f"Expected 78 M5 bars, got {len(result)}"

    def test_m15_full_rth_day_gives_26_bars(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, M15)
        assert len(result) == 26, f"Expected 26 M15 bars, got {len(result)}"

    def test_h1_two_days_gives_14_bars(self, two_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(two_day_minutes, H1)
        assert len(result) == 14

    def test_m5_two_days_gives_156_bars(self, two_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(two_day_minutes, M5)
        assert len(result) == 156

    def test_m15_two_days_gives_52_bars(self, two_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(two_day_minutes, M15)
        assert len(result) == 52


# ---------------------------------------------------------------------------
# 09:30 anchor
# ---------------------------------------------------------------------------

class TestAnchor:
    def test_h1_first_bar_at_0930(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, H1)
        first = result.index[0]
        assert (first.hour, first.minute) == (9, 30), f"First H1 bar at {first.time()}, expected 09:30"

    def test_m5_first_bar_at_0930(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, M5)
        first = result.index[0]
        assert (first.hour, first.minute) == (9, 30)

    def test_m15_first_bar_at_0930(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, M15)
        first = result.index[0]
        assert (first.hour, first.minute) == (9, 30)


# ---------------------------------------------------------------------------
# closed/label semantics — label = bar open time, first minute IS bar's open
# ---------------------------------------------------------------------------

class TestClosedLabel:
    def test_h1_label_is_bar_open_not_close(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, H1)
        # The first bar open price must match the first minute's open price
        first_minute_open = one_day_minutes.iloc[0]["open"]
        assert result.iloc[0]["open"] == pytest.approx(first_minute_open)

    def test_m5_label_is_bar_open(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, M5)
        first_minute_open = one_day_minutes.iloc[0]["open"]
        assert result.iloc[0]["open"] == pytest.approx(first_minute_open)


# ---------------------------------------------------------------------------
# Backward-compat: _resample_minute(df, H1) == _resample_minute_to_h1(df)
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_resample_minute_h1_equals_old_alias(self, two_day_minutes):
        from src.data.download import _resample_minute, _resample_minute_to_h1
        new = _resample_minute(two_day_minutes.copy(), H1)
        old = _resample_minute_to_h1(two_day_minutes.copy())
        pd.testing.assert_frame_equal(new, old)

    def test_old_alias_importable(self):
        from src.data.download import _resample_minute_to_h1
        assert callable(_resample_minute_to_h1)

    def test_download_bars_importable(self):
        from src.data.download import download_bars
        assert callable(download_bars)

    def test_download_h1_importable(self):
        from src.data.download import download_h1
        assert callable(download_h1)


# ---------------------------------------------------------------------------
# Columns preserved
# ---------------------------------------------------------------------------

class TestColumns:
    def test_h1_has_ohlcv_columns(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, H1)
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}

    def test_m5_has_ohlcv_columns(self, one_day_minutes):
        from src.data.download import _resample_minute
        result = _resample_minute(one_day_minutes, M5)
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# load_class_weights — token routing
# ---------------------------------------------------------------------------

class TestLoadClassWeights:
    def _write_json(self, path: Path, data) -> None:
        with path.open("w") as fh:
            json.dump(data, fh)

    def test_h1_spy_reads_class_weights_spy_h1(self, tmp_path):
        from src.data.download import load_class_weights
        weights = {"0": 1.0, "1": 10.0, "2": 8.0}
        self._write_json(tmp_path / "class_weights_spy_h1.json", weights)
        result = load_class_weights(tmp_path, token="h1")
        assert result == pytest.approx([1.0, 10.0, 8.0])

    def test_5m_spy_reads_class_weights_spy_5m(self, tmp_path):
        from src.data.download import load_class_weights
        weights = {"0": 1.0, "1": 12.0, "2": 9.0}
        self._write_json(tmp_path / "class_weights_spy_5m.json", weights)
        result = load_class_weights(tmp_path, token="5m")
        assert result == pytest.approx([1.0, 12.0, 9.0])

    def test_15m_spy_reads_class_weights_spy_15m(self, tmp_path):
        from src.data.download import load_class_weights
        weights = [1.0, 11.0, 9.5]
        self._write_json(tmp_path / "class_weights_spy_15m.json", weights)
        result = load_class_weights(tmp_path, token="15m")
        assert result == pytest.approx([1.0, 11.0, 9.5])

    def test_default_scope_is_spy_h1(self, tmp_path):
        from src.data.download import load_class_weights
        weights = {"0": 2.0, "1": 5.0, "2": 4.0}
        self._write_json(tmp_path / "class_weights_spy_h1.json", weights)
        # call with no args — defaults to scope="spy", token="h1"
        result = load_class_weights(tmp_path)
        assert result == pytest.approx([2.0, 5.0, 4.0])

    def test_multisym_scope_reads_class_weights_multisym_h1(self, tmp_path):
        from src.data.download import load_class_weights
        weights = {"0": 1.0, "1": 10.0, "2": 8.0}
        self._write_json(tmp_path / "class_weights_multisym_h1.json", weights)
        result = load_class_weights(tmp_path, token="h1", scope="multisym")
        assert result == pytest.approx([1.0, 10.0, 8.0])

    def test_missing_raises_file_not_found(self, tmp_path):
        from src.data.download import load_class_weights
        # Only a bare class_weights.json — new scheme does not read it → FileNotFoundError
        (tmp_path / "class_weights.json").write_text("{}")
        with pytest.raises(FileNotFoundError):
            load_class_weights(tmp_path, token="h1")

    def test_accepts_str_data_dir(self, tmp_path):
        from src.data.download import load_class_weights
        weights = {"0": 1.0, "1": 10.0, "2": 8.0}
        self._write_json(tmp_path / "class_weights_spy_h1.json", weights)
        result = load_class_weights(str(tmp_path))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _sanity_check_bar_count TF-aware scaling
# ---------------------------------------------------------------------------

class TestSanityCheckBarCount:
    def _make_df(self, n: int) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="America/New_York")
        return pd.DataFrame({"open": 1.0}, index=idx)

    def test_h1_raises_when_below_5000(self):
        from src.data.download import _sanity_check_bar_count
        df = self._make_df(100)
        with pytest.raises(ValueError, match="Sanity check"):
            _sanity_check_bar_count(df, "2018-01-01", "2020-01-01", H1)

    def test_h1_passes_when_above_5000(self):
        from src.data.download import _sanity_check_bar_count
        df = self._make_df(6000)
        _sanity_check_bar_count(df, "2018-01-01", "2020-01-01", H1)  # no raise

    def test_m5_threshold_scaled_up(self):
        from src.data.download import _sanity_check_bar_count
        # M5 bars_per_rth_day=78, H1=7 → scale=78/7≈11.1 → min_bars≈55714
        # A count of 5000 would pass H1 but should fail M5
        df = self._make_df(5000)
        with pytest.raises(ValueError, match="Sanity check"):
            _sanity_check_bar_count(df, "2018-01-01", "2020-01-01", M5)

    def test_short_date_range_exempt(self):
        from src.data.download import _sanity_check_bar_count
        df = self._make_df(10)
        # < 365 days → no raise regardless of count
        _sanity_check_bar_count(df, "2020-01-01", "2020-06-01", H1)

    def test_end_none_exempt(self):
        from src.data.download import _sanity_check_bar_count
        df = self._make_df(10)
        _sanity_check_bar_count(df, "2018-01-01", None, H1)  # no raise
