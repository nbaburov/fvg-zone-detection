"""Tests for src/data/timeframe.py — Timeframe value object, helpers, constants."""

import math
import pytest

from src.data.timeframe import (
    Timeframe,
    H1,
    M5,
    M15,
    blackout_bars,
    warmup_days,
)


# ---------------------------------------------------------------------------
# H1 field values — must match today's hard-coded literals exactly
# ---------------------------------------------------------------------------

class TestH1Fields:
    def test_pandas_rule(self):
        assert H1.pandas_rule == "60min"

    def test_minutes(self):
        assert H1.minutes == 60

    def test_rth_offset(self):
        assert H1.rth_offset == "30min"

    def test_token(self):
        assert H1.token == "h1"

    def test_bars_per_rth_day(self):
        # Verified from spy_h1_train.parquet: 7 bars per day (09:30..15:30 inclusive)
        assert H1.bars_per_rth_day == 7

    def test_max_intra_window_gap_minutes(self):
        # Must equal the literal _MAX_INTRA_WINDOW_GAP_MINUTES = 90 in src/data/window.py
        assert H1.max_intra_window_gap_minutes == 90

    def test_boundary_minutes(self):
        # H1 bars start at :30 (offset="30min") — only boundary within the hour
        assert H1.boundary_minutes == (30,)

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            H1.minutes = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# M15 field values
# ---------------------------------------------------------------------------

class TestM15Fields:
    def test_pandas_rule(self):
        assert M15.pandas_rule == "15min"

    def test_minutes(self):
        assert M15.minutes == 15

    def test_rth_offset(self):
        assert M15.rth_offset == "30min"

    def test_token(self):
        assert M15.token == "15m"

    def test_bars_per_rth_day(self):
        # RTH = 390 min / 15 = 26
        assert M15.bars_per_rth_day == 26

    def test_max_intra_window_gap_minutes(self):
        # round(1.5 * 15) = round(22.5) = 22
        assert M15.max_intra_window_gap_minutes == 22

    def test_boundary_minutes(self):
        assert M15.boundary_minutes == (0, 15, 30, 45)


# ---------------------------------------------------------------------------
# M5 field values
# ---------------------------------------------------------------------------

class TestM5Fields:
    def test_pandas_rule(self):
        assert M5.pandas_rule == "5min"

    def test_minutes(self):
        assert M5.minutes == 5

    def test_rth_offset(self):
        assert M5.rth_offset == "30min"

    def test_token(self):
        assert M5.token == "5m"

    def test_bars_per_rth_day(self):
        # RTH = 390 min / 5 = 78
        assert M5.bars_per_rth_day == 78

    def test_max_intra_window_gap_minutes(self):
        # round(1.5 * 5) = round(7.5) = 8
        assert M5.max_intra_window_gap_minutes == 8

    def test_boundary_minutes(self):
        assert M5.boundary_minutes == tuple(range(0, 60, 5))


# ---------------------------------------------------------------------------
# from_token round-trips and error
# ---------------------------------------------------------------------------

class TestFromToken:
    def test_h1_roundtrip(self):
        assert Timeframe.from_token("h1") is H1

    def test_15m_roundtrip(self):
        assert Timeframe.from_token("15m") is M15

    def test_5m_roundtrip(self):
        assert Timeframe.from_token("5m") is M5

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="(?i)valid tokens"):
            Timeframe.from_token("1d")

    def test_unknown_lists_valid_tokens(self):
        with pytest.raises(ValueError) as exc_info:
            Timeframe.from_token("bogus")
        msg = str(exc_info.value)
        assert "h1" in msg
        assert "15m" in msg
        assert "5m" in msg


# ---------------------------------------------------------------------------
# from_minutes round-trips and error
# ---------------------------------------------------------------------------

class TestFromMinutes:
    def test_60_is_h1(self):
        assert Timeframe.from_minutes(60) is H1

    def test_15_is_m15(self):
        assert Timeframe.from_minutes(15) is M15

    def test_5_is_m5(self):
        assert Timeframe.from_minutes(5) is M5

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            Timeframe.from_minutes(30)


# ---------------------------------------------------------------------------
# blackout_bars — ceil(30 / tf.minutes)
# ---------------------------------------------------------------------------

class TestBlackoutBars:
    def test_h1(self):
        assert blackout_bars(H1) == 1   # ceil(30/60) = 1

    def test_m15(self):
        assert blackout_bars(M15) == 2  # ceil(30/15) = 2

    def test_m5(self):
        assert blackout_bars(M5) == 6   # ceil(30/5)  = 6


# ---------------------------------------------------------------------------
# warmup_days — ceil(window_size / bars_per_rth_day) + 1
# ---------------------------------------------------------------------------

class TestWarmupDays:
    def test_h1_ws60(self):
        # ceil(60/7) + 1 = 9 + 1 = 10
        assert warmup_days(H1, 60) == 10

    def test_m15_ws60(self):
        # ceil(60/26) + 1 = 3 + 1 = 4
        assert warmup_days(M15, 60) == 4

    def test_m5_ws60(self):
        # ceil(60/78) + 1 = 1 + 1 = 2
        assert warmup_days(M5, 60) == 2

    def test_h1_ws120(self):
        # ceil(120/7) + 1 = 18 + 1 = 19  (non-trivial, confirms formula not just cached)
        assert warmup_days(H1, 120) == math.ceil(120 / 7) + 1

    def test_formula_general(self):
        for tf in (H1, M15, M5):
            for ws in (10, 30, 60, 100):
                expected = math.ceil(ws / tf.bars_per_rth_day) + 1
                assert warmup_days(tf, ws) == expected


# ---------------------------------------------------------------------------
# Immutability / identity: same token → same object (module constants)
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_h1_is_singleton(self):
        assert Timeframe.from_token("h1") is Timeframe.from_token("h1")

    def test_all_three_distinct(self):
        assert H1 is not M15
        assert H1 is not M5
        assert M15 is not M5
