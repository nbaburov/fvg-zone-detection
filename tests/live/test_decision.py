"""Unit tests for SingleModelDecision."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inspect.base import ModelAdapter
from src.live.decision import SingleModelDecision, TradeAction
from src.live.window_builder import H1Bar, WindowEvent


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class MockAdapter(ModelAdapter):
    """Returns a fixed proba array for all inputs."""

    name = "_mock_decision_test"

    def __init__(self, proba: list[float]) -> None:
        self._proba = np.array(proba, dtype=np.float32)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        return np.tile(self._proba, (n, 1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_window_event(
    window: np.ndarray | None,
    raw_window: np.ndarray | None,
    skip_reason: str | None = None,
    ts: str = "2024-01-02 10:30:00",
) -> WindowEvent:
    h1_ts = pd.Timestamp(ts, tz="America/New_York")
    h1_bar = H1Bar(
        timestamp=h1_ts,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=5000.0,
    )
    return WindowEvent(
        h1_timestamp=h1_ts,
        h1_bar=h1_bar,
        window=window,
        raw_window=raw_window,
        skip_reason=skip_reason,
    )


def make_raw_window(n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = 100.0 + rng.standard_normal((n, 4)).cumsum(axis=0)
    vol = rng.uniform(1000, 5000, (n, 1))
    raw = np.hstack([base, vol])
    return raw.astype(np.float64)


# ---------------------------------------------------------------------------
# Tests: threshold
# ---------------------------------------------------------------------------


class TestDecisionThreshold:
    def test_below_threshold_returns_none_signal(self):
        adapter = MockAdapter([0.8, 0.1, 0.1])  # max bull/bear = 0.1
        decision = SingleModelDecision(adapter, threshold=0.5)

        raw = make_raw_window()
        window = raw.astype(np.float32)
        event = make_window_event(window, raw)
        action = decision.decide(event)

        assert action.signal == "none"
        assert action.skip_reason == "BELOW_THRESHOLD"

    def test_above_threshold_bull(self):
        adapter = MockAdapter([0.1, 0.8, 0.1])  # bull wins
        decision = SingleModelDecision(adapter, threshold=0.5)

        raw = make_raw_window()
        window = raw.astype(np.float32)
        event = make_window_event(window, raw)
        action = decision.decide(event)

        assert action.signal == "bull"
        assert action.confidence == pytest.approx(0.8, abs=1e-5)

    def test_above_threshold_bear(self):
        adapter = MockAdapter([0.1, 0.1, 0.8])  # bear wins
        decision = SingleModelDecision(adapter, threshold=0.5)

        raw = make_raw_window()
        window = raw.astype(np.float32)
        event = make_window_event(window, raw)
        action = decision.decide(event)

        assert action.signal == "bear"

    def test_none_window_returns_none_signal(self):
        adapter = MockAdapter([0.1, 0.8, 0.1])
        decision = SingleModelDecision(adapter, threshold=0.5)

        event = make_window_event(None, None, skip_reason="WARMUP")
        action = decision.decide(event)

        assert action.signal == "none"
        assert action.skip_reason == "WARMUP"

    def test_exactly_at_threshold_is_below(self):
        """Exactly at threshold → below (strict >)."""
        adapter = MockAdapter([0.5, 0.5, 0.0])
        decision = SingleModelDecision(adapter, threshold=0.5)

        raw = make_raw_window()
        event = make_window_event(raw.astype(np.float32), raw)
        action = decision.decide(event)

        assert action.signal == "none"


# ---------------------------------------------------------------------------
# Tests: gap extraction
# ---------------------------------------------------------------------------


class TestGapExtraction:
    def _make_raw_known_gap(self, signal: str) -> np.ndarray:
        """Build synthetic 60-bar window with known bar[-3], [-2], [-1] values."""
        raw = np.zeros((60, 5), dtype=np.float64)
        raw[:, :4] = 100.0
        raw[:, 4] = 1000.0

        # Set specific known values for last 3 bars
        # bar[-3] (index 57): high=110, low=90
        raw[57, 0] = 100.0  # open
        raw[57, 1] = 110.0  # high  ← impulse high
        raw[57, 2] = 90.0   # low   ← impulse low
        raw[57, 3] = 100.0  # close

        # bar[-2] (index 58): gap candle
        raw[58, 0] = 100.0
        raw[58, 1] = 105.0
        raw[58, 2] = 95.0
        raw[58, 3] = 100.0

        # bar[-1] (index 59): label bar — high=108, low=92
        raw[59, 0] = 100.0
        raw[59, 1] = 108.0  # high  ← bear gap_high source
        raw[59, 2] = 92.0   # low   ← bull gap_low source
        raw[59, 3] = 100.0

        return raw

    def test_bull_gap_extraction(self):
        adapter = MockAdapter([0.0, 0.9, 0.1])  # bull signal
        decision = SingleModelDecision(adapter, threshold=0.5)

        raw = self._make_raw_known_gap("bull")
        event = make_window_event(raw.astype(np.float32), raw)
        action = decision.decide(event)

        assert action.signal == "bull"
        assert action.gap_low == pytest.approx(92.0)   # raw[-1, 2]
        assert action.gap_high == pytest.approx(110.0)  # raw[-3, 1]

    def test_bear_gap_extraction(self):
        adapter = MockAdapter([0.0, 0.1, 0.9])  # bear signal
        decision = SingleModelDecision(adapter, threshold=0.5)

        raw = self._make_raw_known_gap("bear")
        event = make_window_event(raw.astype(np.float32), raw)
        action = decision.decide(event)

        assert action.signal == "bear"
        assert action.gap_low == pytest.approx(90.0)   # raw[-3, 2]
        assert action.gap_high == pytest.approx(108.0)  # raw[-1, 1]
