"""window_builder.py — RTH H1 accumulator that emits 60-bar windows.

Accumulates 1-min bars, resamples to RTH-anchored H1 (09:30, 10:30 ... 15:30 ET),
and emits WindowEvent objects on each H1 close. Window content is guaranteed to be
mathematically identical to the training resample (pd.resample closed='left',
label='left', offset='30min'). normalise_window() is imported directly — never
duplicated here.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.normalize import normalise_window
from src.data.window import _MAX_INTRA_WINDOW_GAP_MINUTES
from src.live.stream import MinuteBar

logger = logging.getLogger(__name__)

# RTH bounds (inclusive start, exclusive end — same as training RTH filter)
_RTH_START = pd.Timestamp("09:30:00").time()
_RTH_END = pd.Timestamp("16:00:00").time()

# H1 boundary minutes anchored at :30 — mirrors offset='30min' in training resample
_H1_BOUNDARY_MINUTES = [30]  # minute-of-hour that marks H1 boundary start


@dataclass
class H1Bar:
    """A single RTH-anchored H1 OHLCV bar."""

    timestamp: pd.Timestamp  # boundary start (label='left')
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class WindowEvent:
    """Emitted on each H1 close."""

    h1_timestamp: pd.Timestamp
    h1_bar: H1Bar
    window: Optional[np.ndarray]      # (60, 5) float32 normalised, or None
    raw_window: Optional[np.ndarray]  # (60, 5) float64 raw OHLCV, or None
    skip_reason: Optional[str]        # "WARMUP", "CROSS_SESSION_GAP", or None


def _is_rth(ts: pd.Timestamp) -> bool:
    """Return True if timestamp falls within RTH (09:30 <= ts.time() < 16:00 ET)."""
    t = ts.time()
    return _RTH_START <= t < _RTH_END


def _h1_boundary_for(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the H1 boundary start for a given 1-min bar timestamp.

    H1 boundaries are anchored at :30 of each hour: 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30.
    A 1-min bar at 10:47 belongs to boundary 10:30.
    A 1-min bar at 09:30 belongs to boundary 09:30.
    A 1-min bar at 09:29 is pre-RTH (rejected before reaching here).
    """
    minute = ts.minute
    if minute >= 30:
        boundary_hour = ts.hour
        boundary_minute = 30
    else:
        boundary_hour = ts.hour - 1
        boundary_minute = 30

    return ts.replace(hour=boundary_hour, minute=boundary_minute, second=0, microsecond=0)


def _next_h1_boundary(boundary: pd.Timestamp) -> pd.Timestamp:
    """Return the next H1 boundary after the given one (add 60 minutes)."""
    return boundary + pd.Timedelta(hours=1)


def _has_session_gap_h1(bars: list[H1Bar]) -> bool:
    """Return True if any consecutive pair of H1 bars is > _MAX_INTRA_WINDOW_GAP_MINUTES apart."""
    if len(bars) < 2:
        return False
    for a, b in zip(bars[:-1], bars[1:]):
        gap = (b.timestamp - a.timestamp).total_seconds() / 60
        if gap > _MAX_INTRA_WINDOW_GAP_MINUTES:
            return True
    return False


class LiveWindowBuilder:
    """Accumulates 1-min bars and emits WindowEvent on each H1 close.

    Thread safety: not thread-safe. Designed for single-threaded async event loop.
    """

    WINDOW_SIZE = 60

    def __init__(self) -> None:
        # Buffer of accumulated H1 bars (up to WINDOW_SIZE)
        self._h1_buffer: deque[H1Bar] = deque(maxlen=self.WINDOW_SIZE)
        # Buffer of 1-min bars for current (open) H1
        self._current_1m: list[MinuteBar] = []
        # Timestamp of the current open H1 boundary (None if no bar received yet)
        self._current_boundary: Optional[pd.Timestamp] = None
        # Deduplicate gap_fill inserts
        self._seen_1m_timestamps: set[pd.Timestamp] = set()

    @property
    def h1_count(self) -> int:
        """Number of complete H1 bars in buffer."""
        return len(self._h1_buffer)

    def on_bar(self, bar: MinuteBar) -> Optional[WindowEvent]:
        """Feed a 1-min bar. Returns WindowEvent on H1 close, else None.

        Parameters
        ----------
        bar : MinuteBar
            A 1-min bar from Alpaca (new or updated).
        """
        # Reject non-RTH bars
        if not _is_rth(bar.timestamp):
            return None

        ts = bar.timestamp
        boundary = _h1_boundary_for(ts)

        if self._current_boundary is None:
            self._current_boundary = boundary

        if boundary == self._current_boundary:
            # Belongs to current open H1
            self._upsert_1m(bar)
            return None

        if boundary > self._current_boundary:
            # New H1 boundary — close the current H1 and start a new one
            event = self._close_current_h1()
            self._current_boundary = boundary
            self._current_1m = []
            self._upsert_1m(bar)
            return event

        # bar is from a boundary earlier than current (late arrival or gap_fill ordering issue)
        # Insert into seen set but do not re-trigger H1 close
        self._upsert_1m(bar)
        return None

    def gap_fill(self, bars: list[MinuteBar]) -> None:
        """Insert historical 1-min bars (from REST backfill or reconnect).

        Bars are inserted in chronological order. Deduplication by timestamp prevents
        double-counting bars already in the buffer.
        """
        sorted_bars = sorted(bars, key=lambda b: b.timestamp)
        for bar in sorted_bars:
            if bar.timestamp in self._seen_1m_timestamps:
                continue
            self.on_bar(bar)

    def _upsert_1m(self, bar: MinuteBar) -> None:
        """Insert or replace a 1-min bar in current_1m list (handles updated_bar)."""
        self._seen_1m_timestamps.add(bar.timestamp)
        for i, existing in enumerate(self._current_1m):
            if existing.timestamp == bar.timestamp:
                self._current_1m[i] = bar
                return
        self._current_1m.append(bar)

    def _close_current_h1(self) -> Optional[WindowEvent]:
        """Assemble H1 bar from buffered 1-min bars, append to h1_buffer, return WindowEvent."""
        if not self._current_1m or self._current_boundary is None:
            return None

        bars = sorted(self._current_1m, key=lambda b: b.timestamp)
        h1 = H1Bar(
            timestamp=self._current_boundary,
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars),
        )
        self._h1_buffer.append(h1)
        logger.debug(
            "H1 closed: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
            h1.timestamp, h1.open, h1.high, h1.low, h1.close, h1.volume,
        )

        return self._build_window_event(h1)

    def _build_window_event(self, h1: H1Bar) -> WindowEvent:
        """Build WindowEvent from current h1_buffer state."""
        if len(self._h1_buffer) < self.WINDOW_SIZE:
            return WindowEvent(
                h1_timestamp=h1.timestamp,
                h1_bar=h1,
                window=None,
                raw_window=None,
                skip_reason="WARMUP",
            )

        bars_list = list(self._h1_buffer)

        if _has_session_gap_h1(bars_list):
            logger.warning(
                "Cross-session gap detected in window ending at %s. Skipping.", h1.timestamp
            )
            return WindowEvent(
                h1_timestamp=h1.timestamp,
                h1_bar=h1,
                window=None,
                raw_window=None,
                skip_reason="CROSS_SESSION_GAP",
            )

        raw = np.array(
            [[b.open, b.high, b.low, b.close, b.volume] for b in bars_list],
            dtype=np.float64,
        )  # (60, 5)
        normalised = normalise_window(raw)

        return WindowEvent(
            h1_timestamp=h1.timestamp,
            h1_bar=h1,
            window=normalised,
            raw_window=raw,
            skip_reason=None,
        )
