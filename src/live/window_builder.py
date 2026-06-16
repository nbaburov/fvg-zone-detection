"""window_builder.py — RTH accumulator that emits sliding windows.

Accumulates 1-min bars, resamples to RTH-anchored bars at the requested
timeframe (H1 default: 09:30, 10:30 ... 15:30 ET; M15: 09:30, 09:45, …;
M5: 09:30, 09:35, …), and emits WindowEvent objects on each bar close.
Window content is guaranteed to be mathematically identical to the training
resample (pd.resample closed='left', label='left', offset='30min').
normalise_window() is imported directly — never duplicated here.

H1 default path behaves byte-identically to the original implementation.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.normalize import normalise_window
from src.data.timeframe import H1, Timeframe, blackout_bars
from src.data.window import _MAX_INTRA_WINDOW_GAP_MINUTES
from src.live.stream import MinuteBar

logger = logging.getLogger(__name__)

# RTH bounds (inclusive start, exclusive end — same as training RTH filter)
_RTH_START = pd.Timestamp("09:30:00").time()
_RTH_END = pd.Timestamp("16:00:00").time()

# H1 boundary minutes anchored at :30 — mirrors offset='30min' in training resample.
# Kept for backward compat (execution.py / tests import this name).
_H1_BOUNDARY_MINUTES = [30]  # minute-of-hour that marks H1 boundary start


@dataclass
class H1Bar:
    """A single RTH-anchored bar (name kept as H1Bar for backward compat)."""

    timestamp: pd.Timestamp  # boundary start (label='left')
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class WindowEvent:
    """Emitted on each bar close."""

    h1_timestamp: pd.Timestamp
    h1_bar: H1Bar
    window: Optional[np.ndarray]      # (window_size, 5) float32 normalised, or None
    raw_window: Optional[np.ndarray]  # (window_size, 5) float64 raw OHLCV, or None
    skip_reason: Optional[str]        # "WARMUP", "CROSS_SESSION_GAP", or None
    session_bar_index: int = 0        # 0-based index of this bar within the RTH session


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


def _bar_boundary_for(ts: pd.Timestamp, tf: Timeframe) -> pd.Timestamp:
    """Return the bar boundary start for *ts* given *tf*.

    Uses tf.boundary_minutes — the set of minute-of-hour values at which a new
    bar begins.  The returned timestamp is the most-recent boundary that is
    <= ts.

    For H1 (boundary_minutes=(30,)) this is identical to _h1_boundary_for.
    For M15 (boundary_minutes=(0,15,30,45)) a bar at 10:23 returns 10:15.
    For M5  (boundary_minutes=(0,5,10,...,55)) a bar at 10:23 returns 10:20.
    """
    if tf.minutes == 60:
        # Fast path — byte-identical to the original _h1_boundary_for.
        return _h1_boundary_for(ts)

    bm = tf.boundary_minutes  # sorted tuple
    m = ts.minute
    # Find the largest boundary_minute <= m
    boundary_minute = bm[0]
    for b in bm:
        if b <= m:
            boundary_minute = b
        else:
            break
    return ts.replace(minute=boundary_minute, second=0, microsecond=0)


def _session_bar_index(boundary: pd.Timestamp, tf: Timeframe) -> int:
    """Return the 0-based position of *boundary* within the RTH session.

    RTH starts at 09:30.  Minutes since RTH open / tf.minutes gives the index.
    E.g. H1: 09:30→0, 10:30→1, …, 15:30→6.
         M15: 09:30→0, 09:45→1, …, 15:30→24.
         M5:  09:30→0, 09:35→1, …, 15:55→77.
    """
    rth_open = boundary.replace(hour=9, minute=30, second=0, microsecond=0)
    delta_minutes = int((boundary - rth_open).total_seconds() / 60)
    return delta_minutes // tf.minutes


def _next_boundary(boundary: pd.Timestamp, tf: Timeframe) -> pd.Timestamp:
    """Return the next bar boundary after *boundary*."""
    return boundary + pd.Timedelta(minutes=tf.minutes)


def _has_session_gap(bars: list[H1Bar], max_gap_minutes: int) -> bool:
    """Return True if any consecutive pair of bars is > max_gap_minutes apart."""
    if len(bars) < 2:
        return False
    for a, b in zip(bars[:-1], bars[1:]):
        gap = (b.timestamp - a.timestamp).total_seconds() / 60
        if gap > max_gap_minutes:
            return True
    return False


# Backward-compat alias used by some tests
def _has_session_gap_h1(bars: list[H1Bar]) -> bool:
    return _has_session_gap(bars, _MAX_INTRA_WINDOW_GAP_MINUTES)


class LiveWindowBuilder:
    """Accumulates 1-min bars and emits WindowEvent on each bar close.

    Parameters
    ----------
    timeframe : Timeframe
        Bar timeframe.  Default ``H1`` — all behaviour is byte-identical to
        the original single-TF implementation when this default is used.
    window_size : int
        Sliding window depth in bars.  Default 60.
    drop_cross_session : bool
        Whether to skip windows spanning a >max_intra_window_gap session gap.
        Default False to match training / inspect (runner.py).

    Thread safety: not thread-safe. Designed for single-threaded async event loop.
    """

    WINDOW_SIZE = 60  # kept for backward compat; instance uses self._window_size

    def __init__(
        self,
        timeframe: Timeframe = H1,
        window_size: int = 60,
        drop_cross_session: bool = False,
    ) -> None:
        self._tf = timeframe
        self._window_size = window_size
        self._drop_cross_session = drop_cross_session
        # Buffer of accumulated bars (up to window_size)
        self._h1_buffer: deque[H1Bar] = deque(maxlen=self._window_size)
        # Buffer of 1-min bars for current (open) bar
        self._current_1m: list[MinuteBar] = []
        # Timestamp of the current open bar boundary (None if no bar received yet)
        self._current_boundary: Optional[pd.Timestamp] = None
        # Deduplicate gap_fill inserts
        self._seen_1m_timestamps: set[pd.Timestamp] = set()

    @property
    def h1_count(self) -> int:
        """Number of complete bars in buffer."""
        return len(self._h1_buffer)

    def on_bar(self, bar: MinuteBar) -> Optional[WindowEvent]:
        """Feed a 1-min bar. Returns WindowEvent on bar close, else None.

        Parameters
        ----------
        bar : MinuteBar
            A 1-min bar from Alpaca (new or updated).
        """
        # Reject non-RTH bars
        if not _is_rth(bar.timestamp):
            return None

        ts = bar.timestamp
        boundary = _bar_boundary_for(ts, self._tf)

        if self._current_boundary is None:
            self._current_boundary = boundary

        if boundary == self._current_boundary:
            # Belongs to current open bar
            self._upsert_1m(bar)
            return None

        if boundary > self._current_boundary:
            # New boundary — close the current bar and start a new one
            event = self._close_current_bar()
            self._current_boundary = boundary
            self._current_1m = []
            self._upsert_1m(bar)
            return event

        # bar is from an earlier boundary (late arrival or gap_fill ordering issue)
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

    def _close_current_bar(self) -> Optional[WindowEvent]:
        """Assemble bar from buffered 1-min bars, append to buffer, return WindowEvent."""
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
            "Bar closed [%s]: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
            self._tf.token, h1.timestamp, h1.open, h1.high, h1.low, h1.close, h1.volume,
        )

        return self._build_window_event(h1)

    # Backward-compat alias used by tests
    def _close_current_h1(self) -> Optional[WindowEvent]:
        return self._close_current_bar()

    def _build_window_event(self, h1: H1Bar) -> WindowEvent:
        """Build WindowEvent from current buffer state."""
        idx = _session_bar_index(h1.timestamp, self._tf)

        if len(self._h1_buffer) < self._window_size:
            return WindowEvent(
                h1_timestamp=h1.timestamp,
                h1_bar=h1,
                window=None,
                raw_window=None,
                skip_reason="WARMUP",
                session_bar_index=idx,
            )

        bars_list = list(self._h1_buffer)

        if self._drop_cross_session and _has_session_gap(bars_list, self._tf.max_intra_window_gap_minutes):
            logger.warning(
                "Cross-session gap detected in window ending at %s. Skipping.", h1.timestamp
            )
            return WindowEvent(
                h1_timestamp=h1.timestamp,
                h1_bar=h1,
                window=None,
                raw_window=None,
                skip_reason="CROSS_SESSION_GAP",
                session_bar_index=idx,
            )

        raw = np.array(
            [[b.open, b.high, b.low, b.close, b.volume] for b in bars_list],
            dtype=np.float64,
        )  # (window_size, 5)
        normalised = normalise_window(raw)

        return WindowEvent(
            h1_timestamp=h1.timestamp,
            h1_bar=h1,
            window=normalised,
            raw_window=raw,
            skip_reason=None,
            session_bar_index=idx,
        )
