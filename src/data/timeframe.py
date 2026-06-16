"""Timeframe value object for SMC pipeline.

A frozen dataclass encoding all per-timeframe constants needed by the data
pipeline, windowing layer, and live builder.  Pure value object — no IO,
no pandas calls beyond holding rule strings.

Module-level constants:
    H1   — 60-minute (current production default)
    M15  — 15-minute
    M5   — 5-minute

Helper functions:
    blackout_bars(tf)             — ceil(30 / tf.minutes)
    warmup_days(tf, window_size)  — ceil(window_size / tf.bars_per_rth_day) + 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Timeframe:
    """Immutable descriptor for a candlestick timeframe.

    All fields are set at construction; no mutable state.

    Attributes:
        pandas_rule: Resample rule string passed to ``DataFrame.resample``
            (e.g. ``"60min"``).  ``"60min"`` and ``"1h"`` are equivalent in
            pandas; ``"60min"`` is used here for explicitness.
        minutes: Timeframe duration in minutes (5, 15, or 60).
        rth_offset: Offset string for the resample anchor so bars align with
            09:30 ET open.  Always ``"30min"`` across all supported TFs.
        token: Short tag used in filenames and checkpoint metadata
            (``"5m"``, ``"15m"``, ``"h1"``).
        bars_per_rth_day: Number of complete bars in a full RTH session
            (390 min ÷ minutes).  Verified empirically from
            ``data/processed/spy_h1_train.parquet`` → 7 for H1
            (09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30).
            Note: the plan draft said 6; the parquet says 7 — parquet wins.
        max_intra_window_gap_minutes: Maximum gap between consecutive candles
            that still counts as the same session.  Equals
            ``round(1.5 * minutes)``.  H1 → 90, matching the literal
            ``_MAX_INTRA_WINDOW_GAP_MINUTES = 90`` in ``src/data/window.py``.
        boundary_minutes: Tuple of minute-of-hour values (within 0–59) at
            which a new bar boundary occurs.  Used by the live builder to
            detect when a complete bar is ready.
            H1 → ``(30,)``; M15 → ``(0, 15, 30, 45)``; M5 → ``(0, 5, …, 55)``.
    """

    pandas_rule: str
    minutes: int
    token: str
    bars_per_rth_day: int
    max_intra_window_gap_minutes: int
    boundary_minutes: tuple[int, ...]
    rth_offset: str = "30min"

    # Registry of all known instances, keyed by token. Populated at module
    # load via the ``_register`` class-level helper below.
    _registry: ClassVar[dict[str, "Timeframe"]] = {}
    _by_minutes: ClassVar[dict[int, "Timeframe"]] = {}

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_token(cls, token: str) -> "Timeframe":
        """Return the Timeframe instance for *token*.

        Raises:
            ValueError: If *token* is not one of the registered tokens.
                The error message lists all valid tokens.
        """
        try:
            return cls._registry[token]
        except KeyError:
            valid = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown timeframe token {token!r}. Valid tokens: {valid}"
            ) from None

    @classmethod
    def from_minutes(cls, minutes: int) -> "Timeframe":
        """Return the Timeframe instance for *minutes*.

        Raises:
            ValueError: If *minutes* is not one of the registered values.
        """
        try:
            return cls._by_minutes[minutes]
        except KeyError:
            valid = ", ".join(str(m) for m in sorted(cls._by_minutes))
            raise ValueError(
                f"Unknown timeframe minutes {minutes!r}. Valid values: {valid}"
            ) from None


def _make(
    pandas_rule: str,
    minutes: int,
    token: str,
    bars_per_rth_day: int,
) -> Timeframe:
    """Construct a Timeframe and register it."""
    gap = round(1.5 * minutes)  # H1→90, M15→22 (round-half-even), M5→8
    if minutes == 60:
        boundary = (30,)
    else:
        boundary = tuple(range(0, 60, minutes))
    tf = Timeframe(
        pandas_rule=pandas_rule,
        minutes=minutes,
        token=token,
        bars_per_rth_day=bars_per_rth_day,
        max_intra_window_gap_minutes=gap,
        boundary_minutes=boundary,
    )
    Timeframe._registry[token] = tf
    Timeframe._by_minutes[minutes] = tf
    return tf


# ---------------------------------------------------------------------------
# Module-level constants — canonical instances
# ---------------------------------------------------------------------------

H1: Timeframe = _make(
    pandas_rule="60min",
    minutes=60,
    token="h1",
    bars_per_rth_day=7,  # verified: spy_h1_train.parquet → 7 bars/day (09:30–15:30)
)
"""60-minute timeframe — current production default."""

M15: Timeframe = _make(
    pandas_rule="15min",
    minutes=15,
    token="15m",
    bars_per_rth_day=26,  # 390 / 15 = 26
)
"""15-minute timeframe."""

M5: Timeframe = _make(
    pandas_rule="5min",
    minutes=5,
    token="5m",
    bars_per_rth_day=78,  # 390 / 5 = 78
)
"""5-minute timeframe."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def blackout_bars(tf: Timeframe) -> int:
    """Return the number of bars to black out at the open/close of a session.

    Defined as ``ceil(30 / tf.minutes)``.  The 30-minute window is the
    first/last half-hour of RTH, where FVG signals are unreliable due to
    gap fills and auction noise (plan §3.6).

    Examples:
        H1  → ceil(30/60) = 1
        M15 → ceil(30/15) = 2
        M5  → ceil(30/5)  = 6
    """
    return math.ceil(30 / tf.minutes)


def warmup_days(tf: Timeframe, window_size: int) -> int:
    """Return how many trading sessions of 1-min history to backfill before going live.

    Defined as ``ceil(window_size / tf.bars_per_rth_day) + 1``.
    The ``+1`` buffers against holidays and half-day sessions (plan §3.9).

    Args:
        tf: The Timeframe for which warm-up depth is needed.
        window_size: Number of bars in the sliding window (e.g. 60).

    Examples:
        H1,  ws=60 → ceil(60/7)  + 1 = 10
        M15, ws=60 → ceil(60/26) + 1 = 4
        M5,  ws=60 → ceil(60/78) + 1 = 2
    """
    return math.ceil(window_size / tf.bars_per_rth_day) + 1
