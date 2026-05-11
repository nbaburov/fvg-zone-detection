"""decision.py — Single-model confidence threshold → TradeAction.

Maps a WindowEvent + ModelAdapter to a TradeAction. One model per process.
No voting, no consensus. Gap boundaries extracted from raw (unnormalised) window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.inspect.base import ModelAdapter
from src.live.window_builder import WindowEvent

logger = logging.getLogger(__name__)


@dataclass
class TradeAction:
    """Output of SingleModelDecision.decide()."""

    signal: Literal["bull", "bear", "none"]
    confidence: float          # max proba among bull/bear
    proba: np.ndarray          # (3,) [none, bull, bear]
    h1_timestamp: pd.Timestamp
    gap_low: float             # FVG gap lower boundary (raw price)
    gap_high: float            # FVG gap upper boundary (raw price)
    skip_reason: Optional[str]


class SingleModelDecision:
    """Wraps a ModelAdapter and applies a confidence threshold.

    Parameters
    ----------
    adapter : ModelAdapter
        Any registered ModelAdapter. predict_proba((1, 60, 5)) → (1, 3).
    threshold : float
        Minimum confidence to emit a non-"none" signal. Default 0.5.
    """

    def __init__(self, adapter: ModelAdapter, threshold: float = 0.5) -> None:
        self._adapter = adapter
        self._threshold = threshold

    def decide(self, event: WindowEvent) -> TradeAction:
        """Convert a WindowEvent to a TradeAction.

        Steps
        -----
        1. If event.window is None → signal="none" with event.skip_reason.
        2. Run adapter.predict_proba(window[np.newaxis]) → squeeze to (3,).
        3. Apply threshold.
        4. Extract gap boundaries from raw_window.
        """
        if event.window is None:
            return TradeAction(
                signal="none",
                confidence=0.0,
                proba=np.zeros(3, dtype=np.float32),
                h1_timestamp=event.h1_timestamp,
                gap_low=0.0,
                gap_high=0.0,
                skip_reason=event.skip_reason,
            )

        windows_batch = event.window[np.newaxis].astype(np.float32)  # (1, 60, 5)
        proba_batch = self._adapter.predict_proba(windows_batch)     # (1, 3)
        proba = proba_batch[0]                                        # (3,)

        bull_p = float(proba[1])
        bear_p = float(proba[2])
        max_p = max(bull_p, bear_p)

        if max_p <= self._threshold:
            return TradeAction(
                signal="none",
                confidence=max_p,
                proba=proba,
                h1_timestamp=event.h1_timestamp,
                gap_low=0.0,
                gap_high=0.0,
                skip_reason="BELOW_THRESHOLD",
            )

        signal: Literal["bull", "bear"] = "bull" if bull_p >= bear_p else "bear"

        # Extract FVG gap boundaries from raw (unnormalised) window.
        # 3-candle FVG: bars[-3]=impulse N-2, bars[-2]=gap candle N-1, bars[-1]=label bar N.
        # Bull FVG: gap between bar[-3].high and bar[-1].low.
        # Bear FVG: gap between bar[-3].low and bar[-1].high.
        raw = event.raw_window  # (60, 5) float64 — columns: open, high, low, close, volume
        if signal == "bull":
            gap_low = float(raw[-1, 2])   # bar N low
            gap_high = float(raw[-3, 1])  # bar N-2 high
        else:
            gap_low = float(raw[-3, 2])   # bar N-2 low
            gap_high = float(raw[-1, 1])  # bar N high

        logger.info(
            "Decision: %s @ %s conf=%.3f gap=[%.2f, %.2f]",
            signal, event.h1_timestamp, max_p, gap_low, gap_high,
        )

        return TradeAction(
            signal=signal,
            confidence=max_p,
            proba=proba,
            h1_timestamp=event.h1_timestamp,
            gap_low=gap_low,
            gap_high=gap_high,
            skip_reason=None,
        )
