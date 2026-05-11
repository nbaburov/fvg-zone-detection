"""FVG (Fair Value Gap) labeller — vectorised, no row-loops."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.labels import register
from src.data.labels.base import BaseLabeller


@register("fvg")
@register("fvg_raw")
class FVGLabeller(BaseLabeller):
    """
    Bullish FVG at candle i: high[i-1] < low[i+1] AND close[i] > open[i].
        Label value 1 placed at i+1 (N+1 convention).

    Bearish FVG at candle i: low[i-1] > high[i+1] AND close[i] < open[i].
        Label value -1 placed at i+1 (N+1 convention).

    Boundary: i ranges from 1 to len(df)-2. First and last row always 0.
    Pure vectorised operation — no Python-level for-loops.
    """

    label_index_offset: int = 1
    num_classes: int = 3
    class_names: list[str] = ["none", "bullish", "bearish"]
    encoded_map: dict[int, int] = {0: 0, 1: 1, -1: 2}

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Input:  DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
        Output: Series[int] same index as df. Values: {-1, 0, 1}. No NaN.
        """
        n = len(df)
        raw = np.zeros(n, dtype=np.int8)

        if n < 3:
            return pd.Series(raw, index=df.index, dtype=int)

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        open_ = df["open"].to_numpy()
        close = df["close"].to_numpy()

        # i ranges 1..n-2 (centre candle of 3-candle pattern)
        i = np.arange(1, n - 1)

        # Bullish FVG at i: gap between i-1 high and i+1 low, bullish body
        bull_gap = high[i - 1] < low[i + 1]
        bull_body = close[i] > open_[i]
        bull_mask = bull_gap & bull_body

        # Bearish FVG at i: gap between i-1 low and i+1 high, bearish body
        bear_gap = low[i - 1] > high[i + 1]
        bear_body = close[i] < open_[i]
        bear_mask = bear_gap & bear_body

        # Label placed at i+1 (N+1 convention)
        label_positions = i + 1  # shape (n-2,)

        # Bear before bull so bull wins on overlap (bullish takes priority)
        raw[label_positions[bear_mask]] = -1
        raw[label_positions[bull_mask]] = 1

        # Enforce boundary: first and last positions must be 0
        raw[0] = 0
        raw[n - 1] = 0

        return pd.Series(raw.astype(int), index=df.index)
