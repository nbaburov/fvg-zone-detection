"""Base class for all SMC labellers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class BaseLabeller(ABC):
    label_index_offset: ClassVar[int]       # bars after pattern close before label knowable; FVG = 1
    num_classes: ClassVar[int]              # output head size; FVG = 3
    class_names: ClassVar[list[str]]        # for viz tooltips; FVG = ["none", "bullish", "bearish"]
    encoded_map: ClassVar[dict[int, int]]   # raw detector int → PyTorch class index

    @abstractmethod
    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Input:  DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
        Output: Series[int] same index as df.
                Values are RAW detector codes (before PyTorch encoding).
                FVG: {-1=bearish, 0=none, 1=bullish}.
                First label_index_offset rows must be 0 (no lookback available).
                Last label_index_offset rows must be 0 (no lookahead available).
                No NaN values permitted.
        """
        ...

    def encode(self, raw: pd.Series) -> pd.Series:
        """Map raw detector codes to PyTorch class indices via encoded_map. Non-override."""
        return raw.map(self.encoded_map).astype(int)
