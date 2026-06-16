"""Sliding window dataset builder for SMC pipeline."""

from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.labels.base import BaseLabeller
from src.data.normalize import normalise_window
from src.data.timeframe import H1 as _H1

# Maximum gap between consecutive candles to consider them same session.
# Single source of truth: Timeframe.H1.max_intra_window_gap_minutes (= 90 = 1.5 * 60).
_MAX_INTRA_WINDOW_GAP_MINUTES = _H1.max_intra_window_gap_minutes


def _has_session_gap(
    df: pd.DataFrame,
    start: int,
    end: int,
    max_gap_minutes: int = _MAX_INTRA_WINDOW_GAP_MINUTES,
) -> bool:
    """
    Return True if any two consecutive bars in df[start:end] are more than
    *max_gap_minutes* apart (i.e., a holiday/weekend gap).

    Args:
        df: DataFrame with a DatetimeIndex.
        start: Slice start index (inclusive).
        end:   Slice end index (exclusive).
        max_gap_minutes: Gap threshold in minutes. Defaults to the module-level
            H1 constant (90 min). Pass ``tf.max_intra_window_gap_minutes`` to
            make this TF-aware.
    """
    idx = df.index[start:end]
    if len(idx) < 2:
        return False
    diffs = idx[1:] - idx[:-1]
    actual_max = diffs.max().total_seconds() / 60
    return actual_max > max_gap_minutes


def _window_generator(
    df: pd.DataFrame,
    labeller: BaseLabeller,
    stride: int,
    window_size: int,
    drop_cross_session_windows: bool,
    max_gap_minutes: int = _MAX_INTRA_WINDOW_GAP_MINUTES,
) -> Generator[tuple[np.ndarray, int], None, None]:
    """
    Yield (normalised_window_array, encoded_label) for each valid window.

    Window validity rules:
    1. Label candle is at position i + window_size - 1 (i.e., i+59 for 60-bar windows).
    2. The label at that position must be present in df['label'] (already encoded).
    3. If drop_cross_session_windows=True, skip windows with any consecutive pair >90 min apart.
    4. If df contains a 'symbol' column, skip any window whose bars span more than one symbol
       (guards against cross-symbol boundary contamination after multi-symbol pooling).
       This check is entirely optional — single-symbol DataFrames without the column are
       unaffected and behave identically to the pre-multi-symbol code path.
    """
    n = len(df)
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    ohlcv = df[ohlcv_cols].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int64)
    has_symbol_col: bool = "symbol" in df.columns
    symbols = df["symbol"].to_numpy() if has_symbol_col else None

    for i in range(0, n - window_size + 1, stride):
        label_pos = i + window_size - 1
        if label_pos >= n:
            break

        if drop_cross_session_windows and _has_session_gap(df, i, i + window_size, max_gap_minutes):
            continue

        if has_symbol_col and len(set(symbols[i : i + window_size])) > 1:
            continue

        raw_window = ohlcv[i : i + window_size]
        normalised = normalise_window(raw_window)
        label = int(labels[label_pos])

        yield normalised, label


def build_windows(
    df: pd.DataFrame,
    labeller: BaseLabeller,
    stride: int = 1,
    window_size: int = 60,
    drop_cross_session_windows: bool = True,
    max_gap_minutes: int = _MAX_INTRA_WINDOW_GAP_MINUTES,
) -> list[tuple[np.ndarray, int]]:
    """
    Returns list of (window_array shape (window_size,5) float32, encoded_label int).
    Calls normalise_window on each window before returning.

    Args:
        max_gap_minutes: Gap threshold for session-gap detection.
            Defaults to 90 (H1). Pass ``tf.max_intra_window_gap_minutes`` for
            TF-aware behaviour.
    """
    return list(
        _window_generator(df, labeller, stride, window_size, drop_cross_session_windows, max_gap_minutes)
    )


class SMCWindowDataset(Dataset):
    """
    PyTorch Dataset that iterates (window_tensor: Tensor[60, 5], label: int) pairs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labeller: BaseLabeller,
        stride: int = 1,
        window_size: int = 60,
        drop_cross_session_windows: bool = True,
        max_gap_minutes: int = _MAX_INTRA_WINDOW_GAP_MINUTES,
    ) -> None:
        self._windows = build_windows(df, labeller, stride, window_size, drop_cross_session_windows, max_gap_minutes)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        window_arr, label = self._windows[idx]
        tensor = torch.from_numpy(window_arr)  # float32 already
        return tensor, label

    @property
    def label_counts(self) -> dict[int, int]:
        """Returns {class_index: count} for all windows in this dataset."""
        counts: dict[int, int] = {}
        for _, label in self._windows:
            counts[label] = counts.get(label, 0) + 1
        return counts
