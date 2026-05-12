"""Per-window causal normalisation for SMC pipeline."""

from __future__ import annotations

import numpy as np

_DEGENERATE_STD_THRESHOLD: float = 1e-8


def normalise_window(window: np.ndarray) -> np.ndarray:
    """
    Normalise a single (60, 5) raw OHLCV window.

    Input:  np.ndarray shape (60, 5), dtype float64. Columns: [open, high, low, close, volume].
    Output: np.ndarray shape (60, 5), dtype float32.

    OHLC columns (0:4):
        z-score using scalar mean and std computed across ALL ohlc cells (60*4=240 values).
        mean = np.mean(window[:, 0:4])
        std  = np.std(window[:, 0:4])
        If std < _DEGENERATE_STD_THRESHOLD, return zeros for OHLC (degenerate window).

    Volume column (4):
        log1p transform, then z-score using within-window volume stats.
        If volume std < _DEGENERATE_STD_THRESHOLD after log1p, return zeros.

    Pure function. No side effects. No state. No NaN in output.
    """
    n_bars = window.shape[0]
    out = np.zeros((n_bars, 5), dtype=np.float32)

    # OHLC normalisation
    ohlc = window[:, :4].astype(np.float64)
    ohlc_mean = np.mean(ohlc)
    ohlc_std = np.std(ohlc)

    if ohlc_std >= _DEGENERATE_STD_THRESHOLD:
        out[:, :4] = ((ohlc - ohlc_mean) / ohlc_std).astype(np.float32)
    # else: leave zeros (degenerate window)

    # Volume normalisation: log1p → z-score
    vol = window[:, 4].astype(np.float64)
    # Guard against negative volume (shouldn't happen, but be safe)
    vol_safe = np.maximum(vol, 0.0)
    log_vol = np.log1p(vol_safe)

    vol_mean = np.mean(log_vol)
    vol_std = np.std(log_vol)

    if vol_std >= _DEGENERATE_STD_THRESHOLD:
        out[:, 4] = ((log_vol - vol_mean) / vol_std).astype(np.float32)
    # else: leave zeros (degenerate volume column)

    return out
