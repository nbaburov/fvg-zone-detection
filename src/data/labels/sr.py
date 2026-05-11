"""Causal pivot high/low support and resistance levels — pure functions, no classes."""

from __future__ import annotations

import pandas as pd


def compute_pivot_levels(
    df: pd.DataFrame,
    lookback: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute causal pivot high and low S/R levels.

    pivot_high[i] = max(high[i-lookback..i-1])  — rolling max, shift(1).
    pivot_low[i]  = min(low[i-lookback..i-1])   — rolling min, shift(1).

    NaN for first `lookback` bars (insufficient history).
    Does NOT use bar i itself — fully causal.

    Args:
        df:       DataFrame with 'high' and 'low' columns.
        lookback: Number of past bars to compute the rolling extreme over.

    Returns:
        (pivot_high, pivot_low): Two Series with same index as df.
    """
    pivot_high = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    pivot_low = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    return pivot_high, pivot_low
