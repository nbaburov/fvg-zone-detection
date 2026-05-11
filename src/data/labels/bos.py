"""Causal Break of Structure (BOS) detection — pure functions, no classes."""

from __future__ import annotations

import pandas as pd


def compute_bos(
    df: pd.DataFrame,
    swing_lookback: int = 50,
    bos_lookback: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute causal BOS (Break of Structure) indicators.

    A bullish BOS at bar j = close[j] > max(high[j-swing_lookback..j-1]).
    A bearish BOS at bar j = close[j] < min(low[j-swing_lookback..j-1]).

    bull_bos_recent[i] = True if any bullish BOS occurred in bars [i-bos_lookback..i-1].
    bear_bos_recent[i] = True if any bearish BOS occurred in bars [i-bos_lookback..i-1].

    Both use shift(1) — BOS must precede bar i, not occur at bar i.
    Fully vectorised pandas.

    Args:
        df:             DataFrame with 'high', 'low', 'close' columns.
        swing_lookback: Lookback window to define the prior swing high/low.
        bos_lookback:   Recency window to check if a BOS has occurred recently.

    Returns:
        (bull_bos_recent, bear_bos_recent): Two bool Series with same index as df.
    """
    # Prior swing high/low — causal (shift 1 so bar i uses only bars before i)
    prior_swing_high = df["high"].rolling(swing_lookback, min_periods=1).max().shift(1)
    prior_swing_low = df["low"].rolling(swing_lookback, min_periods=1).min().shift(1)

    # Bullish BOS: close breaks above prior swing high
    bull_bos = (df["close"] > prior_swing_high).astype(int)
    # Bearish BOS: close breaks below prior swing low
    bear_bos = (df["close"] < prior_swing_low).astype(int)

    # "Recent BOS": was there any BOS in the past bos_lookback bars?
    # max of int indicator over rolling window — then shift(1) so bar i checks bars before it
    bull_bos_recent = (
        bull_bos.rolling(bos_lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
    )
    bear_bos_recent = (
        bear_bos.rolling(bos_lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
    )

    return bull_bos_recent, bear_bos_recent
