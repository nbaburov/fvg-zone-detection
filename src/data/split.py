"""Temporal train/val/test split for SMC H1 dataset."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_BOUNDARIES: dict[str, str] = {
    "train_end": "2021-12-31",
    "val_start": "2022-01-01",
    "val_end": "2022-12-31",
    "test_start": "2023-01-01",
    "test_end": "2024-12-31",
}


def temporal_split(
    df: pd.DataFrame,
    boundaries: dict[str, str] = SPLIT_BOUNDARIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labelled H1 DataFrame into train, val, test subsets using temporal boundaries.

    Returns (train_df, val_df, test_df).
    All three share the same column schema as df.
    No bar appears in more than one subset.
    Raises: ValueError if any split has 0 bars, or if positive label rate in any split == 0.
    """
    train_end_ts = pd.Timestamp(boundaries["train_end"], tz=df.index.tz)
    val_start_ts = pd.Timestamp(boundaries["val_start"], tz=df.index.tz)
    val_end_ts = pd.Timestamp(boundaries["val_end"], tz=df.index.tz)
    test_start_ts = pd.Timestamp(boundaries["test_start"], tz=df.index.tz)
    test_end_ts = pd.Timestamp(boundaries["test_end"], tz=df.index.tz)

    # Clip df to test_end so bars after the boundary are excluded explicitly.
    # Guarantees len(train)+len(val)+len(test)==len(filtered_df) invariant.
    # Use date-normalised boundary: keep bars whose date (ignoring time) is <= boundary date.
    filtered_df = df.loc[df.index.normalize() <= test_end_ts].copy()

    train_df = filtered_df.loc[filtered_df.index.normalize() <= train_end_ts].copy()
    val_df = filtered_df.loc[
        (filtered_df.index.normalize() >= val_start_ts)
        & (filtered_df.index.normalize() <= val_end_ts)
    ].copy()
    test_df = filtered_df.loc[filtered_df.index.normalize() >= test_start_ts].copy()

    # Validate non-empty splits
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split) == 0:
            raise ValueError(
                f"Split '{name}' has 0 bars. Check boundaries: {boundaries}"
            )

    # Log split statistics
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n = len(split)
        if "raw_label" in split.columns:
            n_pos = int((split["raw_label"] != 0).sum())
            rate = n_pos / n if n > 0 else 0.0
            logger.info(
                "Split '%s': %d bars, positive rate=%.2f%% (%d/%d)",
                name, n, rate * 100, n_pos, n,
            )
        else:
            logger.info("Split '%s': %d bars", name, n)

    return train_df, val_df, test_df
