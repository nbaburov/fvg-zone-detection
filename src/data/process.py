"""Thin orchestration: download → label → encode → persist."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

from src.data.download import download_spy_h1
from src.data.labels import LABELLERS

logger = logging.getLogger(__name__)

_MIN_POSITIVE_RATE: float = 0.01
_WARN_POSITIVE_RATE: float = 0.03


def build_labelled_dataset(
    labeller_name: str = "fvg",
    h1_cache_path: str = "data/raw/spy_minute.parquet",
    output_path: str = "data/processed/spy_h1.parquet",
    use_cache: bool = True,
    start: str = "2018-01-01",
    end: str | None = "2025-12-31",
) -> pd.DataFrame:
    """
    Returns H1 DataFrame with all columns from download_spy_h1()
    plus: raw_label (int, raw detector codes), label (int, PyTorch class index 0/1/2).
    Writes output_path as parquet.
    Raises: ValueError if positive label rate (bull+bear) < 1%.
    Warns: if positive label rate < 3%.
    """
    if labeller_name not in LABELLERS:
        raise KeyError(f"Unknown labeller: '{labeller_name}'. Registered: {list(LABELLERS)}")

    labeller = LABELLERS[labeller_name]()

    df = download_spy_h1(start=start, end=end, use_cache=use_cache, cache_path=h1_cache_path)

    raw_labels = labeller.label(df)
    encoded_labels = labeller.encode(raw_labels)

    df = df.copy()
    df["raw_label"] = raw_labels
    df["label"] = encoded_labels

    # Positive label rate check
    n_total = len(df)
    n_positive = int((raw_labels != 0).sum())
    rate = n_positive / n_total if n_total > 0 else 0.0

    logger.info(
        "Label distribution: none=%.1f%%, positive=%.1f%% (bull+bear) — %d/%d bars",
        (1 - rate) * 100,
        rate * 100,
        n_positive,
        n_total,
    )

    if rate < _MIN_POSITIVE_RATE:
        raise ValueError(
            f"positive label rate is {rate:.3%} ({n_positive}/{n_total} bars). "
            "Expected >= 1%. Check FVG detector against 9-candle falsification fixture."
        )

    if rate < _WARN_POSITIVE_RATE:
        warnings.warn(
            f"positive label rate is {rate:.3%} — below 3% threshold. "
            "Investigate detector before Phase 4.",
            UserWarning,
            stacklevel=2,
        )

    # Persist
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    logger.info("Labelled H1 DataFrame written to %s (%d rows)", out, len(df))

    return df
