"""Tests for split.py — temporal train/val/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split import temporal_split, SPLIT_BOUNDARIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_full_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Synthetic H1 DataFrame spanning 2018–2024 with mock labels.
    One bar per calendar day (simplified — real data has many per day but
    this is enough to cover the split boundary logic).
    """
    rng = np.random.default_rng(seed)
    # ~2000 trading days from 2018-01-02 to 2024-12-31
    idx = pd.bdate_range("2018-01-02", "2024-12-31", freq="B", tz="America/New_York")
    n = len(idx)
    prices = 400.0 + rng.normal(0, 1.0, n).cumsum() + 350.0
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.01, 0.5, n),
            "low": prices - rng.uniform(0.01, 0.5, n),
            "close": prices + rng.normal(0, 0.1, n),
            "volume": rng.integers(5000, 20000, n).astype(float),
            "session_type": pd.Categorical(["full"] * n, categories=["full", "half"]),
            "raw_label": rng.integers(0, 2, n),
            "label": rng.integers(0, 3, n),
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_rows_lost_or_duplicated():
    """len(train) + len(val) + len(test) == len(df)."""
    df = _make_full_dataset()
    train, val, test = temporal_split(df)
    assert len(train) + len(val) + len(test) == len(df), (
        f"Row count mismatch: {len(train)} + {len(val)} + {len(test)} != {len(df)}"
    )


def test_strict_temporal_ordering():
    """train.index.max() < val.index.min() and val.index.max() < test.index.min()."""
    df = _make_full_dataset()
    train, val, test = temporal_split(df)
    assert train.index.max() < val.index.min(), (
        f"train max {train.index.max()} >= val min {val.index.min()}"
    )
    assert val.index.max() < test.index.min(), (
        f"val max {val.index.max()} >= test min {test.index.min()}"
    )


def test_default_boundaries_produce_expected_date_ranges():
    """Default boundaries produce splits in the correct year ranges."""
    df = _make_full_dataset()
    train, val, test = temporal_split(df)

    assert train.index.min().year >= 2018
    assert train.index.max().year <= 2021
    assert val.index.min().year == 2022
    assert val.index.max().year == 2022
    assert test.index.min().year == 2023
    assert test.index.max().year <= 2024


def test_raises_on_empty_split():
    """ValueError raised when a boundary excludes all rows from a split."""
    df = _make_full_dataset()
    # Boundaries that make val empty (val range before dataset starts)
    bad_boundaries = {
        "train_end": "2017-12-31",
        "val_start": "2016-01-01",
        "val_end": "2016-12-31",
        "test_start": "2023-01-01",
        "test_end": "2024-12-31",
    }
    with pytest.raises(ValueError):
        temporal_split(df, boundaries=bad_boundaries)


def test_no_index_overlap_between_splits():
    """No bar index appears in more than one split."""
    df = _make_full_dataset()
    train, val, test = temporal_split(df)

    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)

    assert len(train_idx & val_idx) == 0, "Overlap between train and val"
    assert len(val_idx & test_idx) == 0, "Overlap between val and test"
    assert len(train_idx & test_idx) == 0, "Overlap between train and test"


def test_post_test_end_rows_are_clipped():
    """Rows after test_end must be excluded; invariant holds on filtered_df.

    test_end is now 2025-12-31, so we inject a bar at 2026-01-02 (outside boundary).
    """
    df = _make_full_dataset()

    # Inject one bar at 2026-01-02 — outside test_end=2025-12-31
    future_idx = pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="America/New_York")])
    future_row = df.iloc[[-1]].copy()
    future_row.index = future_idx
    df_with_future = pd.concat([df, future_row])

    train, val, test = temporal_split(df_with_future)

    # The 2026-01-02 bar must NOT appear in any split
    all_split_idx = set(train.index) | set(val.index) | set(test.index)
    assert future_idx[0] not in all_split_idx, "2026-01-02 bar leaked into splits"

    # Row-count invariant holds against filtered (non-future) rows
    filtered_count = len(df_with_future[df_with_future.index.normalize() <= pd.Timestamp("2025-12-31", tz="America/New_York")])
    assert len(train) + len(val) + len(test) == filtered_count, (
        f"Row count mismatch after clip: {len(train)} + {len(val)} + {len(test)} != {filtered_count}"
    )
