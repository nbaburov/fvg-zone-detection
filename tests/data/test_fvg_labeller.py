"""Tests for FVGLabeller — TDD: written before implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labels.fvg import FVGLabeller
from src.data.labels import LABELLERS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(ohlcv: list[tuple]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with DatetimeIndex."""
    idx = pd.date_range("2020-01-02 09:30", periods=len(ohlcv), freq="1h", tz="America/New_York")
    return pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume"], index=idx)


# spy_9candle_fvg fixture lives in tests/conftest.py — used here via pytest's automatic fixture discovery.

# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------

def test_fvg_registered_in_labellers():
    assert "fvg" in LABELLERS
    assert LABELLERS["fvg"] is FVGLabeller


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------

def test_base_labeller_cannot_instantiate():
    from src.data.labels.base import BaseLabeller
    with pytest.raises(TypeError):
        BaseLabeller()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Falsification fixture — canonical correctness test
# ---------------------------------------------------------------------------

def test_bullish_fvg_label_at_n_plus_1(spy_9candle_fvg):
    labeller = FVGLabeller()
    labels = labeller.label(spy_9candle_fvg)
    assert labels.iloc[4] == 1, f"Expected bullish at index 4, got {labels.iloc[4]}"


def test_bearish_fvg_label_at_n_plus_1(spy_9candle_fvg):
    labeller = FVGLabeller()
    labels = labeller.label(spy_9candle_fvg)
    assert labels.iloc[7] == -1, f"Expected bearish at index 7, got {labels.iloc[7]}"


def test_first_candle_always_zero(spy_9candle_fvg):
    labeller = FVGLabeller()
    labels = labeller.label(spy_9candle_fvg)
    assert labels.iloc[0] == 0, f"First candle must be 0, got {labels.iloc[0]}"


def test_last_candle_always_zero(spy_9candle_fvg):
    labeller = FVGLabeller()
    labels = labeller.label(spy_9candle_fvg)
    assert labels.iloc[8] == 0, f"Last candle must be 0, got {labels.iloc[8]}"


def test_no_other_nonzero_labels(spy_9candle_fvg):
    labeller = FVGLabeller()
    labels = labeller.label(spy_9candle_fvg)
    nonzero_idx = labels[labels != 0].index.tolist()
    expected_nonzero = [spy_9candle_fvg.index[4], spy_9candle_fvg.index[7]]
    assert nonzero_idx == expected_nonzero, f"Unexpected non-zero labels at: {nonzero_idx}"


def test_no_nan_values(spy_9candle_fvg):
    labeller = FVGLabeller()
    labels = labeller.label(spy_9candle_fvg)
    assert not labels.isna().any(), "Labels contain NaN values"


# ---------------------------------------------------------------------------
# Encode tests
# ---------------------------------------------------------------------------

def test_encode_maps_correctly(spy_9candle_fvg):
    labeller = FVGLabeller()
    raw = labeller.label(spy_9candle_fvg)
    encoded = labeller.encode(raw)
    # raw=0 → encoded=0, raw=1 → encoded=1, raw=-1 → encoded=2
    assert all(encoded.isin([0, 1, 2])), f"Encoded values out of range: {encoded.unique()}"
    # Check specific mappings
    assert encoded.iloc[4] == 1   # bullish raw=1 → encoded=1
    assert encoded.iloc[7] == 2   # bearish raw=-1 → encoded=2
    assert encoded.iloc[0] == 0   # none raw=0 → encoded=0


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------

def test_labeller_attributes():
    labeller = FVGLabeller()
    assert labeller.label_index_offset == 1
    assert labeller.num_classes == 3
    assert labeller.class_names == ["none", "bullish", "bearish"]
    assert labeller.encoded_map == {0: 0, 1: 1, -1: 2}


# ---------------------------------------------------------------------------
# Edge case: single-row or two-row DataFrame — all zeros
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_minimal_df_all_zeros(n_rows):
    data = [(100.0, 101.0, 99.0, 100.5, 1000.0)] * n_rows
    df = _make_df(data)
    labeller = FVGLabeller()
    labels = labeller.label(df)
    assert (labels == 0).all(), f"Expected all zeros for {n_rows}-row df, got {labels.tolist()}"
    assert not labels.isna().any()


# ---------------------------------------------------------------------------
# Label is placed at N+1, not N (explicit off-by-one check)
# ---------------------------------------------------------------------------

def test_label_not_at_fvg_candle_itself(spy_9candle_fvg):
    """The FVG pattern closes at candle 3; label must NOT be at 3."""
    labeller = FVGLabeller()
    labels = labeller.label(spy_9candle_fvg)
    assert labels.iloc[3] == 0, f"Label must not appear at the FVG candle itself (index 3), got {labels.iloc[3]}"
    assert labels.iloc[6] == 0, f"Label must not appear at the FVG candle itself (index 6), got {labels.iloc[6]}"
