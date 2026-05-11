"""test_runner_integration.py — Integration tests for the inspection runner.

Does NOT load real checkpoints. Uses stub adapters for all inference.
Exercises real windowing logic against the val parquet (or a synthetic DataFrame
if the parquet is unavailable in CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inspect.base import ModelAdapter
from src.inspect.runner import InspectionResults, run

# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------

class _ZeroAdapter(ModelAdapter):
    """Always predicts class 0 (none)."""

    name = "stub_zero"

    def __init__(self, checkpoint_dir=None, **_):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _UniformAdapter(ModelAdapter):
    """Always returns uniform distribution."""

    name = "stub_uniform"

    def __init__(self, checkpoint_dir=None, **_):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        return np.full((n, 3), 1.0 / 3.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VAL_PARQUET = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "spy_h1_val.parquet"
)


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    """Minimal synthetic H1 DataFrame with encoded label column."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-03 09:30", periods=n, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 1.0, n).cumsum()
    prices = np.abs(prices) + 100.0
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.1, 1.0, n),
            "low": prices - rng.uniform(0.1, 1.0, n),
            "close": prices + rng.normal(0, 0.2, n),
            "volume": rng.integers(5000, 20000, n).astype(float),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    # Add encoded label (mostly 0, a few 1s and 2s)
    labels = np.zeros(n, dtype=int)
    labels[80:85] = 1
    labels[140:143] = 2
    df["label"] = labels
    return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    """200-bar synthetic DataFrame with label column."""
    return _make_synthetic_df(200)


@pytest.fixture
def labeller():
    """ValidFVGLabeller instance (used for metadata only in runner)."""
    from src.data.labels.valid_fvg import ValidFVGLabeller
    return ValidFVGLabeller()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_runner_produces_windows(small_df, labeller):
    adapters = [_ZeroAdapter()]
    results = run(adapters, small_df, labeller, window_size=60, stride=1)
    assert isinstance(results, InspectionResults)
    assert results.n > 0
    assert results.n == len(results.labels)
    assert results.windows_raw.shape == (results.n, 60, 5)
    assert results.windows_norm.shape == (results.n, 60, 5)


def test_runner_two_adapters_correct_shapes(small_df, labeller):
    adapters = [_ZeroAdapter(), _UniformAdapter()]
    results = run(adapters, small_df, labeller, window_size=60, stride=1)

    assert set(results.model_names) == {"stub_zero", "stub_uniform"}
    for name in results.model_names:
        assert results.probas[name].shape == (results.n, 3)
        assert results.preds[name].shape == (results.n,)


def test_runner_zero_adapter_all_class_zero(small_df, labeller):
    adapters = [_ZeroAdapter()]
    results = run(adapters, small_df, labeller, window_size=60)
    assert np.all(results.preds["stub_zero"] == 0)


def test_runner_uniform_adapter_probas_sum_to_one(small_df, labeller):
    adapters = [_UniformAdapter()]
    results = run(adapters, small_df, labeller, window_size=60)
    row_sums = results.probas["stub_uniform"].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_runner_timestamps_align_with_last_bar(small_df, labeller):
    adapters = [_ZeroAdapter()]
    results = run(adapters, small_df, labeller, window_size=60, stride=1)
    # Each timestamp must be in small_df.index
    assert len(results.timestamps) == results.n
    for ts in results.timestamps:
        assert ts in small_df.index


def test_runner_slice_too_short_returns_empty(labeller):
    short_df = _make_synthetic_df(30)  # fewer than 60 bars
    adapters = [_ZeroAdapter()]
    results = run(adapters, short_df, labeller, window_size=60)
    assert results.n == 0
    assert results.windows_raw.shape == (0, 60, 5)
    assert len(results.timestamps) == 0


def test_runner_all_zero_positive_labels_no_crash(labeller):
    """Runner should not crash when no FVG-positive labels exist in the slice."""
    df = _make_synthetic_df(200)
    df["label"] = 0  # all negative
    adapters = [_ZeroAdapter(), _UniformAdapter()]
    results = run(adapters, df, labeller, window_size=60)
    assert results.n > 0
    assert np.all(results.labels == 0)


def test_runner_missing_columns_raises_value_error(labeller):
    df = _make_synthetic_df(100)
    df = df.drop(columns=["label"])
    adapters = [_ZeroAdapter()]
    with pytest.raises(ValueError, match="missing required columns"):
        run(adapters, df, labeller)


def test_runner_adapter_shape_mismatch_raises(small_df, labeller):
    """Adapter returning wrong shape triggers ValueError."""

    class _BadShapeAdapter(ModelAdapter):
        name = "bad_shape"
        def __init__(self, *a, **kw): pass
        def predict_proba(self, windows):
            n = windows.shape[0]
            return np.zeros((n, 2), dtype=np.float32)  # wrong: 2 classes not 3

    adapters = [_BadShapeAdapter()]
    with pytest.raises(ValueError, match="bad_shape"):
        run(adapters, small_df, labeller)


@pytest.mark.skipif(
    not _VAL_PARQUET.exists(),
    reason="Val parquet not present; skipping real-data integration test.",
)
def test_runner_on_real_val_parquet_5_windows(labeller):
    """Load real val parquet, take a tiny slice, assert 5+ windows are produced."""
    df = pd.read_parquet(_VAL_PARQUET)
    # Take first 200 bars
    df_slice = df.head(200)
    adapters = [_ZeroAdapter()]
    results = run(adapters, df_slice, labeller, window_size=60, stride=1)
    assert results.n >= 5, f"Expected >= 5 windows, got {results.n}"
    assert results.windows_raw.shape[1] == 60
    assert results.windows_raw.shape[2] == 5
