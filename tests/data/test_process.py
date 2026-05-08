"""Tests for process.py — build_labelled_dataset."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_h1_fixture(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Synthetic H1 DataFrame with correct schema."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 1.0, n).cumsum()
    prices = np.abs(prices) + 350.0
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.01, 0.5, n),
            "low": prices - rng.uniform(0.01, 0.5, n),
            "close": prices + rng.normal(0, 0.1, n),
            "volume": rng.integers(5000, 20000, n).astype(float),
            "session_type": pd.Categorical(["full"] * n, categories=["full", "half"]),
        },
        index=idx,
    )
    return df


def _make_label_series(df: pd.DataFrame, n_positive: int = 5) -> pd.Series:
    """Return a raw label Series with n_positive non-zero entries."""
    raw = pd.Series(0, index=df.index, dtype=int)
    if n_positive > 0:
        pos_idx = raw.index[5 : 5 + n_positive]
        raw.loc[pos_idx] = 1  # bullish
    return raw


def _make_labeller_dict(h1: pd.DataFrame, n_positive: int = 5) -> dict:
    """Build a fake LABELLERS dict whose 'fvg' class instantiates to a configured mock."""
    labeller_instance = MagicMock()
    raw = _make_label_series(h1, n_positive=n_positive)
    labeller_instance.label.return_value = raw
    labeller_instance.encode.return_value = raw.map({0: 0, 1: 1, -1: 2})

    labeller_class = MagicMock(return_value=labeller_instance)
    return {"fvg": labeller_class}, labeller_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("src.data.process.download_spy_h1")
def test_output_has_raw_label_and_label_columns(mock_download):
    """Output DataFrame must have raw_label and label columns."""
    from src.data.process import build_labelled_dataset

    h1 = _make_h1_fixture(50)
    mock_download.return_value = h1
    fake_labellers, _ = _make_labeller_dict(h1, n_positive=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "spy_h1.parquet")
        with patch("src.data.process.LABELLERS", fake_labellers):
            result = build_labelled_dataset(
                labeller_name="fvg",
                h1_cache_path=os.path.join(tmpdir, "spy_minute.parquet"),
                output_path=out,
                use_cache=False,
            )

    assert "raw_label" in result.columns, "Missing raw_label column"
    assert "label" in result.columns, "Missing label column"


@patch("src.data.process.download_spy_h1")
def test_label_values_in_valid_range(mock_download):
    """label column must contain only values in {0, 1, 2}."""
    from src.data.process import build_labelled_dataset

    h1 = _make_h1_fixture(50)
    mock_download.return_value = h1
    fake_labellers, _ = _make_labeller_dict(h1, n_positive=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "spy_h1.parquet")
        with patch("src.data.process.LABELLERS", fake_labellers):
            result = build_labelled_dataset(
                labeller_name="fvg",
                h1_cache_path=os.path.join(tmpdir, "spy_minute.parquet"),
                output_path=out,
                use_cache=False,
            )

    invalid = result["label"][~result["label"].isin([0, 1, 2])]
    assert len(invalid) == 0, f"Invalid label values: {invalid.unique()}"


@patch("src.data.process.download_spy_h1")
def test_raises_value_error_when_too_few_positives(mock_download):
    """ValueError raised when bull+bear < 1% of all bars."""
    from src.data.process import build_labelled_dataset

    n = 200
    h1 = _make_h1_fixture(n)
    mock_download.return_value = h1
    # 0 positives → 0% rate → must raise
    fake_labellers, _ = _make_labeller_dict(h1, n_positive=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "spy_h1.parquet")
        with patch("src.data.process.LABELLERS", fake_labellers):
            with pytest.raises(ValueError, match="positive label rate"):
                build_labelled_dataset(
                    labeller_name="fvg",
                    h1_cache_path=os.path.join(tmpdir, "spy_minute.parquet"),
                    output_path=out,
                    use_cache=False,
                )


@patch("src.data.process.download_spy_h1")
def test_raises_when_below_1pct_boundary(mock_download):
    """1 positive in 200 rows = 0.5% < 1% must raise ValueError."""
    from src.data.process import build_labelled_dataset

    n = 200
    h1 = _make_h1_fixture(n)
    mock_download.return_value = h1
    # 1 positive / 200 rows = 0.5% — below 1% threshold
    fake_labellers, _ = _make_labeller_dict(h1, n_positive=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "spy_h1.parquet")
        with patch("src.data.process.LABELLERS", fake_labellers):
            with pytest.raises(ValueError, match="positive label rate"):
                build_labelled_dataset(
                    labeller_name="fvg",
                    h1_cache_path=os.path.join(tmpdir, "spy_minute.parquet"),
                    output_path=out,
                    use_cache=False,
                )


@patch("src.data.process.download_spy_h1")
def test_does_not_raise_above_1pct_boundary(mock_download):
    """3 positives in 200 rows = 1.5% >= 1% must NOT raise ValueError."""
    import warnings
    from src.data.process import build_labelled_dataset

    n = 200
    h1 = _make_h1_fixture(n)
    mock_download.return_value = h1
    # 3 positives / 200 rows = 1.5% — above 1% threshold (but below 3% → warning)
    fake_labellers, _ = _make_labeller_dict(h1, n_positive=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "spy_h1.parquet")
        with patch("src.data.process.LABELLERS", fake_labellers):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = build_labelled_dataset(
                    labeller_name="fvg",
                    h1_cache_path=os.path.join(tmpdir, "spy_minute.parquet"),
                    output_path=out,
                    use_cache=False,
                )
    assert result is not None, "Expected result for 1.5% positive rate"


@patch("src.data.process.download_spy_h1")
def test_parquet_written_at_output_path(mock_download):
    """Output parquet file must be created at the specified path."""
    from src.data.process import build_labelled_dataset

    h1 = _make_h1_fixture(50)
    mock_download.return_value = h1
    fake_labellers, _ = _make_labeller_dict(h1, n_positive=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "spy_h1.parquet")
        with patch("src.data.process.LABELLERS", fake_labellers):
            build_labelled_dataset(
                labeller_name="fvg",
                h1_cache_path=os.path.join(tmpdir, "spy_minute.parquet"),
                output_path=out,
                use_cache=False,
            )
        assert os.path.exists(out), f"Parquet file not written at {out}"
