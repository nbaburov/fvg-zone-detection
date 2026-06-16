"""tests/data/test_positive_rate_gate.py — M2: pipeline-level positive-rate gate.

build_pipeline raises when the post-label per-class FVG positive rate is out of
band ([0.3%, 15%]); passes when in-band; and is bypassable via
``enforce_positive_rate_gate=False`` for synthetic/unit data.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _make_df(raw_labels: np.ndarray) -> pd.DataFrame:
    """Build a continuous-hourly labelled H1 df with the given raw_label vector.

    Covers 2018–2024 so the temporal split yields non-empty splits. ``label``
    is the encoded {0,1,2} form (only needed for downstream windowing).
    """
    n = len(raw_labels)
    idx = pd.date_range("2018-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
    rng = np.random.default_rng(0)
    prices = np.abs(400.0 + rng.normal(0, 0.5, n).cumsum() + 350.0)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 10000.0,
            "session_type": pd.Categorical(["full"] * n, categories=["full", "half"]),
            "raw_label": raw_labels,
            "label": np.where(raw_labels == 1, 1, np.where(raw_labels == -1, 2, 0)),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def _in_band_labels(n: int = 70000) -> np.ndarray:
    """~1.5% bull + ~1.5% bear (matches H1 ValidFVG baseline)."""
    labels = np.zeros(n, dtype=int)
    step = int(1 / 0.015)  # ~every 66th bar
    labels[::step] = 1          # bull
    labels[step // 2 :: step] = -1  # bear (offset so they don't collide)
    return labels


def _out_of_band_labels(n: int = 70000) -> np.ndarray:
    """0% positive — degenerate, below the 0.3% floor."""
    return np.zeros(n, dtype=int)


class _StubDataset:
    """Stub SMCWindowDataset isolating the gate from the windowing stage.

    Windowing recomputes labels from OHLCV, so injected raw_labels never reach
    it; stubbing it lets us assert solely on whether the positive-rate GATE
    raises (M2), not on unrelated downstream window-count checks.
    """

    def __init__(self, df, labeller, **kw):
        self._df = df

    def __len__(self):
        return 1

    @property
    def label_counts(self):
        return {0: 1, 1: 1, 2: 1}  # non-degenerate so the window check passes


def _run(raw_labels, **kwargs):
    full_df = _make_df(raw_labels)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.data.pipeline.build_labelled_dataset", return_value=full_df), \
             patch("src.data.pipeline.PROCESSED_DIR", tmpdir), \
             patch("src.data.pipeline.SMCWindowDataset", _StubDataset), \
             patch("src.data.pipeline._compute_class_weights",
                   return_value=__import__("torch").ones(3)):
            from src.data.pipeline import build_pipeline

            return build_pipeline(labeller_name="fvg_valid", window_size=60, **kwargs)


def test_out_of_band_raises():
    """0% positive rate trips the gate → ValueError."""
    with pytest.raises(ValueError, match="Positive-rate gate FAILED"):
        _run(_out_of_band_labels())


def test_in_band_passes():
    """~1.5%/class is in band → no raise (returns datasets)."""
    train_ds, val_ds, test_ds, weights = _run(_in_band_labels())
    assert train_ds is not None and val_ds is not None and test_ds is not None


def test_gate_disabled_no_raise():
    """enforce_positive_rate_gate=False bypasses the gate even on 0% data."""
    train_ds, _, _, _ = _run(_out_of_band_labels(), enforce_positive_rate_gate=False)
    assert train_ds is not None
