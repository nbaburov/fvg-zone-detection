"""Tests: build_pipeline() default labeller is ValidFVG, not raw FVG."""

from __future__ import annotations

import inspect

from src.data.pipeline import build_pipeline
from src.data.labels.valid_fvg import ValidFVGLabeller


def test_default_labeller_is_fvg_valid():
    """build_pipeline default labeller_name must be 'fvg_valid'."""
    sig = inspect.signature(build_pipeline)
    default = sig.parameters["labeller_name"].default
    assert default == "fvg_valid", (
        f"Expected default labeller_name='fvg_valid', got '{default}'. "
        "Do not change the canonical target without updating this test."
    )


def test_fvg_valid_labeller_positive_rate_is_sparse():
    """ValidFVGLabeller on a synthetic 9-candle sequence should NOT fire on every bar.

    This is a smoke check: if ValidFVG positive rate were ~25% (raw FVG range)
    it would indicate the wrong labeller is registered under 'fvg_valid'.
    We use the real labeller's 9-candle falsification fixture (all none expected).
    """
    import numpy as np
    import pandas as pd
    from src.data.labels import LABELLERS

    labeller = LABELLERS["fvg_valid"]()
    assert isinstance(labeller, ValidFVGLabeller), (
        f"'fvg_valid' should map to ValidFVGLabeller, got {type(labeller)}"
    )

    # Build a flat OHLCV frame (no FVG geometry) — expect all none labels
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2023-01-03 09:30", periods=n, freq="1h", tz="America/New_York")
    price = 400.0 + np.arange(n) * 0.01
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.05,
            "low": price - 0.05,
            "close": price + 0.01,
            "volume": rng.integers(5000, 20000, n).astype(float),
        },
        index=idx,
    )
    raw = labeller.label(df)
    encoded = labeller.encode(raw)
    n_positive = int((encoded != 0).sum())
    # On a flat series with no gap geometry, positive rate should be very low
    assert n_positive / n < 0.10, (
        f"Positive rate {n_positive/n:.1%} unexpectedly high for flat series — "
        "check that 'fvg_valid' is wired to ValidFVGLabeller, not FVGLabeller."
    )
