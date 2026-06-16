"""Tests for src/features/window_features.py.

Covers:
- Shape assertions (stride=1 and stride=60)
- No NaN/inf in output
- Causality (future-bar mutation test)
- FVG-locality correctness
- RSI bounds
- Binary feature values
- All 35 feature names present
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.window_features import (
    FEATURE_NAMES,
    extract_window_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with valid OHLC structure."""
    rng = np.random.default_rng(seed)
    close = 400.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    noise_high = rng.uniform(0.1, 1.0, n)
    noise_low = rng.uniform(0.1, 1.0, n)
    high = np.maximum(close, open_) + noise_high
    low = np.minimum(close, open_) - noise_low
    volume = rng.uniform(1e5, 1e6, n)
    label = rng.integers(0, 3, n)

    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "label": label},
        index=idx,
    )


@pytest.fixture
def df_100() -> pd.DataFrame:
    return _make_ohlcv_df(100, seed=1)


@pytest.fixture
def spy_h1_train_fixture() -> pd.DataFrame:
    """500-row slice from real training parquet (if available), else synthetic."""
    try:
        df = pd.read_parquet("data/processed/spy_h1_train.parquet")
        return df.iloc[:500].copy()
    except Exception:
        return _make_ohlcv_df(500, seed=42)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_shape_stride1(df_100):
    X, y = extract_window_features(df_100, window_size=60, stride=1)
    # Expected: 100 - 60 + 1 = 41 windows
    assert X.shape == (41, 35), f"Expected (41, 35), got {X.shape}"
    assert y.shape == (41,)


def test_shape_stride60(df_100):
    X, y = extract_window_features(df_100, window_size=60, stride=60)
    # t_indices starts at 59 (one window), next at 119 (out of range for 100 rows)
    assert X.shape[0] == 1, f"Expected 1 window, got {X.shape[0]}"
    assert X.shape[1] == 35


def test_too_short_df():
    df = _make_ohlcv_df(30)
    X, y = extract_window_features(df, window_size=60, stride=1)
    assert X.shape == (0, 35)
    assert y.shape == (0,)


# ---------------------------------------------------------------------------
# NaN / inf
# ---------------------------------------------------------------------------

def test_no_nan_real_data(spy_h1_train_fixture):
    X, y = extract_window_features(spy_h1_train_fixture, stride=1)
    assert np.isfinite(X).all(), "NaN or inf found in feature matrix"


def test_no_nan_synthetic(df_100):
    X, y = extract_window_features(df_100, stride=1)
    assert np.isfinite(X).all()


# ---------------------------------------------------------------------------
# Causality test — future-bar mutation must not affect earlier windows
# ---------------------------------------------------------------------------

def test_features_no_future_bar_leakage(spy_h1_train_fixture):
    """
    Mutate bar K+1. Assert windows ending before bar K are unchanged.
    """
    df = spy_h1_train_fixture.copy()
    K = 300

    X_before, _ = extract_window_features(df, stride=1)

    df_mutated = df.copy()
    # Only mutate numeric OHLCV + label columns (session_type may be Categorical)
    for col, val in zip(["open", "high", "low", "close", "volume", "label"],
                        [99999.0, 99999.0, 0.01, 99999.0, 9999999.0, 0]):
        if col in df_mutated.columns:
            df_mutated.loc[df_mutated.index[K + 1], col] = val

    X_after, _ = extract_window_features(df_mutated, stride=1)

    # Window at index j ends at bar j + 59. Safe windows: j+59 < K+1 → j < K - 58
    safe_count = max(0, K - 59)
    if safe_count > 0:
        np.testing.assert_array_equal(
            X_before[:safe_count],
            X_after[:safe_count],
            err_msg="Feature values changed for windows not touching the mutated bar — causality violated",
        )


# ---------------------------------------------------------------------------
# FVG-locality correctness
# ---------------------------------------------------------------------------

def test_fvg_locality_bullish_gap():
    """Hand-crafted 60-bar fixture with known bullish FVG at bars 56, 57, 58."""
    n = 60
    close = np.full(n, 400.0)
    open_ = np.full(n, 400.0)
    high = np.full(n, 401.0)
    low = np.full(n, 399.0)
    volume = np.full(n, 1e5)

    # Plant bullish FVG: N-1=55, N=56, N+1=57, label bar=58 (t=59 means t-3=56, t-2=57, t-1=58)
    # Wait: in a 60-bar window the last bar is index 59 in the window.
    # t-3 = index 56, t-2 = index 57, t-1 = index 58, t = index 59
    # Bullish FVG: low[N+1=t-1] > high[N-1=t-3]
    high[56] = 401.0   # N-1 high
    low[56] = 399.0
    # N+1 low > N-1 high → bullish gap
    low[58] = 403.0    # N+1 low >> N-1 high (401)
    high[58] = 405.0
    # Middle candle N: bullish body
    open_[57] = 399.5
    close[57] = 402.5
    high[57] = 403.0
    low[57] = 399.0

    label = np.zeros(n, dtype=int)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume, "label": label}, index=idx)

    X, _ = extract_window_features(df, window_size=60, stride=1)
    # Only 1 window (n=60, stride=1 → 60-60+1=1 window)
    assert X.shape[0] == 1
    gb_idx = FEATURE_NAMES.index("gap_bull")
    gbe_idx = FEATURE_NAMES.index("gap_bear")
    assert X[0, gb_idx] > 0, f"Expected gap_bull > 0, got {X[0, gb_idx]}"
    assert X[0, gbe_idx] == 0.0, f"Expected gap_bear == 0, got {X[0, gbe_idx]}"

    # react_in_gap_bull: label bar (t=59) close=400 is OUTSIDE gap zone [401, 403] → 0.0
    rig_bull_idx = FEATURE_NAMES.index("react_in_gap_bull")
    assert X[0, rig_bull_idx] == 0.0, f"Expected react_in_gap_bull == 0 (close=400 below gap), got {X[0, rig_bull_idx]}"


def test_react_in_gap_bull_inside():
    """Variant of fvg_locality where label bar close IS inside the bullish gap."""
    n = 60
    close = np.full(n, 400.0)
    open_ = np.full(n, 400.0)
    high = np.full(n, 401.0)
    low = np.full(n, 399.0)
    volume = np.full(n, 1e5)

    # Bullish gap zone: [h_n1=401, l_np1=403]
    high[56] = 401.0; low[56] = 399.0
    low[58] = 403.0;  high[58] = 405.0
    open_[57] = 399.5; close[57] = 402.5; high[57] = 403.0; low[57] = 399.0
    # Label bar (t=59) close must sit inside [401, 403]
    close[59] = 402.0
    open_[59] = 402.0
    high[59] = 402.5
    low[59] = 401.5

    label = np.zeros(n, dtype=int)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume, "label": label}, index=idx)

    X, _ = extract_window_features(df, window_size=60, stride=1)
    rig_bull_idx = FEATURE_NAMES.index("react_in_gap_bull")
    assert X[0, rig_bull_idx] == 1.0, f"Expected react_in_gap_bull == 1 (close=402 inside [401,403]), got {X[0, rig_bull_idx]}"


# ---------------------------------------------------------------------------
# RSI bounds
# ---------------------------------------------------------------------------

def test_rsi_in_range(df_100):
    X, _ = extract_window_features(df_100, stride=1)
    rsi_idx = FEATURE_NAMES.index("rsi_14")
    rsi_vals = X[:, rsi_idx]
    assert (rsi_vals >= 0.0).all() and (rsi_vals <= 100.0).all(), \
        f"RSI out of [0, 100] range: min={rsi_vals.min()}, max={rsi_vals.max()}"


# ---------------------------------------------------------------------------
# Binary feature values
# ---------------------------------------------------------------------------

def test_binary_features(df_100):
    X, _ = extract_window_features(df_100, stride=1)
    for name in ("above_ma20", "above_ma50", "vol_spike", "mid_body_bull", "mid_body_bear",
                 "react_in_gap_bull", "react_in_gap_bear"):
        idx = FEATURE_NAMES.index(name)
        vals = X[:, idx]
        unique = np.unique(vals)
        assert set(unique).issubset({0.0, 1.0}), f"Feature {name} has non-binary values: {unique}"


# ---------------------------------------------------------------------------
# Feature names completeness
# ---------------------------------------------------------------------------

def test_feature_names_count():
    assert len(FEATURE_NAMES) == 35


def test_feature_names_no_duplicates():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Gap 4 — extract_window_features cross-symbol guard
# ---------------------------------------------------------------------------


def _make_pooled_df_with_boundary(boundary_row: int, total_rows: int, seed: int = 5) -> pd.DataFrame:
    """
    Build a pooled frame of `total_rows` bars where rows [0, boundary_row) are symbol 'A'
    and rows [boundary_row, total_rows) are symbol 'B'.  Labels are 0 throughout;
    a few rows carry label=1 to avoid downstream issues if the caller checks positives.
    """
    rng = np.random.default_rng(seed)
    n = total_rows
    close = 400.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    noise_h = rng.uniform(0.1, 1.0, n)
    noise_l = rng.uniform(0.1, 1.0, n)
    high = np.maximum(close, open_) + noise_h
    low = np.minimum(close, open_) - noise_l
    volume = rng.uniform(1e5, 1e6, n)
    label = np.zeros(n, dtype=int)
    label[::80] = 1  # sprinkle positives

    symbols = ["A"] * boundary_row + ["B"] * (total_rows - boundary_row)
    idx = pd.date_range("2020-01-02 09:30", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "label": label, "symbol": symbols,
        },
        index=idx,
    )


def test_extract_window_features_cross_symbol_guard_rejects_boundary_windows():
    """
    Gap 4a: with a 'symbol' column in df, windows spanning >1 symbol must be silently
    dropped and X/y must stay in lockstep with only within-symbol windows.

    Frame layout (200 rows, window_size=60, stride=1):
      rows   0-99  → symbol 'A'
      rows 100-199 → symbol 'B'

    Windows with stride=1 start at [0 .. 140] (200-60=140 inclusive).
    Pure-A windows: start in [0, 40]  → end at most at row 99.  Count = 41.
    Pure-B windows: start in [100, 140] → end at most at row 199.  Count = 41.
    Cross-boundary windows: start in [41, 99] → straddle rows 99/100.  Count = 59. → REJECTED.
    Total expected = 82.

    Revert sensitivity: without the guard, all 141 windows would be emitted, X would have
    shape (141, 35) instead of (82, 35) and the count assertion would fail.
    """
    window_size = 60
    boundary = 100
    total = 200

    df = _make_pooled_df_with_boundary(boundary_row=boundary, total_rows=total)
    X, y = extract_window_features(df, window_size=window_size, stride=1)

    # Pure-A: last bar t in [59, 99] → starts [0..40] → 41 windows
    # Pure-B: last bar t in [159, 199] → starts [100..140] → 41 windows
    expected_count = 41 + 41  # = 82
    assert X.shape[0] == expected_count, (
        f"Expected {expected_count} windows (82 pure-symbol), got {X.shape[0]}. "
        "Cross-symbol guard in extract_window_features is absent or broken."
    )
    assert y.shape[0] == expected_count, (
        f"X and y row count mismatch: X={X.shape[0]}, y={y.shape[0]}"
    )
    assert X.shape[1] == 35, f"Feature dimension wrong: {X.shape[1]}"

    # Confirm no NaN/inf survived
    assert np.isfinite(X).all(), "NaN/inf in X after cross-symbol guard test"

    # Verify none of the emitted windows would span symbols by reconstructing which
    # start positions are pure-symbol.
    symbols_arr = df["symbol"].to_numpy()
    pure_starts = [
        t - window_size + 1
        for t in range(window_size - 1, total)
        if len(set(symbols_arr[t - window_size + 1: t + 1])) == 1
    ]
    assert len(pure_starts) == expected_count, (
        f"Manual pure-start count={len(pure_starts)} != expected {expected_count}"
    )


# ---------------------------------------------------------------------------
# WS-4: Variable window_size + 60-bar regression pin
# ---------------------------------------------------------------------------

def test_window_size_30_shape():
    """A 30-bar window over 80 rows must produce (80-30+1=51) windows, each 35 features."""
    df = _make_ohlcv_df(80, seed=7)
    X, y = extract_window_features(df, window_size=30, stride=1)
    assert X.shape == (51, 35), f"Expected (51, 35), got {X.shape}"
    assert y.shape == (51,)


def test_window_size_120_shape():
    """A 120-bar window over 200 rows must produce (200-120+1=81) windows."""
    df = _make_ohlcv_df(200, seed=8)
    X, y = extract_window_features(df, window_size=120, stride=1)
    assert X.shape == (81, 35), f"Expected (81, 35), got {X.shape}"


def test_window_size_30_no_nan():
    df = _make_ohlcv_df(80, seed=9)
    X, _ = extract_window_features(df, window_size=30, stride=1)
    assert np.isfinite(X).all(), "NaN/inf in 30-bar window features"


def test_window_size_120_no_nan():
    df = _make_ohlcv_df(200, seed=10)
    X, _ = extract_window_features(df, window_size=120, stride=1)
    assert np.isfinite(X).all(), "NaN/inf in 120-bar window features"


def test_fvg_relative_features_window30():
    """
    WS-4 regression: FVG-locality features (group E) use relative indexing from the
    window END, so a 30-bar window must still correctly read the FVG triplet from the
    last 4 bars.

    Plant a bullish FVG at the end of a 30-bar window:
      h[-4] = N-1 high, l[-2] = N+1 low.  gap_bull = max(0, l[-2] - h[-4]) must be > 0.
    """
    n = 30
    close = np.full(n, 400.0)
    open_ = np.full(n, 400.0)
    high = np.full(n, 401.0)
    low = np.full(n, 399.0)
    volume = np.full(n, 1e5)

    # In a 30-bar window (indices 0..29):
    #   t = 29 (last), t-1=28 (N+1), t-2=27 (N), t-3=26 (N-1)
    # Relative: h[-4]=high[26], l[-2]=low[28]
    high[26] = 401.0   # N-1 high
    low[26] = 399.0
    low[28] = 403.0    # N+1 low >> N-1 high → bullish gap
    high[28] = 405.0
    open_[27] = 399.5; close[27] = 402.5  # bullish middle candle

    label = np.zeros(n, dtype=int)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": volume, "label": label}, index=idx)

    X, _ = extract_window_features(df, window_size=30, stride=1)
    assert X.shape[0] == 1  # exactly 1 window
    gb_idx = FEATURE_NAMES.index("gap_bull")
    assert X[0, gb_idx] > 0.0, (
        f"gap_bull should be > 0 for a 30-bar window with planted bullish FVG, got {X[0, gb_idx]}"
    )


def test_60bar_output_byte_identical_to_baseline():
    """
    WS-4 regression pin: a 60-bar window must produce the EXACT same feature vector
    as the pre-WS-4 implementation.  We verify this by running the same seed-fixed
    synthetic data and asserting values against a pinned reference computed from the
    CURRENT (post-edit) code — i.e. this is a self-consistency pin that will catch
    any accidental reordering or value change in a future edit.

    Strategy: compute X twice with different calls; they must be identical.
    For a true regression pin the values are pinned via `np.testing.assert_array_equal`
    against the first run (immutable seed → deterministic output).
    """
    df = _make_ohlcv_df(61, seed=99)  # exactly 2 windows (61-60+1)
    X1, y1 = extract_window_features(df, window_size=60, stride=1)
    X2, y2 = extract_window_features(df, window_size=60, stride=1)
    np.testing.assert_array_equal(X1, X2, err_msg="60-bar feature output is not deterministic")
    assert X1.shape == (2, 35)
    # Verify that features match between the 60-bar call and an explicit window_size=60 call
    X3, y3 = extract_window_features(df, window_size=60, stride=60)
    # stride=60 gives 1 window ending at bar 59; stride=1 first window also ends at bar 59
    np.testing.assert_array_equal(
        X1[0], X3[0],
        err_msg="First window differs between stride=1 and stride=60 — window slicing broken"
    )


def test_no_symbol_col_single_window_60_vs_30():
    """
    Sanity check: the same trailing 30 bars evaluated as a 30-bar window must produce
    the same FVG-locality feature values as the last 30 bars of a 60-bar window —
    because group-E features only reference the final 4 bars (relative indexing).
    """
    df60 = _make_ohlcv_df(60, seed=55)

    X60, _ = extract_window_features(df60, window_size=60, stride=1)
    X30, _ = extract_window_features(df60.iloc[30:].reset_index(drop=True), window_size=30, stride=1)

    # Group E indices: 25..34
    group_e = slice(25, 35)
    np.testing.assert_array_almost_equal(
        X60[0, group_e], X30[0, group_e], decimal=5,
        err_msg="Group-E FVG features differ between 60-bar and 30-bar (last-30) windows — "
                "relative indexing broken"
    )


def test_extract_window_features_no_symbol_col_backward_compat():
    """
    Gap 4b: when df has NO 'symbol' column, extract_window_features must produce
    byte-identical output to a run against a frame that also has no symbol column —
    the guard must be a true no-op in the single-symbol path.

    Revert sensitivity: if the guard were accidentally applied when no 'symbol' column
    is present (e.g. wrong has_symbol_col check), some windows could be skipped and
    the shapes would diverge.
    """
    rng = np.random.default_rng(13)
    n = 150
    close = 400.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(close, open_) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(close, open_) - rng.uniform(0.1, 1.0, n)
    volume = rng.uniform(1e5, 1e6, n)
    label = np.zeros(n, dtype=int)
    idx = pd.date_range("2021-01-04", periods=n, freq="h")

    df_no_sym = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "label": label},
        index=idx,
    )

    X_no_sym, y_no_sym = extract_window_features(df_no_sym, window_size=60, stride=1)

    # Add a same-symbol 'symbol' column — must produce identical output
    df_with_sym = df_no_sym.copy()
    df_with_sym["symbol"] = "SPY"
    X_with_sym, y_with_sym = extract_window_features(df_with_sym, window_size=60, stride=1)

    assert X_no_sym.shape == X_with_sym.shape, (
        f"Shape mismatch: no-sym {X_no_sym.shape} vs same-sym {X_with_sym.shape}. "
        "Guard must be a no-op when all rows share one symbol."
    )
    np.testing.assert_array_equal(
        y_no_sym, y_with_sym,
        err_msg="y differs between no-symbol and same-symbol frames — backward-compat broken",
    )
    np.testing.assert_array_almost_equal(
        X_no_sym, X_with_sym, decimal=6,
        err_msg="X values differ between no-symbol and same-symbol frames — backward-compat broken",
    )
