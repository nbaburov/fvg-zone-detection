"""window_features.py — Tabular feature engineering for XGBoost baseline.

Receives a split DataFrame (open, high, low, close, volume, label) and emits
a 2D feature matrix (n_windows, 35) + label vector (n_windows,).

All 35 features are strictly causal: only bars [t-59 .. t] are referenced.
Label bar = t (the window's last bar).

Feature groups:
  A: Price return features         (8)
  B: Volatility features           (5)
  C: Momentum / trend features     (7)
  D: Volume features               (5)
  E: FVG-locality features        (10)
  Total: 35
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    # Group A
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "high_low_range", "body_ratio", "upper_wick",
    # Group B
    "atr_14", "atr_28", "vol_ratio_14_28", "range_60", "range_20",
    # Group C
    "rsi_14", "macd_signal", "macd_norm", "pos_in_range",
    "trend_slope", "above_ma20", "above_ma50",
    # Group D
    "vol_zscore_5", "vol_zscore_20", "vol_trend", "vol_spike", "vol_body_corr",
    # Group E
    "gap_bull", "gap_bear", "gap_norm_bull", "gap_norm_bear",
    "mid_body_bull", "mid_body_bear", "mid_range_norm",
    "react_in_gap_bull", "react_in_gap_bear", "prior_trend_5",
]

assert len(FEATURE_NAMES) == 35, f"Expected 35 features, got {len(FEATURE_NAMES)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_window_features(
    df: pd.DataFrame,
    window_size: int = 60,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of `window_size` bars over `df` with the given stride.

    Parameters
    ----------
    df : DataFrame with columns [open, high, low, close, volume, label].
    window_size : int, default 60.
    stride : int, default 1.

    Returns
    -------
    X : ndarray shape (n_windows, 35) float32 — no NaN.
    y : ndarray shape (n_windows,) int — label values at window end bar.
    """
    if len(df) < window_size:
        return np.empty((0, 35), dtype=np.float32), np.empty(0, dtype=int)

    # Pre-extract numpy arrays for speed
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)
    lbl = df["label"].to_numpy(dtype=int)

    # Cross-symbol guard: mirrors src/data/window.py::_window_generator.
    # When df has a 'symbol' column (pooled multi-symbol frame), skip any window
    # whose bars span >1 symbol — mixing two unrelated price series is silent
    # contamination.  Single-symbol DataFrames without the column are entirely
    # unaffected and follow the identical code path as before.
    has_symbol_col: bool = "symbol" in df.columns
    symbols: np.ndarray | None = df["symbol"].to_numpy() if has_symbol_col else None

    # Window endpoints: t from window_size-1 to len(df)-1, step=stride
    t_indices = list(range(window_size - 1, len(df), stride))

    # Pre-allocate for worst case; trim to actual count after the loop.
    n_max = len(t_indices)
    X = np.zeros((n_max, 35), dtype=np.float32)
    y = np.zeros(n_max, dtype=int)
    out_idx: int = 0

    for t in t_indices:
        start = t - window_size + 1  # inclusive

        # Skip windows that cross a symbol boundary (multi-symbol path only).
        if has_symbol_col and len(set(symbols[start: t + 1])) > 1:  # type: ignore[index]
            continue

        # Slices for the full window
        c_w = c[start: t + 1]   # shape (60,)
        h_w = h[start: t + 1]
        l_w = l[start: t + 1]
        o_w = o[start: t + 1]
        v_w = v[start: t + 1]

        feats = _compute_features(c_w, h_w, l_w, o_w, v_w)
        X[out_idx] = feats
        y[out_idx] = lbl[t]
        out_idx += 1

    # Trim to actual number of emitted windows.
    X = X[:out_idx]
    y = y[:out_idx]

    # NaN guard: fill any NaN/inf with 0
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    assert np.isfinite(X).all(), "NaN/inf survived nan_to_num — internal error"
    return X, y


# ---------------------------------------------------------------------------
# Per-window feature computation (all inputs are 1D numpy arrays of length 60)
# ---------------------------------------------------------------------------

def _compute_features(
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    o: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Compute all 35 features from a single 60-bar window. Returns float32[35]."""
    feats = np.empty(35, dtype=np.float32)
    ct = c[-1]  # label bar close
    eps = 1e-9

    # ------------------------------------------------------------------
    # Group A: Price return features (indices 0..7)
    # ------------------------------------------------------------------
    feats[0] = (ct - c[-2]) / (c[-2] + eps)                      # ret_1
    feats[1] = (ct - c[-6]) / (c[-6] + eps)                      # ret_5
    feats[2] = (ct - c[-11]) / (c[-11] + eps)                    # ret_10
    feats[3] = (ct - c[-21]) / (c[-21] + eps)                    # ret_20
    feats[4] = (ct - c[0]) / (c[0] + eps)                        # ret_60
    hl = h[-1] - l[-1]
    feats[5] = hl / (ct + eps)                                    # high_low_range
    feats[6] = abs(ct - o[-1]) / (hl + eps)                      # body_ratio
    feats[7] = (h[-1] - max(o[-1], ct)) / (hl + eps)             # upper_wick

    # ------------------------------------------------------------------
    # Group B: Volatility features (indices 8..12)
    # ------------------------------------------------------------------
    abs_diff = np.abs(np.diff(c))   # shape (59,)
    atr14 = float(abs_diff[-14:].mean()) if len(abs_diff) >= 14 else float(abs_diff.mean())
    atr28 = float(abs_diff[-28:].mean()) if len(abs_diff) >= 28 else float(abs_diff.mean())
    feats[8] = atr14                                              # atr_14
    feats[9] = atr28                                              # atr_28
    feats[10] = atr14 / (atr28 + eps)                            # vol_ratio_14_28
    range60 = (h.max() - l.min()) / (ct + eps)
    range20 = (h[-20:].max() - l[-20:].min()) / (ct + eps)
    feats[11] = range60                                           # range_60
    feats[12] = range20                                           # range_20

    # ------------------------------------------------------------------
    # Group C: Momentum / trend features (indices 13..19)
    # ------------------------------------------------------------------
    feats[13] = _rsi_14(c)                                        # rsi_14
    macd_val = _macd(c)
    feats[14] = macd_val                                          # macd_signal
    feats[15] = macd_val / (ct + eps)                             # macd_norm
    c_min = l.min()
    c_max = h.max()
    rng_abs = c_max - c_min
    feats[16] = (ct - c_min) / (rng_abs + eps)                   # pos_in_range
    feats[17] = _trend_slope(c)                                   # trend_slope
    feats[18] = 1.0 if ct > c[-20:].mean() else 0.0              # above_ma20
    feats[19] = 1.0 if ct > c[-50:].mean() else 0.0              # above_ma50

    # ------------------------------------------------------------------
    # Group D: Volume features (indices 20..24)
    # ------------------------------------------------------------------
    v_mean5 = v[-5:].mean()
    v_std5 = v[-5:].std()
    feats[20] = (v[-1] - v_mean5) / (v_std5 + 1.0)              # vol_zscore_5
    v_mean20 = v[-20:].mean()
    v_std20 = v[-20:].std()
    feats[21] = (v[-1] - v_mean20) / (v_std20 + 1.0)            # vol_zscore_20
    prev5_mean = v[-10:-5].mean()
    feats[22] = v_mean5 / (prev5_mean + 1.0)                     # vol_trend
    feats[23] = 1.0 if v[-1] > 2.0 * v_mean20 else 0.0          # vol_spike
    feats[24] = (v[-1] * abs(ct - o[-1])) / (v[-1] + 1.0)       # vol_body_corr

    # ------------------------------------------------------------------
    # Group E: FVG-locality features (indices 25..34)
    # In a 60-bar window ending at t=59 (0-indexed):
    #   label bar = t = index 59 (= N+2)
    #   N+1 bar = index 58 = t-1
    #   N   bar = index 57 = t-2  (middle candle)
    #   N-1 bar = index 56 = t-3
    # ------------------------------------------------------------------
    h_n1 = h[-4]    # N-1 high (bar t-3)
    l_n1 = l[-4]    # N-1 low
    h_np1 = h[-2]   # N+1 high (bar t-1)
    l_np1 = l[-2]   # N+1 low
    o_n = o[-3]     # N open (bar t-2, middle)
    c_n = c[-3]     # N close
    h_n = h[-3]     # N high
    l_n = l[-3]     # N low

    gap_bull = max(0.0, l_np1 - h_n1)
    gap_bear = max(0.0, l_n1 - h_np1)
    feats[25] = gap_bull                                          # gap_bull
    feats[26] = gap_bear                                          # gap_bear
    feats[27] = gap_bull / (ct + eps)                            # gap_norm_bull
    feats[28] = gap_bear / (ct + eps)                            # gap_norm_bear
    feats[29] = 1.0 if c_n > o_n else 0.0                       # mid_body_bull
    feats[30] = 1.0 if c_n < o_n else 0.0                       # mid_body_bear
    feats[31] = (h_n - l_n) / (ct + eps)                        # mid_range_norm

    # react_in_gap_bull: reaction candle (bar t) close sits inside bullish gap zone [h_n1, l_np1]
    if gap_bull > 0 and h_n1 <= ct <= l_np1:
        feats[32] = 1.0
    else:
        feats[32] = 0.0
    # react_in_gap_bear: reaction candle close sits inside bearish gap
    if gap_bear > 0 and h_np1 <= ct <= l_n1:
        feats[33] = 1.0
    else:
        feats[33] = 0.0

    # prior_trend_5: mean(c[t-7..t-3]) - mean(c[t-12..t-8]) normalised
    pre_pattern = c[-8:-3].mean()   # bars t-7..t-3
    pre_pre = c[-13:-8].mean()      # bars t-12..t-8
    feats[34] = (pre_pattern - pre_pre) / (ct + eps)             # prior_trend_5

    return feats


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _rsi_14(close_window: np.ndarray) -> float:
    """Wilder RSI on last 14 bars (15 prices → 14 diffs)."""
    prices = close_window[-15:]
    if len(prices) < 2:
        return 50.0
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _macd(close_window: np.ndarray) -> float:
    """EMA(12) - EMA(26) on close_window (length 60)."""
    s = pd.Series(close_window)
    ema12 = float(s.ewm(span=12, adjust=False).mean().iloc[-1])
    ema26 = float(s.ewm(span=26, adjust=False).mean().iloc[-1])
    return ema12 - ema26


def _trend_slope(close_window: np.ndarray) -> float:
    """Linear slope of close_window, normalised by last close."""
    x = np.arange(len(close_window), dtype=float)
    slope = np.polyfit(x, close_window, 1)[0]
    return float(slope / (close_window[-1] + 1e-9))
