"""runner.py — Runs inference across all loaded adapters on a data slice."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.labels.base import BaseLabeller
from src.data.normalize import normalise_window
from src.data.timeframe import H1, Timeframe
from src.data.window import _has_session_gap
from src.inspect.base import ModelAdapter

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
_REQUIRED_COLS = _OHLCV_COLS + ["label"]


@dataclass
class InspectionResults:
    """Container for all runner outputs.

    Attributes
    ----------
    windows_raw : np.ndarray, shape (N, 60, 5)
        Un-normalised OHLCV windows used for visualisation and XGBoost features.
    windows_norm : np.ndarray, shape (N, 60, 5)
        Normalised windows (same normalisation as training) used by DL adapters.
    labels : np.ndarray, shape (N,)
        Encoded ground-truth labels {0=none, 1=bullish, 2=bearish}.
    timestamps : pd.DatetimeIndex, length N
        Timestamp of each window's LAST bar.
    probas : dict[str, np.ndarray]
        model_name -> (N, 3) float32 class probabilities.
    preds : dict[str, np.ndarray]
        model_name -> (N,) int class predictions (argmax of probas).
    """

    windows_raw: np.ndarray
    windows_norm: np.ndarray
    labels: np.ndarray
    timestamps: pd.DatetimeIndex
    probas: dict[str, np.ndarray] = field(default_factory=dict)
    preds: dict[str, np.ndarray] = field(default_factory=dict)
    future_ohlcv: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 5), dtype=np.float32)
    )
    future_timestamps: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def model_names(self) -> list[str]:
        return list(self.probas.keys())


def run(
    adapters: list[ModelAdapter],
    df_slice: pd.DataFrame,
    labeller: BaseLabeller,
    window_size: int = 60,
    stride: int = 1,
    drop_cross_session: bool = False,
    lookahead_bars: int = 0,
    timeframe: Timeframe = H1,
) -> InspectionResults:
    """Run inference across all adapters on ``df_slice``.

    Parameters
    ----------
    adapters : list[ModelAdapter]
        Loaded and initialised adapter instances.
    df_slice : pd.DataFrame
        Must have columns [open, high, low, close, volume, label] and a DatetimeIndex.
        ``label`` must already be encoded (0/1/2).
    labeller : BaseLabeller
        Used for class metadata; label encoding must have already been applied to
        ``df_slice['label']`` before calling this function.
    window_size : int
        Number of bars per window. Default 60.
    stride : int
        Sliding step. Default 1.
    drop_cross_session : bool
        Skip windows that span session gaps (>90 min between consecutive bars).
        Default False — matches training configuration (SPY H1 60-bar windows
        always span overnight gaps; dropping them yields 0 windows).
    timeframe : Timeframe
        The bar timeframe being processed.  Controls the intra-window gap
        threshold used by ``_has_session_gap`` when ``drop_cross_session`` is
        True.  Defaults to ``H1`` (90 min gap threshold) — identical to the
        previous hard-coded behaviour.

    Returns
    -------
    InspectionResults
        N=0 if the slice is too short or no valid windows are found.
    """
    # Validate columns
    missing = [c for c in _REQUIRED_COLS if c not in df_slice.columns]
    if missing:
        raise ValueError(f"df_slice missing required columns: {missing}")

    n_bars = len(df_slice)
    if n_bars < window_size:
        return _empty_results()

    ohlcv = df_slice[_OHLCV_COLS].to_numpy(dtype=np.float64)
    labels_arr = df_slice["label"].to_numpy(dtype=np.int64)
    timestamps = df_slice.index

    raw_windows: list[np.ndarray] = []
    norm_windows: list[np.ndarray] = []
    window_labels: list[int] = []
    window_timestamps: list[pd.Timestamp] = []
    future_windows: list[np.ndarray] = []
    future_ts_per_window: list[list] = []

    nan_future = np.full((lookahead_bars, 5), np.nan, dtype=np.float32) if lookahead_bars > 0 else None

    for i in range(0, n_bars - window_size + 1, stride):
        end = i + window_size
        if end > n_bars:
            break
        if drop_cross_session and _has_session_gap(
            df_slice, i, end, timeframe.max_intra_window_gap_minutes
        ):
            continue

        raw_w = ohlcv[i:end].astype(np.float32)       # (60, 5)
        norm_w = normalise_window(ohlcv[i:end])         # (60, 5) float32
        label = int(labels_arr[end - 1])
        ts = timestamps[end - 1]

        raw_windows.append(raw_w)
        norm_windows.append(norm_w)
        window_labels.append(label)
        window_timestamps.append(ts)

        if lookahead_bars > 0:
            future_end = end + lookahead_bars
            if future_end <= n_bars:
                future_w = ohlcv[end:future_end].astype(np.float32)
                future_ts = list(timestamps[end:future_end])
            else:
                # Partial / missing future — pad with NaN
                avail = ohlcv[end:n_bars].astype(np.float32)
                future_w = nan_future.copy()
                if len(avail) > 0:
                    future_w[: len(avail)] = avail
                future_ts = list(timestamps[end:n_bars]) + [pd.NaT] * (lookahead_bars - (n_bars - end))
            future_windows.append(future_w)
            future_ts_per_window.append(future_ts)

    if not raw_windows:
        return _empty_results()

    windows_raw = np.stack(raw_windows, axis=0)     # (N, 60, 5)
    windows_norm = np.stack(norm_windows, axis=0)   # (N, 60, 5)
    labels_np = np.array(window_labels, dtype=np.int64)
    ts_index = pd.DatetimeIndex(window_timestamps)

    if lookahead_bars > 0 and future_windows:
        future_arr = np.stack(future_windows, axis=0)
    else:
        future_arr = np.empty((len(raw_windows), 0, 5), dtype=np.float32)

    results = InspectionResults(
        windows_raw=windows_raw,
        windows_norm=windows_norm,
        labels=labels_np,
        timestamps=ts_index,
        future_ohlcv=future_arr,
        future_timestamps=future_ts_per_window,
    )

    for adapter in adapters:
        # XGBoost adapters expect RAW windows; DL adapters expect NORMALISED.
        # We pass raw to everyone and let adapters decide which to use.
        # For DL (LSTM etc.) windows_norm is the correct input.
        # The adapter name "xgboost" signals raw windows; all others get normalised.
        if adapter.name == "xgboost":
            input_windows = windows_raw
        else:
            input_windows = windows_norm

        probas = adapter.predict_proba(input_windows)

        if probas.shape != (results.n, 3):
            raise ValueError(
                f"Adapter '{adapter.name}' returned shape {probas.shape}, "
                f"expected ({results.n}, 3)."
            )

        preds = np.argmax(probas, axis=1).astype(np.int64)
        results.probas[adapter.name] = probas
        results.preds[adapter.name] = preds

    return results


def _empty_results() -> InspectionResults:
    return InspectionResults(
        windows_raw=np.empty((0, 60, 5), dtype=np.float32),
        windows_norm=np.empty((0, 60, 5), dtype=np.float32),
        labels=np.empty(0, dtype=np.int64),
        timestamps=pd.DatetimeIndex([]),
        future_ohlcv=np.empty((0, 0, 5), dtype=np.float32),
        future_timestamps=[],
    )
