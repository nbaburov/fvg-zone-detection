"""predict.py — Build per-model tracks for the demo animator.

Thin reuse layer: loads each adapter via the registry, runs inference over
the full spy_h1_test.parquet (warmup automatic), computes trade outcomes,
then slices outputs to the display window.

Design: mirrors scripts/inspect_models.py lines 697-700.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Data contracts
# --------------------------------------------------------------------------

class ModelSpec(NamedTuple):
    """Spec for a single model to load."""
    arch: str           # "lstm", "cnn_lstm", "transformer", "xgboost"
    tf: str             # "h1"
    dataset: str        # "multisym"
    tuned: bool         # False for all H1 multisym
    label: str          # Display label, e.g. "XGBoost · H1 · multisym · F1 0.738"


@dataclass(frozen=True)
class ModelTrack:
    """All per-model data needed for serialization."""
    name: str                     # arch name, e.g. "xgboost"
    label: str                    # display label
    bars: pd.DataFrame            # OHLCV slice for the display window
    pred_ts: pd.DatetimeIndex     # window last-bar timestamp per prediction (ALL windows)
    preds: np.ndarray             # (N,) int 0/1/2, ALL windows
    probas: np.ndarray            # (N,3), ALL windows
    trades: list                  # Trade objects (positive preds only, with outcomes)
    windows_raw_all: np.ndarray   # (N,60,5) ALL windows (for gap geometry)
    # Note: window_ts_all was removed — pred_ts is the single source of truth
    # for all-window timestamps. serialize.py reads track.pred_ts directly.


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_DATA_PATH = _REPO_ROOT / "data" / "processed" / "spy_h1_test.parquet"
_CKPT_ROOT = _REPO_ROOT / "checkpoints"

# Default lookahead: 20 bars (≈4 trading days on H1) to resolve outcomes
_LOOKAHEAD_BARS = 20


def build_model_tracks(
    window_start: str,
    window_end: str,
    model_specs: list[ModelSpec],
    data_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    lookahead_bars: int = _LOOKAHEAD_BARS,
) -> list[ModelTrack]:
    """Load adapters, run inference, compute trades, slice to display window.

    Parameters
    ----------
    window_start : str
        ISO timestamp string for the first bar of the display window.
    window_end : str
        ISO timestamp string for the last bar of the display window (inclusive).
    model_specs : list[ModelSpec]
        One entry per model to load and run.
    data_path : Path | None
        Override for the SPY H1 test parquet path.
    checkpoint_dir : Path | None
        Override for checkpoint root directory.
    lookahead_bars : int
        How many bars after each window to include for outcome resolution.

    Returns
    -------
    list[ModelTrack]
        One ModelTrack per spec, in the same order as model_specs.
    """
    from src.data.labels.valid_fvg import ValidFVGLabeller
    from src.data.timeframe import H1
    from src.inspect.multisym import default_checkpoint_path
    from src.inspect.outcomes import compute_trades_for_model
    from src.inspect.registry import load_adapters
    from src.inspect.runner import run

    dp = data_path or _DATA_PATH
    ckpt_root = checkpoint_dir or _CKPT_ROOT

    # Load the SPY H1 test parquet (full, so runner gets warmup)
    df = pd.read_parquet(dp)

    # Single labeller instance — hoisted here to avoid duplicate construction (Fix E)
    labeller = ValidFVGLabeller()

    # Ensure label column is present
    if "label" not in df.columns:
        raw_labels = labeller.label(df)
        df["label"] = labeller.encode(raw_labels)

    # Resolve checkpoint paths (seed-0 for each arch)
    checkpoint_paths: dict[str, str] = {}
    for spec in model_specs:
        p = default_checkpoint_path(
            spec.arch, ckpt_root, token=spec.tf, dataset=spec.dataset, tuned=spec.tuned
        )
        if p is not None:
            checkpoint_paths[spec.arch] = str(p)

    arch_names = [spec.arch for spec in model_specs]
    adapters = load_adapters(arch_names, ckpt_root, checkpoint_paths=checkpoint_paths)

    # Run inference over full parquet (warmup is automatic).
    # stride=1 is explicit: serialize.py x-span derivation assumes stride=1
    # (consecutive windows, bar_56 = window_idx-3, bar_58 = window_idx-1).
    results = run(
        adapters=adapters,
        df_slice=df,
        labeller=labeller,
        lookahead_bars=lookahead_bars,
        timeframe=H1,
        stride=1,  # Fix F: make explicit; runner default is already 1
    )

    # Parse window bounds (timezone-aware).
    # Fix H: tz_convert crashes on a tz-naive timestamp (e.g. bare ISO string
    # from --start without offset).  Guard: localize if naive, convert if aware.
    tz = df.index.tz
    _ts_start = pd.Timestamp(window_start)
    ts_start = _ts_start.tz_localize(tz) if _ts_start.tzinfo is None else _ts_start.tz_convert(tz)
    _ts_end = pd.Timestamp(window_end)
    ts_end = _ts_end.tz_localize(tz) if _ts_end.tzinfo is None else _ts_end.tz_convert(tz)

    # The display window bars (raw OHLCV)
    bars_df = df.loc[ts_start:ts_end, ["open", "high", "low", "close", "volume"]]

    tracks: list[ModelTrack] = []
    for spec, adapter in zip(model_specs, adapters):
        arch = spec.arch
        preds_all = results.preds[arch]
        probas_all = results.probas[arch]
        ts_all = results.timestamps  # last-bar timestamp per window

        # Compute trades (positive preds) over ALL windows + outcomes
        trades = compute_trades_for_model(
            results.windows_raw,
            results.future_ohlcv,
            results.future_timestamps,
            preds_all,
            probas=probas_all,
        )

        tracks.append(ModelTrack(
            name=arch,
            label=spec.label,
            bars=bars_df.copy(),
            pred_ts=ts_all,
            preds=preds_all,
            probas=probas_all,
            trades=trades,
            windows_raw_all=results.windows_raw,
            # window_ts_all removed — pred_ts is the single source of truth (Fix E)
        ))

    return tracks
