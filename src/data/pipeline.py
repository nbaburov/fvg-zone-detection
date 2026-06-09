"""Top-level pipeline: download → label → split → window → class weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.download import download_h1
from src.data.labels import LABELLERS
from src.data.process import build_labelled_dataset
from src.data.split import SPLIT_BOUNDARIES, temporal_split
from src.data.window import SMCWindowDataset

logger = logging.getLogger(__name__)

# Default output directory — tests can patch this
PROCESSED_DIR = "data/processed"
MULTISYM_DIR = "data/processed/multisym"


def _compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = 3,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from train split only.
    Formula: num_classes / (class_count * total_windows)
    Normalised so weights sum to num_classes.

    Returns: Tensor shape (num_classes,), dtype float32.
    """
    labels = train_df["label"].to_numpy()
    total = len(labels)

    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.where(counts > 0, counts, 1.0)  # default weight for unseen class
    weights = total / (counts * num_classes)

    # Normalise so weights sum to num_classes
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights * (num_classes / weight_sum)

    return torch.tensor(weights, dtype=torch.float32)


def build_pipeline(
    labeller_name: str = "fvg_valid",
    window_size: int = 60,
    start: str = "2018-01-01",
    end: str | None = "2025-12-31",
    use_cache: bool = True,
) -> tuple[SMCWindowDataset, SMCWindowDataset, SMCWindowDataset, torch.Tensor]:
    """
    Full pipeline: download → label → split → window → return datasets + class weights.

    Returns (train_dataset, val_dataset, test_dataset, class_weights).
    class_weights: Tensor of shape (num_classes,), dtype float32.
                   Computed on train split only.
                   Persisted to {PROCESSED_DIR}/class_weights_{labeller_name}.json
                   (and legacy alias {PROCESSED_DIR}/class_weights.json).
    train_dataset stride = 1.
    val_dataset stride = 60.
    test_dataset stride = 60.
    Raises: ValueError if any dataset has 0 positive-class windows.
    """
    out_dir = Path(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if labeller_name not in LABELLERS:
        raise KeyError(f"Unknown labeller: '{labeller_name}'. Registered: {list(LABELLERS)}")

    labeller_cls = LABELLERS[labeller_name]
    labeller = labeller_cls()

    # Step 1: Get full labelled H1 DataFrame
    full_df = build_labelled_dataset(
        labeller_name=labeller_name,
        start=start,
        end=end,
        use_cache=use_cache,
    )

    # Persist full dataset
    full_path = out_dir / "spy_h1.parquet"
    full_df.to_parquet(full_path)
    logger.info("Full labelled H1 written to %s (%d rows)", full_path, len(full_df))

    # Step 2: Temporal split — capture boundaries explicitly so sidecar and split always agree
    boundaries = SPLIT_BOUNDARIES
    train_df, val_df, test_df = temporal_split(full_df, boundaries=boundaries)

    # Persist splits
    for name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out_dir / f"spy_h1_{name}.parquet"
        df_split.to_parquet(path)
        logger.info("Split '%s' written to %s (%d rows)", name, path, len(df_split))

    # Write dataset_meta sidecar — self-documents the provenance of every artifact in this dir
    meta = {
        "labeller_name": labeller_name,
        "window_size": window_size,
        "start": start,
        "end": end,
        "split_boundaries": boundaries,
        "row_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    meta_path = out_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Dataset meta sidecar written to %s", meta_path)

    # Step 3: Class weights from train split only
    num_classes = labeller.num_classes
    class_weights = _compute_class_weights(train_df, num_classes=num_classes)

    # Persist class weights — write both a labeller-suffixed file and the legacy alias.
    # Suffixed file: class_weights_{labeller_name}.json (canonical name for new consumers).
    # Legacy alias:  class_weights.json (backwards compat — existing checkpoint .meta.json
    #                and training scripts that hardcode this name continue to work).
    weights_dict = {str(i): float(class_weights[i]) for i in range(num_classes)}
    suffixed_path = out_dir / f"class_weights_{labeller_name}.json"
    legacy_path = out_dir / "class_weights.json"
    for weights_path in (suffixed_path, legacy_path):
        with open(weights_path, "w") as f:
            json.dump(weights_dict, f, indent=2)
    logger.info(
        "Class weights written to %s (and legacy alias %s): %s",
        suffixed_path, legacy_path, weights_dict,
    )

    # Step 4: Build window datasets
    # Note: drop_cross_session_windows=False — RTH H1 = 7 bars/day, every window spans
    # multiple overnight gaps (~18h each) and weekends. These are expected, not exceptional.
    # FVG patterns across overnight gaps are structurally meaningful in SMC theory.
    train_ds = SMCWindowDataset(train_df, labeller, stride=1, window_size=window_size, drop_cross_session_windows=False)
    val_ds = SMCWindowDataset(val_df, labeller, stride=window_size, window_size=window_size, drop_cross_session_windows=False)
    test_ds = SMCWindowDataset(test_df, labeller, stride=window_size, window_size=window_size, drop_cross_session_windows=False)

    logger.info(
        "Datasets: train=%d windows, val=%d windows, test=%d windows",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # Validate: each dataset must have at least one positive-class window
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = ds.label_counts
        n_positive = sum(v for k, v in counts.items() if k != 0)
        if n_positive == 0:
            raise ValueError(
                f"Dataset '{name}' has 0 positive-class windows. "
                "Check the FVG detector and split boundaries."
            )

    return train_ds, val_ds, test_ds, class_weights


def build_multi_symbol_pipeline(
    symbols: list[str],
    labeller_name: str = "fvg_valid",
    window_size: int = 60,
    start: str = "2018-01-01",
    end: str | None = "2025-12-31",
    use_cache: bool = True,
) -> tuple[SMCWindowDataset, SMCWindowDataset, SMCWindowDataset, torch.Tensor]:
    """
    Multi-symbol pipeline: per-symbol download → label → split, then pool across
    symbols → class weights → window datasets.

    Split is performed **per symbol before any concatenation** so that temporal
    boundaries are applied independently within each symbol's time series.  A
    ``symbol`` column is added to each per-symbol split DataFrame before pooling
    so that ``SMCWindowDataset`` / ``_window_generator`` can reject windows that
    span a symbol boundary.

    Outputs are written to ``data/processed/multisym/`` — the SPY-only parquets
    in ``data/processed/`` are never touched.

    Output parquet filenames mirror the single-symbol pipeline
    (``spy_h1_{train,val,test}.parquet``) so that ``multiseed_run.py`` can point
    at the multisym directory via ``--data-dir`` / ``data.data_dir`` without any
    naming changes.

    Returns (train_dataset, val_dataset, test_dataset, class_weights).
    class_weights: Tensor shape (num_classes,), dtype float32.
                   Computed on the pooled train split only.
    train_dataset stride = 1.
    val_dataset stride = window_size.
    test_dataset stride = window_size.
    Raises: ValueError if any dataset has 0 positive-class windows.
    """
    if not symbols:
        raise ValueError("symbols must be a non-empty list.")

    if labeller_name not in LABELLERS:
        raise KeyError(f"Unknown labeller: '{labeller_name}'. Registered: {list(LABELLERS)}")

    out_dir = Path(MULTISYM_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeller_cls = LABELLERS[labeller_name]
    labeller = labeller_cls()
    boundaries = SPLIT_BOUNDARIES

    per_symbol_train: list[pd.DataFrame] = []
    per_symbol_val: list[pd.DataFrame] = []
    per_symbol_test: list[pd.DataFrame] = []
    per_symbol_row_counts: dict[str, dict[str, int]] = {}

    for symbol in symbols:
        logger.info("Processing symbol: %s", symbol)

        # Step 1: Download H1 bars for this symbol
        h1_df = download_h1(symbol=symbol, start=start, end=end, use_cache=use_cache)

        # Step 2: Label
        raw_labels = labeller.label(h1_df)
        encoded_labels = labeller.encode(raw_labels)
        labelled_df = h1_df.copy()
        labelled_df["raw_label"] = raw_labels
        labelled_df["label"] = encoded_labels

        # Step 3: Temporal split — per symbol, before any concatenation
        train_df, val_df, test_df = temporal_split(labelled_df, boundaries=boundaries)

        # Step 4: Tag each split with the symbol identity BEFORE pooling.
        # The window guard in _window_generator reads this column to reject
        # windows spanning more than one symbol's contiguous bars.
        for df_split in (train_df, val_df, test_df):
            df_split["symbol"] = symbol

        per_symbol_row_counts[symbol] = {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        }
        logger.info(
            "Symbol %s — train=%d, val=%d, test=%d rows",
            symbol, len(train_df), len(val_df), len(test_df),
        )

        # Warn (do not raise) when a symbol contributes too few rows to produce
        # even a single window.  Zero-row splits get a harder error; low-but-
        # nonzero rows just get flagged so illiquid-symbol edge cases don't abort.
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            n_rows = len(split_df)
            if n_rows == 0:
                raise ValueError(
                    f"Symbol '{symbol}' has 0 rows in split '{split_name}'. "
                    "Check the date boundaries and download for this symbol."
                )
            if n_rows < window_size:
                logger.warning(
                    "Symbol '%s' has only %d rows in split '%s' (window_size=%d) — "
                    "no windows can be produced for this symbol×split combination.",
                    symbol, n_rows, split_name, window_size,
                )

        per_symbol_train.append(train_df)
        per_symbol_val.append(val_df)
        per_symbol_test.append(test_df)

    # Step 5: Concatenate per split.  Sort by (symbol, timestamp) so each symbol's
    # DatetimeIndex — sort_values on index requires reset/set or use of index name.
    # Sorts so same-symbol bars are contiguous and chronologically ordered —
    # required by the window guard's contiguity assumption.
    def _pool_timeseries(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        pooled = pd.concat(dfs, axis=0)
        # Sort so same-symbol bars are contiguous and chronologically ordered
        pooled = pooled.assign(_ts=pooled.index).sort_values(["symbol", "_ts"]).drop(columns="_ts")
        return pooled

    pooled_train = _pool_timeseries(per_symbol_train)
    pooled_val = _pool_timeseries(per_symbol_val)
    pooled_test = _pool_timeseries(per_symbol_test)

    logger.info(
        "Pooled splits — train=%d, val=%d, test=%d rows",
        len(pooled_train), len(pooled_val), len(pooled_test),
    )

    # Step 6: Persist pooled splits.  Filenames match the single-symbol pipeline
    # so multiseed_run.py can point at this directory without source changes.
    for split_name, df_split in [("train", pooled_train), ("val", pooled_val), ("test", pooled_test)]:
        path = out_dir / f"spy_h1_{split_name}.parquet"
        df_split.to_parquet(path)
        logger.info("Multisym split '%s' written to %s (%d rows)", split_name, path, len(df_split))

    # Step 7: Class weights from pooled train split only
    num_classes = labeller.num_classes
    class_weights = _compute_class_weights(pooled_train, num_classes=num_classes)

    weights_dict = {str(i): float(class_weights[i]) for i in range(num_classes)}
    suffixed_path = out_dir / f"class_weights_{labeller_name}.json"
    legacy_path = out_dir / "class_weights.json"
    for weights_path in (suffixed_path, legacy_path):
        with open(weights_path, "w") as f:
            json.dump(weights_dict, f, indent=2)
    logger.info(
        "Multisym class weights written to %s (and alias %s): %s",
        suffixed_path, legacy_path, weights_dict,
    )

    # Step 8: Dataset meta sidecar
    meta = {
        "labeller_name": labeller_name,
        "window_size": window_size,
        "start": start,
        "end": end,
        "symbols": symbols,
        "split_boundaries": boundaries,
        "row_counts": {
            "train": len(pooled_train),
            "val": len(pooled_val),
            "test": len(pooled_test),
        },
        "per_symbol_row_counts": per_symbol_row_counts,
    }
    meta_path = out_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Multisym dataset meta sidecar written to %s", meta_path)

    # Step 9: Build window datasets
    # drop_cross_session_windows=False for the same reason as the single-symbol path
    # (RTH H1 = 7 bars/day, overnight/weekend gaps are expected, not exceptional).
    # Cross-SYMBOL windows are a separate concern handled inside _window_generator via
    # the `symbol` column guard — windows spanning two symbols are skipped there, so the
    # session-gap flag is not the mechanism protecting against symbol-boundary leakage.
    train_ds = SMCWindowDataset(pooled_train, labeller, stride=1, window_size=window_size, drop_cross_session_windows=False)
    val_ds = SMCWindowDataset(pooled_val, labeller, stride=window_size, window_size=window_size, drop_cross_session_windows=False)
    test_ds = SMCWindowDataset(pooled_test, labeller, stride=window_size, window_size=window_size, drop_cross_session_windows=False)

    logger.info(
        "Multisym datasets: train=%d windows, val=%d windows, test=%d windows",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # Validate: each dataset must have at least one positive-class window
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = ds.label_counts
        n_positive = sum(v for k, v in counts.items() if k != 0)
        if n_positive == 0:
            raise ValueError(
                f"Multisym dataset '{name}' has 0 positive-class windows. "
                "Check the FVG detector and split boundaries."
            )

    return train_ds, val_ds, test_ds, class_weights
