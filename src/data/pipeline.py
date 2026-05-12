"""Top-level pipeline: download → label → split → window → class weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.labels import LABELLERS
from src.data.process import build_labelled_dataset
from src.data.split import temporal_split
from src.data.window import SMCWindowDataset

logger = logging.getLogger(__name__)

# Default output directory — tests can patch this
PROCESSED_DIR = "data/processed"


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
    labeller_name: str = "fvg",
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
                   Persisted to {PROCESSED_DIR}/class_weights.json.
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

    # Step 2: Temporal split
    train_df, val_df, test_df = temporal_split(full_df)

    # Persist splits
    for name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out_dir / f"spy_h1_{name}.parquet"
        df_split.to_parquet(path)
        logger.info("Split '%s' written to %s (%d rows)", name, path, len(df_split))

    # Step 3: Class weights from train split only
    num_classes = labeller.num_classes
    class_weights = _compute_class_weights(train_df, num_classes=num_classes)

    # Persist class weights
    weights_path = out_dir / "class_weights.json"
    weights_dict = {str(i): float(class_weights[i]) for i in range(num_classes)}
    with open(weights_path, "w") as f:
        json.dump(weights_dict, f, indent=2)
    logger.info("Class weights written to %s: %s", weights_path, weights_dict)

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
