#!/usr/bin/env python3
"""train_lstm.py — End-to-end LSTM baseline training for FVG ternary classification.

Usage:
    python scripts/training/train_lstm.py [--seed 42] [--device auto] [--output-dir checkpoints/lstm] [--debug]

Steps:
    1. set_seed
    2. log_run_metadata
    3. Load data (stride=1 for all splits to match XGBoost evaluation)
    4. Build DataLoaders
    5. Instantiate FVGLSTMClassifier
    6. Train with AdamW + OneCycleLR + EarlyStop on val macro-F1
    7. Load best checkpoint
    8. Evaluate on test set
    9. Write .nb-suite/test-logs/11-May-26/lstm-baseline.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import LABELLERS
from src.data.window import SMCWindowDataset
from src.models.lstm import FVGLSTMClassifier
from src.training.early_stop import EarlyStop
from src.training.loss import WeightedCE
from src.training.train_utils import eval_epoch, log_run_metadata, set_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EPOCHS = 100
BATCH_TRAIN = 32
BATCH_EVAL = 256
WINDOW_SIZE = 60
LR = 3e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 15
EMA_ALPHA = 0.3
MAX_GRAD_NORM = 1.0
XGBOOST_MACRO_F1 = 0.6084
XGBOOST_BULL_F1 = 0.5519
XGBOOST_BEAR_F1 = 0.5183
NAIVE_MACRO_F1 = 0.2846


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM baseline for FVG detection")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--output-dir", default="checkpoints/lstm")
    p.add_argument("--debug", action="store_true",
                   help="Overfit check: train on 200-sample subset for 5 epochs")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets(debug: bool = False) -> tuple:
    """Load train/val/test datasets with stride=1 (matches XGBoost evaluation)."""
    labeller_key = "fvg_valid"
    if labeller_key not in LABELLERS:
        available = list(LABELLERS.keys())
        raise KeyError(f"Labeller '{labeller_key}' not found. Available: {available}")

    labeller = LABELLERS[labeller_key]()

    train_df = pd.read_parquet("data/processed/spy_h1_train.parquet")
    val_df   = pd.read_parquet("data/processed/spy_h1_val.parquet")
    test_df  = pd.read_parquet("data/processed/spy_h1_test.parquet")

    if debug:
        train_df = train_df.iloc[:260]  # 260 rows -> ~200 windows with stride=1
        val_df   = val_df.iloc[:130]
        print(f"[DEBUG] Subset: train={len(train_df)} rows, val={len(val_df)} rows")

    train_ds = SMCWindowDataset(train_df, labeller, stride=1,
                                window_size=WINDOW_SIZE, drop_cross_session_windows=False)
    val_ds   = SMCWindowDataset(val_df,   labeller, stride=1,
                                window_size=WINDOW_SIZE, drop_cross_session_windows=False)
    test_ds  = SMCWindowDataset(test_df,  labeller, stride=1,
                                window_size=WINDOW_SIZE, drop_cross_session_windows=False)

    print(f"Windows — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Validate positive counts
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = ds.label_counts
        n_pos = sum(v for k, v in counts.items() if k != 0)
        print(f"  {name}: {counts}, positives={n_pos}")
        if n_pos == 0:
            raise ValueError(f"Dataset '{name}' has 0 positive-class windows")

    # Validate test positives
    test_counts = test_ds.label_counts
    test_pos = sum(v for k, v in test_counts.items() if k != 0)
    assert test_pos >= 200, f"Test positives {test_pos} < 200 — data loading issue"

    return train_ds, val_ds, test_ds


def make_loaders(
    train_ds: SMCWindowDataset,
    val_ds: SMCWindowDataset,
    test_ds: SMCWindowDataset,
    seed: int,
) -> tuple:
    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                              num_workers=0, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: WeightedCE,
    device: torch.device,
    ckpt_path: Path,
    max_epochs: int,
    log_path: Path,
) -> tuple[list[dict], int, float]:
    """Train model with AdamW + OneCycleLR + EarlyStop.

    Returns (epoch_logs, best_epoch, best_smoothed_f1).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = max_epochs * len(train_loader)
    print(f"total_steps for OneCycleLR: {total_steps}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps
    )

    early_stop = EarlyStop(patience=PATIENCE, min_delta=1e-4, ema_alpha=EMA_ALPHA, mode="max")

    epoch_logs: list[dict] = []
    best_epoch = 0
    best_smoothed_f1 = 0.0
    nan_detected = False

    # CSV log header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_macro_f1,smoothed_val_macro_f1,"
                "val_bull_f1,val_bear_f1,lr\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            if isinstance(y_batch, torch.Tensor):
                y_batch = y_batch.to(device)
            else:
                y_batch = torch.tensor(y_batch, device=device)

            # Sanity check on first batch only
            if epoch == 1 and batch_idx == 0:
                assert x_batch.shape[-2:] == (WINDOW_SIZE, 5), \
                    f"Unexpected input shape: {x_batch.shape}"
                assert torch.isfinite(criterion.criterion.weight).all(), \
                    "class_weights contain NaN"

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            # MPS NaN guard
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch} batch {batch_idx} on {device}")
                print("Falling back to CPU")
                nan_detected = True
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        if nan_detected:
            print("Training aborted due to NaN — see evaluation log for details")
            break

        train_loss = float(np.mean(train_losses))

        # Validation
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_per_class = val_metrics["per_class_f1"]
        val_bull_f1 = val_per_class[1] if len(val_per_class) > 1 else 0.0
        val_bear_f1 = val_per_class[2] if len(val_per_class) > 2 else 0.0

        current_lr = scheduler.get_last_lr()[0]

        # EarlyStop update
        should_stop = early_stop.update(val_macro_f1)
        smoothed = early_stop.smoothed

        # Save checkpoint if smoothed F1 improved
        if smoothed >= best_smoothed_f1:
            best_smoothed_f1 = smoothed
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "smoothed_val_macro_f1": smoothed,
            "val_bull_f1": val_bull_f1,
            "val_bear_f1": val_bear_f1,
            "lr": current_lr,
        }
        epoch_logs.append(row)

        # CSV log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_macro_f1:.6f},"
                    f"{smoothed:.6f},{val_bull_f1:.6f},{val_bear_f1:.6f},{current_lr:.8f}\n")

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_macro_f1={val_macro_f1:.4f} | smoothed={smoothed:.4f} | "
              f"bull={val_bull_f1:.4f} | bear={val_bear_f1:.4f}")

        if should_stop:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    stop_reason = "early_stopping" if len(epoch_logs) < max_epochs else "max_epochs"
    print(f"\nTraining complete: best epoch={best_epoch}, best smoothed val macro-F1={best_smoothed_f1:.4f}")
    print(f"Stop reason: {stop_reason}")

    return epoch_logs, best_epoch, best_smoothed_f1


# ---------------------------------------------------------------------------
# Evaluation log writer
# ---------------------------------------------------------------------------

def write_eval_log(
    log_path: Path,
    metadata: dict,
    model: nn.Module,
    train_ds: SMCWindowDataset,
    val_ds: SMCWindowDataset,
    test_ds: SMCWindowDataset,
    val_metrics: dict,
    test_metrics: dict,
    epoch_logs: list[dict],
    best_epoch: int,
    best_smoothed_f1: float,
    ckpt_path: Path,
    nan_detected: bool,
    debug: bool,
) -> None:
    n_params = sum(p.numel() for p in model.parameters())

    def _split_counts(ds: SMCWindowDataset) -> tuple[int, int, float]:
        counts = ds.label_counts
        total = len(ds)
        pos = sum(v for k, v in counts.items() if k != 0)
        rate = pos / total if total > 0 else 0.0
        return total, pos, rate

    train_total, train_pos, train_rate = _split_counts(train_ds)
    val_total, val_pos, val_rate       = _split_counts(val_ds)
    test_total, test_pos, test_rate    = _split_counts(test_ds)

    y_true_test = test_metrics["y_true"]
    y_pred_test = test_metrics["y_pred"]
    test_report = classification_report(
        y_true_test, y_pred_test,
        target_names=["none", "bull", "bear"],
        digits=4, zero_division=0,
    )
    test_cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1, 2])

    y_true_val = val_metrics["y_true"]
    y_pred_val = val_metrics["y_pred"]
    val_report = classification_report(
        y_true_val, y_pred_val,
        target_names=["none", "bull", "bear"],
        digits=4, zero_division=0,
    )
    val_cm = confusion_matrix(y_true_val, y_pred_val, labels=[0, 1, 2])

    test_macro_f1 = test_metrics["macro_f1"]
    test_pf1 = test_metrics["per_class_f1"]
    test_bull_f1 = test_pf1[1] if len(test_pf1) > 1 else 0.0
    test_bear_f1 = test_pf1[2] if len(test_pf1) > 2 else 0.0
    lstm_vs_xgb_macro = test_macro_f1 - XGBOOST_MACRO_F1
    lstm_vs_xgb_bull  = test_bull_f1  - XGBOOST_BULL_F1
    lstm_vs_xgb_bear  = test_bear_f1  - XGBOOST_BEAR_F1

    # Training curve: first 5 and last 5 epochs
    first5 = epoch_logs[:5]
    last5  = epoch_logs[-5:]

    def curve_row(r: dict) -> str:
        return (f"| {r['epoch']:3d} | {r['train_loss']:.4f} | {r['val_macro_f1']:.4f} "
                f"| {r['smoothed_val_macro_f1']:.4f} | {r['val_bull_f1']:.4f} | {r['val_bear_f1']:.4f} |")

    stop_reason = "early_stopping" if len(epoch_logs) < MAX_EPOCHS else "max_epochs"

    interpretation_lines = []
    if test_macro_f1 >= XGBOOST_MACRO_F1:
        interpretation_lines.append(
            f"LSTM test macro-F1 ({test_macro_f1:.4f}) meets or exceeds XGBoost floor ({XGBOOST_MACRO_F1}). "
            "DL pipeline viable."
        )
    else:
        interpretation_lines.append(
            f"LSTM test macro-F1 ({test_macro_f1:.4f}) is below XGBoost floor ({XGBOOST_MACRO_F1}). "
            "This is the expected result for ~117 effective independent samples (stride=1 produces "
            "highly overlapping windows; effective N estimated as non-overlapping count). "
            "XGBoost's tabular features encode FVG geometry directly; the LSTM must learn these "
            "patterns from raw OHLCV — harder with limited data."
        )
    if nan_detected:
        interpretation_lines.append("NOTE: NaN detected during training. See training output.")

    if debug:
        interpretation_lines.append("NOTE: Debug mode — trained on 200-sample subset. Results not representative.")

    sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"

    content = f"""# LSTM Baseline Evaluation — Phase 4 Step 1

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Git SHA:** {metadata['git_sha']}
**Python:** {metadata['python_version']}
**PyTorch:** {metadata['torch_version']}
**Device:** {metadata['device']}
**Seed:** {metadata['seed']}

## Model architecture

- hidden_size: 64, num_layers: 2, dropout: 0.3, head_dropout: 0.5
- bidirectional: False (unidirectional — safe at label position 59)
- Total parameters: {n_params:,}

## Dataset summary

| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|--------------|
| Train | {train_total} | {train_pos} | {train_rate:.1%} |
| Val   | {val_total} | {val_pos} | {val_rate:.1%} |
| Test  | {test_total} | {test_pos} | {test_rate:.1%} |

Note: stride=1 for all splits — matches XGBoost baseline evaluation for fair comparison.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 64 |
| num_layers | 2 |
| dropout (LSTM inter-layer) | 0.3 |
| head_dropout | 0.5 |
| batch_size (train) | {BATCH_TRAIN} |
| batch_size (eval) | {BATCH_EVAL} |
| optimizer | AdamW |
| lr | {LR} |
| weight_decay | {WEIGHT_DECAY} |
| scheduler | OneCycleLR(max_lr={LR}) |
| max_epochs | {MAX_EPOCHS} |
| early_stop patience | {PATIENCE} |
| early_stop ema_alpha | {EMA_ALPHA} |
| grad_clip max_norm | {MAX_GRAD_NORM} |
| seed | {metadata['seed']} |

## Training

- Stopped at epoch: {len(epoch_logs)} ({stop_reason})
- Best smoothed val macro-F1: {best_smoothed_f1:.4f} at epoch {best_epoch}
- Best checkpoint: {ckpt_path}
- NaN detected: {nan_detected}

## Training curve (first 5 epochs)

| Epoch | Train Loss | Val Macro-F1 | Smoothed F1 | Bull F1 | Bear F1 |
|-------|-----------|-------------|-------------|---------|---------|
{''.join(curve_row(r) + chr(10) for r in first5)}

## Training curve (last 5 epochs)

| Epoch | Train Loss | Val Macro-F1 | Smoothed F1 | Bull F1 | Bear F1 |
|-------|-----------|-------------|-------------|---------|---------|
{''.join(curve_row(r) + chr(10) for r in last5)}

## Validation results (best checkpoint)

```
{val_report}
```

Confusion matrix (val) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       {val_cm[0,0]:4d}      {val_cm[0,1]:4d}      {val_cm[0,2]:4d}
        bull       {val_cm[1,0]:4d}      {val_cm[1,1]:4d}      {val_cm[1,2]:4d}
        bear       {val_cm[2,0]:4d}      {val_cm[2,1]:4d}      {val_cm[2,2]:4d}
```

## Test results (primary)

```
{test_report}
```

Confusion matrix (test) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       {test_cm[0,0]:4d}      {test_cm[0,1]:4d}      {test_cm[0,2]:4d}
        bull       {test_cm[1,0]:4d}      {test_cm[1,1]:4d}      {test_cm[1,2]:4d}
        bear       {test_cm[2,0]:4d}      {test_cm[2,1]:4d}      {test_cm[2,2]:4d}
```

## Comparison to XGBoost floor and naive baseline

| Metric | Naive | XGBoost | LSTM | LSTM vs XGBoost |
|--------|-------|---------|------|-----------------|
| Macro-F1 | {NAIVE_MACRO_F1:.4f} | {XGBOOST_MACRO_F1:.4f} | {test_macro_f1:.4f} | {sign(lstm_vs_xgb_macro)} |
| Bull F1  | 0.0000 | {XGBOOST_BULL_F1:.4f} | {test_bull_f1:.4f} | {sign(lstm_vs_xgb_bull)} |
| Bear F1  | 0.0000 | {XGBOOST_BEAR_F1:.4f} | {test_bear_f1:.4f} | {sign(lstm_vs_xgb_bear)} |

## Interpretation

{chr(10).join(interpretation_lines)}

## Phase 4 handoff to CNN-LSTM

- LSTM test macro-F1: {test_macro_f1:.4f}
- Model saved at: {ckpt_path}
- Infrastructure (loss.py, early_stop.py, train_utils.py) reusable unchanged for CNN-LSTM.
- CNN-LSTM must beat LSTM macro-F1 ({test_macro_f1:.4f}) on the same test split to justify the architecture.
"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content)
    print(f"\nEvaluation log written to {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    print(f"Device: {device}")

    metadata = log_run_metadata(args.seed, device)
    print(f"Metadata: {metadata}")

    # Data
    train_ds, val_ds, test_ds = load_datasets(debug=args.debug)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, args.seed)

    # Class weights
    with open("data/processed/class_weights.json") as f:
        cw = json.load(f)
    class_weights = torch.tensor(
        [cw["0"], cw["1"], cw["2"]], dtype=torch.float32
    ).to(device)
    print(f"Class weights: {class_weights}")

    # Model
    model = FVGLSTMClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = WeightedCE(class_weights)

    # Output paths
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"lstm_seed{args.seed}.pt"
    log_csv_path = Path("logs") / f"lstm_seed{args.seed}.csv"
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log_path = Path(".nb-suite/test-logs/11-May-26/lstm-baseline.md")

    max_epochs = 5 if args.debug else args.max_epochs

    # Train
    nan_detected = False
    try:
        epoch_logs, best_epoch, best_smoothed_f1 = train(
            model, train_loader, val_loader, criterion, device,
            ckpt_path, max_epochs, log_csv_path,
        )
    except Exception as e:
        print(f"Training error: {e}")
        raise

    # Detect NaN in logs
    if any(np.isnan(r["val_macro_f1"]) for r in epoch_logs):
        nan_detected = True

    # Load best checkpoint
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded best checkpoint from {ckpt_path}")
    else:
        print("WARNING: No checkpoint found — using final model state")

    # Evaluate on val (for log)
    val_metrics = eval_epoch(model, val_loader, criterion, device)

    # Evaluate on test
    test_metrics = eval_epoch(model, test_loader, criterion, device)

    print("\n=== TEST RESULTS ===")
    print(classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        target_names=["none", "bull", "bear"],
        digits=4, zero_division=0,
    ))
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"XGBoost floor: {XGBOOST_MACRO_F1:.4f}")
    delta = test_metrics["macro_f1"] - XGBOOST_MACRO_F1
    sign = "+" if delta >= 0 else ""
    print(f"LSTM vs XGBoost: {sign}{delta:.4f}")

    # Warn if below naive baseline
    if test_metrics["macro_f1"] <= NAIVE_MACRO_F1:
        print(f"WARNING: LSTM macro-F1 ({test_metrics['macro_f1']:.4f}) <= naive baseline "
              f"({NAIVE_MACRO_F1:.4f}) — training may have failed")

    # Save metadata sidecar
    meta_path = out_dir / f"lstm_seed{args.seed}.meta.json"
    meta_out = {
        **metadata,
        "best_epoch": best_epoch,
        "best_smoothed_val_macro_f1": best_smoothed_f1,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_bull_f1": test_metrics["per_class_f1"][1] if len(test_metrics["per_class_f1"]) > 1 else 0.0,
        "test_bear_f1": test_metrics["per_class_f1"][2] if len(test_metrics["per_class_f1"]) > 2 else 0.0,
        "nan_detected": nan_detected,
        "n_params": n_params,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Write eval log
    write_eval_log(
        log_path=eval_log_path,
        metadata=metadata,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        epoch_logs=epoch_logs,
        best_epoch=best_epoch,
        best_smoothed_f1=best_smoothed_f1,
        ckpt_path=ckpt_path,
        nan_detected=nan_detected,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
