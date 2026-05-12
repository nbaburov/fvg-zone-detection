#!/usr/bin/env python3
"""train.py — Unified experiment runner. Reads one experiment YAML, dispatches by arch.

Usage:
    python scripts/training/train.py --config experiments/lstm_g1.yaml
    python scripts/training/train.py --config experiments/xgb_g1.yaml --set train.seeds=[42]
    python scripts/training/train.py --config experiments/lstm_g1.yaml --seed 17

Dispatches:
    arch=lstm  -> _train_lstm(cfg, seed)
    arch=xgb   -> _train_xgb(cfg, seed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ExperimentConfig, load_experiment
from src.config.schema import LSTMModelConfig, XGBModelConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_overrides(set_args: list[str]) -> dict:
    """Parse --set key=value pairs into override dict.

    Value is parsed with ast.literal_eval (safe: only parses literals).
    Examples:
        --set train.seeds=[42]          -> {"train.seeds": [42]}
        --set model.hidden_size=64      -> {"model.hidden_size": 64}
        --set train.lr=1e-3             -> {"train.lr": 0.001}
    """
    import ast
    overrides: dict = {}
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"--set argument must be key=value, got: {item!r}")
        key, raw_val = item.split("=", 1)
        try:
            val = ast.literal_eval(raw_val)
        except (ValueError, SyntaxError):
            val = raw_val  # treat as string
        overrides[key.strip()] = val
    return overrides


def parse_args(default_config: str | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified SMC experiment runner")
    p.add_argument(
        "--config",
        default=default_config,
        help="Path to experiment YAML.",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_args",
        metavar="KEY=VALUE",
        help="Override config field (dotted key). E.g. --set train.seeds=[42]",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed (overrides cfg.train.seeds).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Overfit check: tiny data subset, 5 epochs max.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# LSTM dispatch
# ---------------------------------------------------------------------------

def _train_lstm(cfg: ExperimentConfig, seed: int, debug: bool = False) -> None:
    """Train LSTM with config-driven HP. Uses Adam optimiser to match G1 Optuna search."""
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.config.registry import LOSSES, MODELS
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.training.early_stop import EarlyStop
    from src.training.train_utils import eval_epoch, log_run_metadata, set_seed

    assert isinstance(cfg.model, LSTMModelConfig), \
        f"Expected LSTMModelConfig, got {type(cfg.model)}"

    set_seed(seed)
    device = torch.device("cpu")  # MPS LSTM gradient kernel broken on Apple Silicon — CPU only
    print(f"Device: {device} (forced — MPS LSTM bug, see CLAUDE.md)")

    metadata = log_run_metadata(seed, device)
    metadata["device"] = "cpu (forced — MPS LSTM bug)"
    metadata["config"] = cfg.name

    # --- Data ---
    label_key = cfg.data.labeller   # "fvg" | "fvg_valid"
    labeller = LABELLERS[label_key]()

    data_dir = Path(cfg.data.data_dir)
    splits = cfg.data.splits

    if splits == "legacy":
        from src.data.split import SPLIT_BOUNDARIES_2018_2024, temporal_split
        import pandas as pd
        labeled_df = pd.read_parquet(data_dir / "spy_h1_labeled.parquet")
        tmp = LABELLERS[label_key]()
        labeled_df = labeled_df.copy()
        labeled_df["label"] = tmp.label(labeled_df).map(tmp.encoded_map).astype(int)
        train_df, val_df, test_df = temporal_split(labeled_df, SPLIT_BOUNDARIES_2018_2024)
    elif splits == "default":
        import pandas as pd
        train_df = pd.read_parquet(data_dir / "spy_h1_train.parquet")
        val_df   = pd.read_parquet(data_dir / "spy_h1_val.parquet")
        test_df  = pd.read_parquet(data_dir / "spy_h1_test.parquet")
    else:
        import pandas as pd
        splits_dir = Path(splits)
        train_df = pd.read_parquet(splits_dir / "spy_h1_train.parquet")
        val_df   = pd.read_parquet(splits_dir / "spy_h1_val.parquet")
        test_df  = pd.read_parquet(splits_dir / "spy_h1_test.parquet")
        print(f"  Using custom splits dir: {splits_dir}")

    if debug:
        train_df = train_df.iloc[:260]
        val_df   = val_df.iloc[:130]
        print(f"[DEBUG] Subset: train={len(train_df)} rows, val={len(val_df)} rows")

    w = cfg.data.window_size
    _drop_cs = cfg.data.drop_cross_session
    train_ds = SMCWindowDataset(train_df, labeller, stride=cfg.data.stride, window_size=w,
                                drop_cross_session_windows=_drop_cs)
    val_ds   = SMCWindowDataset(val_df,   labeller, stride=1, window_size=w,
                                drop_cross_session_windows=_drop_cs)
    test_ds  = SMCWindowDataset(test_df,  labeller, stride=1, window_size=w,
                                drop_cross_session_windows=_drop_cs)
    print(f"Windows — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_eval, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.train.batch_eval, shuffle=False, num_workers=0)

    # --- Class weights ---
    cw_dir  = data_dir if splits in ("default", "legacy") else Path(splits)
    cw_file = "class_weights_rawfvg.json" if label_key == "fvg" else "class_weights.json"
    with open(cw_dir / cw_file) as f:
        cw = json.load(f)
    class_weights = torch.tensor([cw["0"], cw["1"], cw["2"]], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    # --- Model ---
    m = cfg.model
    dropout   = 0.0 if cfg.train.ablation_no_dropout else m.dropout
    head_drop = 0.0 if cfg.train.ablation_no_dropout else m.head_dropout

    model = MODELS["lstm"](
        hidden_size=m.hidden_size,
        num_layers=m.num_layers,
        dropout=dropout,
        head_dropout=head_drop,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Loss ---
    loss_cls = LOSSES[cfg.train.loss]
    criterion = loss_cls(class_weights, gamma=cfg.train.focal_gamma) \
        if cfg.train.loss == "focal" else loss_cls(class_weights)

    # --- Optimiser: Adam matches G1 HP search (not AdamW+OneCycleLR) ---
    wd = 0.0 if cfg.train.ablation_no_l2 else cfg.train.weight_decay
    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=wd)

    # --- Scheduler ---
    max_epochs = 5 if debug else cfg.train.max_epochs
    if cfg.train.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.train.lr,
            total_steps=max_epochs * len(train_loader),
        )
    else:
        scheduler = None

    # --- Checkpoint path: lstm_seed{N}{label_tag}.pt ---
    label_tag = "" if label_key == "fvg_valid" else f"_{label_key}"
    ckpt_dir  = Path(cfg.runtime.checkpoint_dir) / "lstm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"lstm_seed{seed}{label_tag}.pt"

    early_stop = EarlyStop(
        patience=cfg.train.patience,
        min_delta=1e-4,
        ema_alpha=cfg.train.ema_alpha,
        mode="max",
    )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_csv = log_dir / f"lstm_seed{seed}{label_tag}.csv"

    epoch_logs, best_epoch, best_f1 = _lstm_train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        ckpt_path=ckpt_path,
        max_epochs=max_epochs,
        log_path=log_csv,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stop=early_stop,
        max_grad_norm=cfg.train.max_grad_norm,
        window_size=w,
    )

    # Load best checkpoint
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded best checkpoint: {ckpt_path}")

    # Eval
    val_metrics  = eval_epoch(model, val_loader,  criterion, device)
    test_metrics = eval_epoch(model, test_loader, criterion, device)

    from sklearn.metrics import classification_report
    print("\n=== TEST RESULTS ===")
    print(classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        target_names=["none", "bull", "bear"], digits=4, zero_division=0,
    ))
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")

    # Meta sidecar
    nan_detected = any(np.isnan(r["val_macro_f1"]) for r in epoch_logs)
    meta_path = ckpt_dir / f"lstm_seed{seed}{label_tag}.meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            **metadata,
            "config": cfg.name,
            "label": label_key,
            "splits": splits,
            "best_epoch": best_epoch,
            "best_smoothed_val_macro_f1": best_f1,
            "test_macro_f1": test_metrics["macro_f1"],
            "test_bull_f1": (test_metrics["per_class_f1"][1]
                             if len(test_metrics["per_class_f1"]) > 1 else 0.0),
            "test_bear_f1": (test_metrics["per_class_f1"][2]
                             if len(test_metrics["per_class_f1"]) > 2 else 0.0),
            "nan_detected": nan_detected,
            "n_params": n_params,
        }, f, indent=2)
    print(f"Metadata saved: {meta_path}")


def _lstm_train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    ckpt_path: Path,
    max_epochs: int,
    log_path: Path,
    optimizer,
    scheduler,
    early_stop,
    max_grad_norm: float,
    window_size: int,
) -> tuple[list[dict], int, float]:
    """Core LSTM training loop. Config-driven. Identical logic to train_lstm.train()."""
    import numpy as np
    import torch
    from src.training.train_utils import eval_epoch

    epoch_logs: list[dict] = []
    best_epoch = 0
    best_smoothed_f1 = 0.0

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_macro_f1,smoothed_val_macro_f1,"
                "val_bull_f1,val_bear_f1,lr\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        nan_hit = False

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device) if isinstance(y_batch, torch.Tensor) else \
                      torch.tensor(y_batch, device=device)

            if epoch == 1 and batch_idx == 0:
                assert x_batch.shape[-2:] == (window_size, 5), \
                    f"Unexpected input shape: {x_batch.shape}"

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch} batch {batch_idx}")
                nan_hit = True
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(loss.item())

        if nan_hit:
            break

        train_loss = float(np.mean(train_losses))
        vm = eval_epoch(model, val_loader, criterion, device)
        val_macro_f1 = vm["macro_f1"]
        vpc = vm["per_class_f1"]
        val_bull_f1 = vpc[1] if len(vpc) > 1 else 0.0
        val_bear_f1 = vpc[2] if len(vpc) > 2 else 0.0
        current_lr = (scheduler.get_last_lr()[0] if scheduler is not None
                      else optimizer.param_groups[0]["lr"])

        should_stop = early_stop.update(val_macro_f1)
        smoothed = early_stop.smoothed

        if smoothed >= best_smoothed_f1:
            best_smoothed_f1 = smoothed
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": vm["loss"],
            "val_macro_f1": val_macro_f1,
            "smoothed_val_macro_f1": smoothed,
            "val_bull_f1": val_bull_f1,
            "val_bear_f1": val_bear_f1,
            "lr": current_lr,
        }
        epoch_logs.append(row)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{vm['loss']:.6f},{val_macro_f1:.6f},"
                    f"{smoothed:.6f},{val_bull_f1:.6f},{val_bear_f1:.6f},{current_lr:.8f}\n")

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_macro_f1={val_macro_f1:.4f} | smoothed={smoothed:.4f} | "
              f"bull={val_bull_f1:.4f} | bear={val_bear_f1:.4f}")

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nTraining complete: best epoch={best_epoch}, "
          f"best smoothed val F1={best_smoothed_f1:.4f}")
    return epoch_logs, best_epoch, best_smoothed_f1


# ---------------------------------------------------------------------------
# XGB dispatch
# ---------------------------------------------------------------------------

def _train_xgb(cfg: ExperimentConfig, seed: int, debug: bool = False) -> None:
    """Train XGBoost with config-driven HP. Subprocess isolation for arm64 libgomp."""
    import subprocess

    assert isinstance(cfg.model, XGBModelConfig), \
        f"Expected XGBModelConfig, got {type(cfg.model)}"

    label_map = {"fvg": "rawfvg", "fvg_valid": "validfvg"}
    label_arg  = label_map.get(cfg.data.labeller, "validfvg")
    splits_arg = cfg.data.splits

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_xgboost.py"),
        "--seed",   str(seed),
        "--label",  label_arg,
        "--splits", splits_arg,
    ]
    print(f"Dispatching XGBoost subprocess: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"XGBoost subprocess failed (code {result.returncode})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(default_config: str | None = None) -> None:
    args = parse_args(default_config=default_config)

    if args.config is None:
        print("ERROR: --config is required (or call via arch-specific wrapper)")
        sys.exit(1)

    overrides = _parse_overrides(args.set_args)
    if args.seed is not None:
        overrides["train.seeds"] = [args.seed]

    cfg = load_experiment(args.config, overrides=overrides if overrides else None)
    print(f"Experiment: {cfg.name}")
    print(f"  arch={cfg.model.arch}  labeller={cfg.data.labeller}  seeds={cfg.train.seeds}")

    for seed in cfg.train.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        if cfg.model.arch == "lstm":
            _train_lstm(cfg, seed, debug=args.debug)
        elif cfg.model.arch == "xgb":
            _train_xgb(cfg, seed, debug=args.debug)
        else:
            raise ValueError(f"Unknown arch: {cfg.model.arch!r}")


if __name__ == "__main__":
    main()
