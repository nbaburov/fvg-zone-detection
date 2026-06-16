#!/usr/bin/env python3
"""train.py — Unified experiment runner. Reads one experiment YAML, dispatches by arch.

Usage:
    python scripts/training/train.py --config experiments/lstm_g1.yaml
    python scripts/training/train.py --config experiments/xgboost_g1.yaml --set train.seeds=[42]
    python scripts/training/train.py --config experiments/lstm_g1.yaml --seed 17

Dispatches:
    arch=lstm        -> _train_lstm(cfg, seed)
    arch=xgb         -> _train_xgb(cfg, seed)
    arch=cnn_lstm    -> _train_cnn_lstm(cfg, seed)
    arch=transformer -> _train_transformer(cfg, seed)
    arch=xlstm       -> _train_xlstm(cfg, seed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ExperimentConfig, load_experiment
from src.config.schema import (
    CNNLSTMModelConfig,
    LSTMModelConfig,
    TransformerModelConfig,
    XGBModelConfig,
    XLSTMModelConfig,
)


# ---------------------------------------------------------------------------
# Checkpoint subdir helper (WS-5)
# ---------------------------------------------------------------------------

def _ckpt_subdir(arch: str, timeframe: str) -> str:
    """Return the checkpoint subdirectory name for *arch* and *timeframe* token.

    Rule (plan §3.2):
      - timeframe "h1" → ``arch`` only (no suffix) — existing paths unchanged.
      - any other token → ``{arch}_{token}`` (e.g. "lstm_5m", "xgb_15m").
    """
    if timeframe == "h1":
        return arch
    return f"{arch}_{timeframe}"


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
    p.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        dest="train_fraction",
        help="Fraction of the train split to use (contiguous, earliest-first head slice, "
             "NEVER random). Default 1.0 is an exact no-op. Applied BEFORE windowing.",
    )
    return p.parse_args()


def _head_slice_train(train_df, train_fraction: float):
    """Contiguous earliest-first head slice of the train DataFrame.

    train_fraction=1.0 returns train_df unchanged (exact no-op). Never shuffles —
    temporal order is preserved (rows [0 : ceil(N * fraction)]).
    """
    if train_fraction == 1.0:
        return train_df
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"--train-fraction must be in (0, 1], got {train_fraction}")
    import math
    n = int(math.ceil(len(train_df) * train_fraction))
    sliced = train_df.iloc[:n]
    print(f"[train-fraction={train_fraction}] head slice: {len(sliced)}/{len(train_df)} rows "
          f"(earliest-first, contiguous)")
    return sliced


# ---------------------------------------------------------------------------
# LSTM dispatch
# ---------------------------------------------------------------------------

def _train_lstm(cfg: ExperimentConfig, seed: int, debug: bool = False,
                train_fraction: float = 1.0) -> None:
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
    tf_token = cfg.data.timeframe  # "h1" | "5m" | "15m" (WS-5)
    scope = cfg.data.dataset       # "spy" | "multisym"

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
        train_df = pd.read_parquet(data_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(data_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(data_dir / f"{scope}_{tf_token}_test.parquet")
    else:
        import pandas as pd
        splits_dir = Path(splits)
        train_df = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_test.parquet")
        print(f"  Using custom splits dir: {splits_dir}")

    if debug:
        train_df = train_df.iloc[:260]
        val_df   = val_df.iloc[:130]
        print(f"[DEBUG] Subset: train={len(train_df)} rows, val={len(val_df)} rows")

    # A2a: contiguous earliest-first head slice BEFORE windowing (no-op at 1.0).
    train_df = _head_slice_train(train_df, train_fraction)

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
    cw_file = "class_weights_rawfvg.json" if label_key == "fvg" else f"class_weights_{scope}_{tf_token}.json"
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
    # Subdir is "lstm" for h1 (backward-compat) and "lstm_{token}" for other TFs (WS-5).
    label_tag = "" if label_key == "fvg_valid" else f"_{label_key}"
    ckpt_dir  = Path(cfg.runtime.checkpoint_dir) / _ckpt_subdir("lstm", tf_token)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"lstm_seed{seed}{label_tag}.pt"

    early_stop = EarlyStop(
        patience=cfg.train.patience,
        min_delta=1e-4,
        ema_alpha=cfg.train.ema_alpha,
        mode="max",
    )

    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
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
            "n_train_windows": len(train_ds),
            "train_fraction": train_fraction,
            "timeframe": tf_token,
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
    """Core torch training loop. Config-driven. Shared by LSTM, CNN-LSTM, Transformer, xLSTM.

    Logs a per-epoch train_macro_f1 column (computed via a no-dropout/no-grad
    eval pass over the train loader) alongside the validation metrics. This is
    the train/val generalisation gap diagnostic — it benefits every torch arch
    routed through this loop.
    """
    import numpy as np
    import torch
    from src.training.train_utils import eval_epoch

    epoch_logs: list[dict] = []
    best_epoch = 0
    best_smoothed_f1 = 0.0

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_macro_f1,val_loss,val_macro_f1,smoothed_val_macro_f1,"
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
        # Train-eval pass: macro-F1 on the train set with no dropout / no grad.
        # F1 is order-independent so the shuffled train_loader is fine here.
        tm = eval_epoch(model, train_loader, criterion, device)
        train_macro_f1 = tm["macro_f1"]
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
            "train_macro_f1": train_macro_f1,
            "val_loss": vm["loss"],
            "val_macro_f1": val_macro_f1,
            "smoothed_val_macro_f1": smoothed,
            "val_bull_f1": val_bull_f1,
            "val_bear_f1": val_bear_f1,
            "lr": current_lr,
        }
        epoch_logs.append(row)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_macro_f1:.6f},{vm['loss']:.6f},"
                    f"{val_macro_f1:.6f},{smoothed:.6f},{val_bull_f1:.6f},"
                    f"{val_bear_f1:.6f},{current_lr:.8f}\n")

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"train_f1={train_macro_f1:.4f} | "
              f"val_macro_f1={val_macro_f1:.4f} | smoothed={smoothed:.4f} | "
              f"bull={val_bull_f1:.4f} | bear={val_bear_f1:.4f}")

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nTraining complete: best epoch={best_epoch}, "
          f"best smoothed val F1={best_smoothed_f1:.4f}")
    return epoch_logs, best_epoch, best_smoothed_f1


# ---------------------------------------------------------------------------
# CNN-LSTM dispatch
# ---------------------------------------------------------------------------

def _train_cnn_lstm(cfg: ExperimentConfig, seed: int, debug: bool = False,
                    train_fraction: float = 1.0) -> None:
    """Train CNN-LSTM with config-driven HP. CPU only — same MPS constraint as LSTM."""
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.config.registry import LOSSES, MODELS
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.training.early_stop import EarlyStop
    from src.training.train_utils import eval_epoch, log_run_metadata, set_seed

    assert isinstance(cfg.model, CNNLSTMModelConfig), \
        f"Expected CNNLSTMModelConfig, got {type(cfg.model)}"

    set_seed(seed)
    device = torch.device("cpu")  # MPS LSTM gradient kernel broken on Apple Silicon — CPU only
    print(f"Device: {device} (forced — MPS LSTM bug, see CLAUDE.md)")

    metadata = log_run_metadata(seed, device)
    metadata["device"] = "cpu (forced — MPS LSTM bug)"
    metadata["config"] = cfg.name

    # --- Data ---
    label_key = cfg.data.labeller
    labeller = LABELLERS[label_key]()

    data_dir = Path(cfg.data.data_dir)
    splits = cfg.data.splits
    tf_token = cfg.data.timeframe  # "h1" | "5m" | "15m" (WS-5)
    scope = cfg.data.dataset       # "spy" | "multisym"

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
        train_df = pd.read_parquet(data_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(data_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(data_dir / f"{scope}_{tf_token}_test.parquet")
    else:
        import pandas as pd
        splits_dir = Path(splits)
        train_df = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_test.parquet")
        print(f"  Using custom splits dir: {splits_dir}")

    if debug:
        train_df = train_df.iloc[:260]
        val_df   = val_df.iloc[:130]
        print(f"[DEBUG] Subset: train={len(train_df)} rows, val={len(val_df)} rows")

    # A2a: contiguous earliest-first head slice BEFORE windowing (no-op at 1.0).
    train_df = _head_slice_train(train_df, train_fraction)

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
    cw_file = "class_weights_rawfvg.json" if label_key == "fvg" else f"class_weights_{scope}_{tf_token}.json"
    with open(cw_dir / cw_file) as f:
        cw = json.load(f)
    class_weights = torch.tensor([cw["0"], cw["1"], cw["2"]], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    # --- Model ---
    m = cfg.model
    dropout   = 0.0 if cfg.train.ablation_no_dropout else m.dropout
    head_drop = 0.0 if cfg.train.ablation_no_dropout else m.head_dropout

    model = MODELS["cnn_lstm"](
        conv_filters=m.conv_filters,
        kernel_size=m.kernel_size,
        n_conv_layers=m.n_conv_layers,
        use_pool=m.use_pool,
        pool_type=m.pool_type,
        lstm_hidden=m.lstm_hidden,
        lstm_layers=m.lstm_layers,
        dropout=dropout,
        head_dropout=head_drop,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Loss ---
    loss_cls = LOSSES[cfg.train.loss]
    criterion = loss_cls(class_weights, gamma=cfg.train.focal_gamma) \
        if cfg.train.loss == "focal" else loss_cls(class_weights)

    # --- Optimiser ---
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

    # --- Checkpoint path: cnn_lstm_seed{N}{label_tag}.pt ---
    # Subdir is "cnn_lstm" for h1 (backward-compat) and "cnn_lstm_{token}" for other TFs (WS-5).
    label_tag = "" if label_key == "fvg_valid" else f"_{label_key}"
    ckpt_dir  = Path(cfg.runtime.checkpoint_dir) / _ckpt_subdir("cnn_lstm", tf_token)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"cnn_lstm_seed{seed}{label_tag}.pt"

    early_stop = EarlyStop(
        patience=cfg.train.patience,
        min_delta=1e-4,
        ema_alpha=cfg.train.ema_alpha,
        mode="max",
    )

    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"cnn_lstm_seed{seed}{label_tag}.csv"

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
    meta_path = ckpt_dir / f"cnn_lstm_seed{seed}{label_tag}.meta.json"
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
            "arch": "cnn_lstm",
            "conv_filters": m.conv_filters,
            "kernel_size": m.kernel_size,
            "n_conv_layers": m.n_conv_layers,
            "use_pool": m.use_pool,
            "lstm_hidden": m.lstm_hidden,
            "lstm_layers": m.lstm_layers,
            "n_train_windows": len(train_ds),
            "train_fraction": train_fraction,
            "timeframe": tf_token,
        }, f, indent=2)
    print(f"Metadata saved: {meta_path}")


# ---------------------------------------------------------------------------
# Transformer dispatch
# ---------------------------------------------------------------------------

def _train_transformer(cfg: ExperimentConfig, seed: int, debug: bool = False,
                       train_fraction: float = 1.0) -> None:
    """Train Transformer encoder with config-driven HP. CPU only — same MPS constraint.

    Reuses the shared torch training loop (_lstm_train_loop): same loss, optimiser,
    early-stop, and metadata sidecar as LSTM. Only model construction differs.
    """
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.config.registry import LOSSES, MODELS
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.training.early_stop import EarlyStop
    from src.training.train_utils import eval_epoch, log_run_metadata, set_seed

    assert isinstance(cfg.model, TransformerModelConfig), \
        f"Expected TransformerModelConfig, got {type(cfg.model)}"

    set_seed(seed)
    device = torch.device("cpu")  # MPS gradient kernel broken on Apple Silicon — CPU only
    print(f"Device: {device} (forced — MPS bug, see CLAUDE.md)")

    metadata = log_run_metadata(seed, device)
    metadata["device"] = "cpu (forced — MPS bug)"
    metadata["config"] = cfg.name

    # --- Data ---
    label_key = cfg.data.labeller
    labeller = LABELLERS[label_key]()

    data_dir = Path(cfg.data.data_dir)
    splits = cfg.data.splits
    tf_token = cfg.data.timeframe  # "h1" | "5m" | "15m" (WS-5)
    scope = cfg.data.dataset       # "spy" | "multisym"

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
        train_df = pd.read_parquet(data_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(data_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(data_dir / f"{scope}_{tf_token}_test.parquet")
    else:
        import pandas as pd
        splits_dir = Path(splits)
        train_df = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_test.parquet")
        print(f"  Using custom splits dir: {splits_dir}")

    if debug:
        train_df = train_df.iloc[:260]
        val_df   = val_df.iloc[:130]
        print(f"[DEBUG] Subset: train={len(train_df)} rows, val={len(val_df)} rows")

    # A2a: contiguous earliest-first head slice BEFORE windowing (no-op at 1.0).
    train_df = _head_slice_train(train_df, train_fraction)

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
    cw_file = "class_weights_rawfvg.json" if label_key == "fvg" else f"class_weights_{scope}_{tf_token}.json"
    with open(cw_dir / cw_file) as f:
        cw = json.load(f)
    class_weights = torch.tensor([cw["0"], cw["1"], cw["2"]], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    # --- Model ---
    m = cfg.model
    dropout   = 0.0 if cfg.train.ablation_no_dropout else m.dropout
    head_drop = 0.0 if cfg.train.ablation_no_dropout else m.head_dropout

    model = MODELS["transformer"](
        d_model=m.d_model,
        nhead=m.nhead,
        num_layers=m.num_layers,
        dim_feedforward=m.dim_feedforward,
        dropout=dropout,
        head_dropout=head_drop,
        pool=m.pool,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Loss ---
    loss_cls = LOSSES[cfg.train.loss]
    criterion = loss_cls(class_weights, gamma=cfg.train.focal_gamma) \
        if cfg.train.loss == "focal" else loss_cls(class_weights)

    # --- Optimiser ---
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

    # --- Checkpoint path: transformer_seed{N}{label_tag}.pt ---
    # Subdir is "transformer" for h1 (backward-compat) and "transformer_{token}" for other TFs (WS-5).
    label_tag = "" if label_key == "fvg_valid" else f"_{label_key}"
    ckpt_dir  = Path(cfg.runtime.checkpoint_dir) / _ckpt_subdir("transformer", tf_token)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"transformer_seed{seed}{label_tag}.pt"

    early_stop = EarlyStop(
        patience=cfg.train.patience,
        min_delta=1e-4,
        ema_alpha=cfg.train.ema_alpha,
        mode="max",
    )

    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"transformer_seed{seed}{label_tag}.csv"

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
    meta_path = ckpt_dir / f"transformer_seed{seed}{label_tag}.meta.json"
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
            "arch": "transformer",
            "d_model": m.d_model,
            "nhead": m.nhead,
            "num_layers": m.num_layers,
            "dim_feedforward": m.dim_feedforward,
            "pool": m.pool,
            "n_train_windows": len(train_ds),
            "train_fraction": train_fraction,
            "timeframe": tf_token,
        }, f, indent=2)
    print(f"Metadata saved: {meta_path}")


# ---------------------------------------------------------------------------
# xLSTM dispatch
# ---------------------------------------------------------------------------

def _train_xlstm(cfg: ExperimentConfig, seed: int, debug: bool = False,
                 train_fraction: float = 1.0) -> None:
    """Train xLSTM (sLSTM-only stack) with config-driven HP. CPU only — same MPS constraint.

    Reuses the shared torch training loop (_lstm_train_loop): same loss, optimiser,
    early-stop, and metadata sidecar as LSTM. Only model construction differs.
    """
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.config.registry import LOSSES, MODELS
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.training.early_stop import EarlyStop
    from src.training.train_utils import eval_epoch, log_run_metadata, set_seed

    assert isinstance(cfg.model, XLSTMModelConfig), \
        f"Expected XLSTMModelConfig, got {type(cfg.model)}"

    set_seed(seed)
    device = torch.device("cpu")  # MPS gradient kernel broken on Apple Silicon — CPU only
    print(f"Device: {device} (forced — MPS bug, see CLAUDE.md)")

    metadata = log_run_metadata(seed, device)
    metadata["device"] = "cpu (forced — MPS bug)"
    metadata["config"] = cfg.name

    # --- Data ---
    label_key = cfg.data.labeller
    labeller = LABELLERS[label_key]()

    data_dir = Path(cfg.data.data_dir)
    splits = cfg.data.splits
    tf_token = cfg.data.timeframe  # "h1" | "5m" | "15m" (WS-5)
    scope = cfg.data.dataset       # "spy" | "multisym"

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
        train_df = pd.read_parquet(data_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(data_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(data_dir / f"{scope}_{tf_token}_test.parquet")
    else:
        import pandas as pd
        splits_dir = Path(splits)
        train_df = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_train.parquet")
        val_df   = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_val.parquet")
        test_df  = pd.read_parquet(splits_dir / f"{scope}_{tf_token}_test.parquet")
        print(f"  Using custom splits dir: {splits_dir}")

    if debug:
        train_df = train_df.iloc[:260]
        val_df   = val_df.iloc[:130]
        print(f"[DEBUG] Subset: train={len(train_df)} rows, val={len(val_df)} rows")

    # A2a: contiguous earliest-first head slice BEFORE windowing (no-op at 1.0).
    train_df = _head_slice_train(train_df, train_fraction)

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
    cw_file = "class_weights_rawfvg.json" if label_key == "fvg" else f"class_weights_{scope}_{tf_token}.json"
    with open(cw_dir / cw_file) as f:
        cw = json.load(f)
    class_weights = torch.tensor([cw["0"], cw["1"], cw["2"]], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    # --- Model ---
    m = cfg.model
    dropout   = 0.0 if cfg.train.ablation_no_dropout else m.dropout
    head_drop = 0.0 if cfg.train.ablation_no_dropout else m.head_dropout

    model = MODELS["xlstm"](
        embedding_dim=m.embedding_dim,
        num_blocks=m.num_blocks,
        num_heads=m.num_heads,
        dropout=dropout,
        head_dropout=head_drop,
        context_length=w,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Loss ---
    loss_cls = LOSSES[cfg.train.loss]
    criterion = loss_cls(class_weights, gamma=cfg.train.focal_gamma) \
        if cfg.train.loss == "focal" else loss_cls(class_weights)

    # --- Optimiser ---
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

    # --- Checkpoint path: xlstm_seed{N}{label_tag}.pt ---
    # Subdir is "xlstm" for h1 (backward-compat) and "xlstm_{token}" for other TFs (WS-5).
    label_tag = "" if label_key == "fvg_valid" else f"_{label_key}"
    ckpt_dir  = Path(cfg.runtime.checkpoint_dir) / _ckpt_subdir("xlstm", tf_token)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"xlstm_seed{seed}{label_tag}.pt"

    early_stop = EarlyStop(
        patience=cfg.train.patience,
        min_delta=1e-4,
        ema_alpha=cfg.train.ema_alpha,
        mode="max",
    )

    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"xlstm_seed{seed}{label_tag}.csv"

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
    meta_path = ckpt_dir / f"xlstm_seed{seed}{label_tag}.meta.json"
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
            "arch": "xlstm",
            "embedding_dim": m.embedding_dim,
            "num_blocks": m.num_blocks,
            "num_heads": m.num_heads,
            "n_train_windows": len(train_ds),
            "train_fraction": train_fraction,
            "timeframe": tf_token,
        }, f, indent=2)
    print(f"Metadata saved: {meta_path}")


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
        "--seed",      str(seed),
        "--label",     label_arg,
        "--splits",    splits_arg,
        "--timeframe", cfg.data.timeframe,  # WS-5: routes to correct parquet + ckpt subdir
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
            _train_lstm(cfg, seed, debug=args.debug, train_fraction=args.train_fraction)
        elif cfg.model.arch == "xgb":
            _train_xgb(cfg, seed, debug=args.debug)
        elif cfg.model.arch == "cnn_lstm":
            _train_cnn_lstm(cfg, seed, debug=args.debug, train_fraction=args.train_fraction)
        elif cfg.model.arch == "transformer":
            _train_transformer(cfg, seed, debug=args.debug, train_fraction=args.train_fraction)
        elif cfg.model.arch == "xlstm":
            _train_xlstm(cfg, seed, debug=args.debug, train_fraction=args.train_fraction)
        else:
            raise ValueError(f"Unknown arch: {cfg.model.arch!r}")


if __name__ == "__main__":
    main()
