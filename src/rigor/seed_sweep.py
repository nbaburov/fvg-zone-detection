"""seed_sweep.py — Multi-seed training sweep for LSTM and XGBoost.

Design constraints:
  - Each seed trains a fresh model with fixed best hyperparams.
  - Test set evaluated ONCE per seed at the end of training — not during training.
  - Checkpoint-skip: if .meta.json already exists for a seed, load results and skip.
  - Supports WeightedCE and FocalLoss (gamma configurable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.train_utils import set_seed, log_run_metadata


@dataclass
class SeedSweepConfig:
    model_type: Literal["lstm", "xgb"]
    hyperparams: dict[str, Any]
    seeds: list[int]
    loss_type: Literal["weighted_ce", "focal"] = "weighted_ce"
    focal_gamma: float = 2.0
    output_dir: Path = field(default_factory=lambda: Path("reports/rigor"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    # Regularisation ablation flags (G9) — affect checkpoint name to avoid cache collisions
    ablation_no_dropout: bool = False
    ablation_no_l2: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.data_dir = Path(self.data_dir)


def run_seed_sweep(config: SeedSweepConfig) -> pd.DataFrame:
    """Train one model per seed, evaluate on test set, return results DataFrame.

    Columns: [seed, macro_f1, none_f1, bull_f1, bear_f1]

    Checkpoint-skip: if <model>_seed<N>[_focal_g<gamma>].meta.json exists, reload.
    Test set access: permitted here — final evaluation only, not during training.
    """
    rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        ckpt_name = _checkpoint_name(config, seed)
        ckpt_dir = config.checkpoint_dir / config.model_type
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if config.model_type == "lstm":
            ckpt_path = ckpt_dir / f"{ckpt_name}.pt"
            meta_path = ckpt_dir / f"{ckpt_name}.meta.json"
        else:
            ckpt_path = ckpt_dir / f"{ckpt_name}.ubj"
            meta_path = ckpt_dir / f"{ckpt_name}.ubj.meta.json"

        # Check if already done
        if meta_path.exists():
            with meta_path.open() as fh:
                meta = json.load(fh)
            if "test_macro_f1" in meta:
                print(f"  Seed {seed}: checkpoint exists — loading cached results.")
                rows.append({
                    "seed": seed,
                    "macro_f1": meta["test_macro_f1"],
                    "none_f1": meta.get("test_per_class_f1", [0, 0, 0])[0],
                    "bull_f1": meta.get("test_per_class_f1", [0, 0, 0])[1],
                    "bear_f1": meta.get("test_per_class_f1", [0, 0, 0])[2],
                })
                continue

        print(f"  Seed {seed}: training {config.model_type} ({config.loss_type})...")

        if config.model_type == "lstm":
            result = _train_lstm(config, seed, ckpt_path, meta_path)
        else:
            result = _train_xgb(config, seed, ckpt_path, meta_path)

        # Save predictions for bootstrap CI
        _save_predictions(config, seed, result)

        rows.append({
            "seed": seed,
            "macro_f1": result["test_macro_f1"],
            "none_f1": result["test_per_class_f1"][0],
            "bull_f1": result["test_per_class_f1"][1],
            "bear_f1": result["test_per_class_f1"][2],
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# LSTM training
# ---------------------------------------------------------------------------

def _train_lstm(
    config: SeedSweepConfig,
    seed: int,
    ckpt_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    FVGLabeller = LABELLERS["fvg_valid"]
    from src.models.lstm import FVGLSTMClassifier
    from src.training.early_stop import EarlyStop
    from src.training.loss import WeightedCE, FocalLoss

    set_seed(seed)

    device = _get_device()

    # Load splits
    train_df, val_df, test_df = _load_splits(config.data_dir)
    labeller = FVGLabeller()  # fvg_valid: recomputes labels from OHLCV on each split

    train_ds = SMCWindowDataset(train_df, labeller, stride=1, window_size=60, drop_cross_session_windows=False)
    val_ds = SMCWindowDataset(val_df, labeller, stride=1, window_size=60, drop_cross_session_windows=False)
    test_ds = SMCWindowDataset(test_df, labeller, stride=1, window_size=60, drop_cross_session_windows=False)

    hp = config.hyperparams
    batch_size = int(hp.get("batch_size", 32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Load class weights
    weights = _load_class_weights(config.data_dir)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = FVGLSTMClassifier(
        hidden_size=int(hp.get("hidden_size", 64)),
        num_layers=int(hp.get("num_layers", 2)),
        dropout=float(hp.get("dropout", 0.3)) if int(hp.get("num_layers", 2)) > 1 else 0.0,
        head_dropout=float(hp.get("head_dropout", 0.5)),
    ).to(device)

    if config.loss_type == "focal":
        criterion = FocalLoss(class_weights=weights_tensor, gamma=config.focal_gamma)
    else:
        criterion = WeightedCE(weights_tensor)

    lr = float(hp.get("lr", 1e-3))
    weight_decay = float(hp.get("weight_decay", 1e-4))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stop = EarlyStop(patience=15, mode="max")

    best_val_f1 = 0.0
    best_state: dict | None = None
    best_epoch = 0

    for epoch in range(100):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = torch.as_tensor(y_batch, device=device)
            optimiser.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()

        # Val evaluation
        val_f1, _, _ = _eval_f1_all(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if early_stop.update(val_f1):
            break

    # Reload best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation — single final measurement
    test_macro_f1, test_per_class_f1, (y_true, y_pred) = _eval_f1_all(model, test_loader, device)

    # Save checkpoint
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        **log_run_metadata(seed, device),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_macro_f1,
        "test_per_class_f1": test_per_class_f1,
        "hyperparams": config.hyperparams,
        "loss_type": config.loss_type,
        "focal_gamma": config.focal_gamma if config.loss_type == "focal" else None,
        "optuna_study": None,
        "optuna_trial_number": None,
        "threshold_config": None,
    }
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)

    return {
        "test_macro_f1": test_macro_f1,
        "test_per_class_f1": test_per_class_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def _train_xgb(
    config: SeedSweepConfig,
    seed: int,
    ckpt_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    import xgboost as xgb
    from sklearn.metrics import f1_score

    from src.features.window_features import extract_window_features

    set_seed(seed)

    train_df, val_df, test_df = _load_splits(config.data_dir)
    weights_arr = _load_class_weights(config.data_dir)

    X_train, y_train = extract_window_features(train_df)
    X_val, y_val = extract_window_features(val_df)
    X_test, y_test = extract_window_features(test_df)

    # Build per-sample weights from class weights
    sample_weight = np.array([weights_arr[int(y)] for y in y_train], dtype=np.float32)

    hp = config.hyperparams

    xgb_metrics = ["mlogloss"]
    clf = xgb.XGBClassifier(
        n_estimators=int(hp.get("n_estimators", 300)),
        max_depth=int(hp.get("max_depth", 4)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        min_child_weight=int(hp.get("min_child_weight", 5)),
        subsample=float(hp.get("subsample", 0.8)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.8)),
        objective="multi:softprob",
        num_class=3,
        eval_metric=xgb_metrics,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=30,
        verbosity=0,
    )

    clf.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = clf.predict(X_test).astype(int)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0.0))
    per_class_f1 = [
        float(v) for v in f1_score(y_test, y_pred, average=None, zero_division=0.0)
    ]

    clf.save_model(str(ckpt_path))

    meta = {
        "seed": seed,
        "test_macro_f1": macro_f1,
        "test_per_class_f1": per_class_f1,
        "hyperparams": hp,
        "loss_type": "weighted_ce",
        "focal_gamma": None,
        "n_estimators_used": (clf.best_iteration + 1) if clf.best_iteration is not None else hp.get("n_estimators"),
        "optuna_study": None,
        "optuna_trial_number": None,
        "threshold_config": None,
    }
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)

    return {
        "test_macro_f1": macro_f1,
        "test_per_class_f1": per_class_f1,
        "y_true": y_test,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(data_dir / "spy_h1_train.parquet")
    val = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test = pd.read_parquet(data_dir / "spy_h1_test.parquet")
    return train, val, test


def _load_class_weights(data_dir: Path) -> list[float]:
    weights_path = data_dir / "class_weights.json"
    with weights_path.open() as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [float(w) for w in data]
    return [float(data[str(i)]) for i in range(3)]


def _eval_f1_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float], tuple[np.ndarray, np.ndarray]]:
    """Return (macro_f1, per_class_f1_list, (y_true, y_pred))."""
    from sklearn.metrics import f1_score

    model.eval()
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_pred_all.extend(preds.tolist())
            if isinstance(y_batch, torch.Tensor):
                y_true_all.extend(y_batch.numpy().tolist())
            else:
                y_true_all.extend(list(y_batch))

    y_true = np.array(y_true_all, dtype=np.int64)
    y_pred = np.array(y_pred_all, dtype=np.int64)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    per_class_f1 = [float(v) for v in f1_score(y_true, y_pred, average=None, zero_division=0.0)]
    return macro_f1, per_class_f1, (y_true, y_pred)


def _checkpoint_name(config: SeedSweepConfig, seed: int) -> str:
    if config.loss_type == "focal":
        return f"{config.model_type}_focal_g{config.focal_gamma:.0f}_seed{seed}"
    name = f"{config.model_type}_seed{seed}"
    # Ablation suffix — ensures no cache collision with G2 control checkpoints
    if config.ablation_no_dropout and config.ablation_no_l2:
        name += "_no_reg"
    elif config.ablation_no_dropout:
        name += "_no_dropout"
    elif config.ablation_no_l2:
        name += "_no_l2"
    return name


def _save_predictions(
    config: SeedSweepConfig,
    seed: int,
    result: dict[str, Any],
) -> None:
    """Save y_true + y_pred npz for bootstrap CI."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    name = _checkpoint_name(config, seed)
    npz_path = config.output_dir / f"{name}_preds.npz"
    np.savez(
        npz_path,
        y_true=result["y_true"],
        y_pred=result["y_pred"],
    )
