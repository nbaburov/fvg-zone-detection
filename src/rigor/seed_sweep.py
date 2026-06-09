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
    model_type: Literal["lstm", "xgb", "cnn_lstm", "transformer", "xlstm"]
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

    @classmethod
    def from_experiment_config(
        cls,
        cfg: "Any",
        output_dir: "Path | None" = None,
        checkpoint_dir: "Path | None" = None,
    ) -> "SeedSweepConfig":
        """Build SeedSweepConfig from an ExperimentConfig.

        Maps ExperimentConfig fields to SeedSweepConfig fields.
        HP dict is built from cfg.model fields (excluding arch discriminator).
        output_dir and checkpoint_dir can be overridden — useful for smoke tests.
        """
        # model hyperparams dict: everything except the arch discriminator
        hp: dict[str, Any] = {
            k: v for k, v in cfg.model.__dict__.items() if k != "arch"
        }
        # include training HP that seed_sweep uses from hp dict
        hp["batch_size"] = cfg.train.batch_size
        hp["lr"] = cfg.train.lr
        hp["weight_decay"] = cfg.train.weight_decay
        hp["max_epochs"] = cfg.train.max_epochs
        hp["patience"] = cfg.train.patience
        # NOTE: max_grad_norm is deliberately NOT forwarded from TrainConfig here.
        # Doing so would carry a non-None clip into _train_torch_generic for xlstm
        # too (it shares that loop), changing the committed xlstm baseline. Grad
        # clip is instead gated by arch == "transformer" inside the loop, and the
        # warmup_steps knob arrives via cfg.model (transformer model-field). So the
        # transformer gets stabilisation; lstm/cnn_lstm/xlstm are untouched.

        return cls(
            model_type=cfg.model.arch,
            hyperparams=hp,
            seeds=list(cfg.train.seeds),
            loss_type=cfg.train.loss,
            focal_gamma=cfg.train.focal_gamma,
            output_dir=output_dir if output_dir is not None else cfg.runtime.output_dir,
            checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else cfg.runtime.checkpoint_dir,
            data_dir=cfg.data.data_dir,
            ablation_no_dropout=cfg.train.ablation_no_dropout,
            ablation_no_l2=cfg.train.ablation_no_l2,
        )


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

        if config.model_type in ("lstm", "cnn_lstm", "transformer", "xlstm"):
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
                # Two meta formats exist: seed_sweep writes a `test_per_class_f1`
                # array; train.py writes `test_bull_f1`/`test_bear_f1` as separate
                # keys. Handle both so cached train.py checkpoints don't silently
                # zero the minority-class columns.
                pcf = meta.get("test_per_class_f1")
                if pcf is None:
                    pcf = [
                        meta.get("test_none_f1", 0.0),
                        meta.get("test_bull_f1", 0.0),
                        meta.get("test_bear_f1", 0.0),
                    ]
                rows.append({
                    "seed": seed,
                    "macro_f1": meta["test_macro_f1"],
                    "none_f1": pcf[0],
                    "bull_f1": pcf[1],
                    "bear_f1": pcf[2],
                })
                continue

        print(f"  Seed {seed}: training {config.model_type} ({config.loss_type})...")

        if config.model_type == "lstm":
            result = _train_lstm(config, seed, ckpt_path, meta_path)
        elif config.model_type == "cnn_lstm":
            result = _train_cnn_lstm(config, seed, ckpt_path, meta_path)
        elif config.model_type in ("transformer", "xlstm"):
            result = _train_torch_generic(config, seed, ckpt_path, meta_path)
        elif config.model_type in ("xgb", "xgboost"):
            result = _train_xgb(config, seed, ckpt_path, meta_path)
        else:
            raise ValueError(
                f"Unknown model_type {config.model_type!r} in run_seed_sweep — "
                "no training branch. (Previously this fell through to XGBoost, "
                "which segfaults in-process on macOS arm64 after torch import.)"
            )

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
# CNN-LSTM training
# ---------------------------------------------------------------------------

def _train_cnn_lstm(
    config: SeedSweepConfig,
    seed: int,
    ckpt_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.models.cnn_lstm import FVGCNNLSTMClassifier
    from src.training.early_stop import EarlyStop
    from src.training.loss import FocalLoss, WeightedCE

    set_seed(seed)
    device = _get_device()

    train_df, val_df, test_df = _load_splits(config.data_dir)
    labeller = LABELLERS["fvg_valid"]()

    hp = config.hyperparams
    window_size = int(hp.get("window_size", 60))
    batch_size = int(hp.get("batch_size", 16))

    train_ds = SMCWindowDataset(train_df, labeller, stride=1, window_size=window_size,
                                drop_cross_session_windows=False)
    val_ds = SMCWindowDataset(val_df, labeller, stride=1, window_size=window_size,
                              drop_cross_session_windows=False)
    test_ds = SMCWindowDataset(test_df, labeller, stride=1, window_size=window_size,
                               drop_cross_session_windows=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    weights = _load_class_weights(config.data_dir)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    n_conv_layers = int(hp.get("n_conv_layers", 2))
    dropout_val = float(hp.get("dropout", 0.318)) if not config.ablation_no_dropout else 0.0
    head_dropout_val = float(hp.get("head_dropout", 0.526)) if not config.ablation_no_dropout else 0.0

    model = FVGCNNLSTMClassifier(
        conv_filters=int(hp.get("conv_filters", 32)),
        kernel_size=int(hp.get("kernel_size", 3)),
        n_conv_layers=n_conv_layers,
        use_pool=bool(hp.get("use_pool", False)),
        pool_type=str(hp.get("pool_type", "max")),
        lstm_hidden=int(hp.get("lstm_hidden", 64)),
        lstm_layers=int(hp.get("lstm_layers", 1)),
        dropout=dropout_val,
        head_dropout=head_dropout_val,
    ).to(device)

    if config.loss_type == "focal":
        criterion = FocalLoss(class_weights=weights_tensor, gamma=config.focal_gamma)
    else:
        criterion = WeightedCE(weights_tensor)

    lr = float(hp.get("lr", 5.3e-4))
    weight_decay = 0.0 if config.ablation_no_l2 else float(hp.get("weight_decay", 3.92e-5))
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

        val_f1, _, _ = _eval_f1_all(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if early_stop.update(val_f1):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_macro_f1, test_per_class_f1, (y_true, y_pred) = _eval_f1_all(model, test_loader, device)
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
        "arch": "cnn_lstm",
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
# Generic torch training (transformer, xlstm) — registry-built, CPU-only
# ---------------------------------------------------------------------------

# HP keys that are training-loop params, NOT model constructor args.
# warmup_steps / max_grad_norm / scheduler are stabilisation knobs consumed by the
# training loop below (gated, transformer-only — see _train_torch_generic) and must
# never reach a model ctor.
_NON_MODEL_HP = {
    "batch_size", "lr", "weight_decay", "window_size", "max_epochs", "patience",
    "warmup_steps", "max_grad_norm", "scheduler", "context_length",
}


def _train_torch_generic(
    config: SeedSweepConfig,
    seed: int,
    ckpt_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    """Train any registry torch model for one seed. Used for transformer + xlstm.

    Builds the model from src.config.registry.MODELS keyed by config.model_type,
    passing every hyperparam except training-loop keys (_NON_MODEL_HP) to the ctor.
    Mirrors the _train_cnn_lstm loop exactly: WeightedCE/Focal, Adam, early-stop(15),
    100-epoch cap, best-val checkpoint, single final test eval. CPU only.
    """
    from src.config.registry import MODELS  # triggers model registrations
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.training.early_stop import EarlyStop
    from src.training.loss import FocalLoss, WeightedCE

    arch = config.model_type
    if arch not in MODELS:
        raise KeyError(
            f"Arch {arch!r} not in model registry {sorted(MODELS)} — "
            "ensure it is decorated in src/config/_model_registrations.py"
        )

    set_seed(seed)
    device = _get_device()  # CPU (forced)

    train_df, val_df, test_df = _load_splits(config.data_dir)
    labeller = LABELLERS["fvg_valid"]()

    hp = config.hyperparams
    window_size = int(hp.get("window_size", 60))
    batch_size = int(hp.get("batch_size", 16))

    train_ds = SMCWindowDataset(train_df, labeller, stride=1, window_size=window_size,
                                drop_cross_session_windows=False)
    val_ds = SMCWindowDataset(val_df, labeller, stride=1, window_size=window_size,
                              drop_cross_session_windows=False)
    test_ds = SMCWindowDataset(test_df, labeller, stride=1, window_size=window_size,
                               drop_cross_session_windows=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    weights = _load_class_weights(config.data_dir)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # Model ctor args = all HP except training-loop keys. Apply dropout ablation.
    model_kwargs = {k: v for k, v in hp.items() if k not in _NON_MODEL_HP}
    if config.ablation_no_dropout:
        for dk in ("dropout", "head_dropout"):
            if dk in model_kwargs:
                model_kwargs[dk] = 0.0
    # xLSTM's sLSTM stack allocates internal state for context_length; it must
    # track the actual window size, not the ctor default (60). Inject explicitly
    # so a window_size sweep can't silently desync the stack from the data.
    if arch == "xlstm":
        model_kwargs["context_length"] = window_size
    # Keep only kwargs the model constructor accepts. Best-HP JSON from Optuna
    # carries metadata (val_macro_f1, trial_number, n_trials_completed,
    # study_name, storage) that must not reach the model ctor.
    import inspect
    _valid = set(inspect.signature(MODELS[arch].__init__).parameters) - {"self"}
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in _valid}
    model = MODELS[arch](**model_kwargs).to(device)

    if config.loss_type == "focal":
        criterion = FocalLoss(class_weights=weights_tensor, gamma=config.focal_gamma)
    else:
        criterion = WeightedCE(weights_tensor)

    lr = float(hp.get("lr", 1e-3))
    weight_decay = 0.0 if config.ablation_no_l2 else float(hp.get("weight_decay", 1e-4))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_epochs = int(hp.get("max_epochs", 100))
    patience = int(hp.get("patience", 15))
    early_stop = EarlyStop(patience=patience, mode="max")

    # ---- Stabilisation knobs (GATED — transformer fair-shot only) -----------
    # LR warmup + gradient clipping help Transformer optimisation but were NOT
    # used to produce the committed lstm/cnn_lstm baselines. They are gated OFF
    # by default so every non-transformer arch routed here is byte-for-byte
    # identical to the original loop:
    #   - warmup_steps defaults to 0  -> the warmup branch is skipped entirely
    #     (base lr is never rescaled), so optimiser.step() behaves as before.
    #   - max_grad_norm defaults to None for non-transformer arches -> no
    #     clip_grad_norm_ call is made.
    # Only the Transformer config sets warmup_steps > 0 and/or max_grad_norm, so
    # only it incurs the warmup ramp and the clip. lstm/cnn_lstm/xlstm leave both
    # unset -> their training is unchanged and their baselines stay valid.
    warmup_steps = int(hp.get("warmup_steps", 0))
    # Gradient clipping is transformer-only by intent. Gate by arch STRUCTURALLY
    # (not just by hp-dict cleanliness) so a stray max_grad_norm key in a
    # directly-constructed SeedSweepConfig can never silently clip xlstm and
    # invalidate its baseline.
    _default_clip = 1.0 if arch == "transformer" else None
    raw_max_grad_norm = hp.get("max_grad_norm", _default_clip)
    max_grad_norm = (
        float(raw_max_grad_norm)
        if (raw_max_grad_norm is not None and arch == "transformer")
        else None
    )
    base_lr = lr
    global_step = 0

    best_val_f1 = 0.0
    best_state: dict | None = None
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = torch.as_tensor(y_batch, device=device)
            optimiser.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            # Gated grad clip — only when max_grad_norm is set (transformer).
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Gated linear LR warmup — only when warmup_steps > 0 (transformer).
            if warmup_steps > 0 and global_step < warmup_steps:
                warm_lr = base_lr * float(global_step) / float(warmup_steps)
                for pg in optimiser.param_groups:
                    pg["lr"] = warm_lr
            elif warmup_steps > 0:
                for pg in optimiser.param_groups:
                    pg["lr"] = base_lr
            optimiser.step()
            global_step += 1

        val_f1, _, _ = _eval_f1_all(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if early_stop.update(val_f1):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_macro_f1, test_per_class_f1, (y_true, y_pred) = _eval_f1_all(model, test_loader, device)
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        **log_run_metadata(seed, device),
        "best_epoch": best_epoch,
        "max_epochs": max_epochs,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_macro_f1,
        "test_per_class_f1": test_per_class_f1,
        "hyperparams": config.hyperparams,
        "loss_type": config.loss_type,
        "focal_gamma": config.focal_gamma if config.loss_type == "focal" else None,
        "arch": arch,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
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
    # CPU-only: MPS gradient kernel is broken on Apple Silicon (torch 2.11) and
    # CUDA is absent on this platform. All torch training runs on CPU per CLAUDE.md.
    return torch.device("cpu")


def _load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(data_dir / "spy_h1_train.parquet")
    val = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test = pd.read_parquet(data_dir / "spy_h1_test.parquet")
    # Multi-symbol parquets must have same-symbol bars contiguous and sorted so
    # _window_generator's cross-symbol guard works correctly.  Parquet spec does
    # not guarantee row order on reload, so re-sort here using the same pattern
    # as _pool_timeseries in pipeline.py.  This is a NO-OP for single-symbol
    # data (no "symbol" column).
    def _sort_if_multisym(df: pd.DataFrame) -> pd.DataFrame:
        if "symbol" not in df.columns:
            return df
        return df.assign(_ts=df.index).sort_values(["symbol", "_ts"]).drop(columns="_ts")

    train = _sort_if_multisym(train)
    val = _sort_if_multisym(val)
    test = _sort_if_multisym(test)
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
    y_true, y_pred = result["y_true"], result["y_pred"]
    # Bootstrap CI requires y_true/y_pred to be the same length (and the full
    # test set). A mismatch here would silently corrupt CIs downstream (numpy
    # broadcast / pandas align), so fail loud at write time.
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{config.model_type} seed {seed}: pred length mismatch "
            f"y_true={len(y_true)} vs y_pred={len(y_pred)} — would corrupt bootstrap CI."
        )
    np.savez(npz_path, y_true=y_true, y_pred=y_pred)
