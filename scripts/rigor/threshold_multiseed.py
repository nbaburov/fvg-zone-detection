"""threshold_multiseed.py — Per-seed threshold tuning, aggregated to G4 output (Gap 4).

Runs threshold_sweep logic for each seed checkpoint (LSTM or CNN-LSTM).
Aggregates delta macro F1 across seeds.
Writes G4/threshold_tuning.json.

GUARD: threshold search uses val set only. Test set accessed only for final measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _get_probas(model_path: Path, df, labeller, device, model_type: str, window_size: int = 60):
    """Load checkpoint and get softmax probabilities for a dataframe."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.window import SMCWindowDataset

    ds = SMCWindowDataset(df, labeller, stride=1, window_size=window_size, drop_cross_session_windows=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    state = torch.load(model_path, map_location="cpu", weights_only=True)

    # Read HP from meta sidecar (avoids fragile state-dict key introspection)
    meta_path = model_path.with_suffix(".meta.json")
    hp: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        hp = meta.get("hyperparams", {})
        if not hp:
            # Older base-run meta stored HP at top level
            for k in ("hidden_size", "num_layers", "conv_filters", "kernel_size",
                      "n_conv_layers", "use_pool", "lstm_hidden", "lstm_layers",
                      "dropout", "head_dropout"):
                if k in meta:
                    hp[k] = meta[k]

    if model_type == "lstm":
        from src.models.lstm import FVGLSTMClassifier
        model = FVGLSTMClassifier(
            hidden_size=int(hp.get("hidden_size", 64)),
            num_layers=int(hp.get("num_layers", 2)),
            dropout=float(hp.get("dropout", 0.3)),
            head_dropout=float(hp.get("head_dropout", 0.5)),
        )
    elif model_type == "cnn_lstm":
        from src.models.cnn_lstm import FVGCNNLSTMClassifier
        model = FVGCNNLSTMClassifier(
            conv_filters=int(hp.get("conv_filters", 32)),
            kernel_size=int(hp.get("kernel_size", 3)),
            n_conv_layers=int(hp.get("n_conv_layers", 2)),
            use_pool=bool(hp.get("use_pool", False)),
            pool_type=str(hp.get("pool_type", "max")),
            lstm_hidden=int(hp.get("lstm_hidden", 64)),
            lstm_layers=int(hp.get("lstm_layers", 1)),
            dropout=float(hp.get("dropout", 0.318)),
            head_dropout=float(hp.get("head_dropout", 0.526)),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    probas, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probas.append(p)
            labels.append(y.numpy())

    return np.vstack(probas), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "cnn_lstm"],
                        help="Model type to evaluate (default: lstm)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to experiments/foo.yaml (optional — sets data_dir, seeds, checkpoint_dir)")
    parser.add_argument("--set", dest="set_overrides", nargs="+", default=[],
                        metavar="key=value",
                        help='Override config fields: --set "train.seeds=[42]"')
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    # Legacy seeds flag
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="[Legacy] seed list. Use --set 'train.seeds=[...]' with YAML config.")
    args = parser.parse_args()

    model_type = args.model

    # Resolve config (optional)
    cfg = None
    if args.config is not None:
        import warnings
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        from src.config.loader import load_experiment, parse_set_args
        overrides = parse_set_args(args.set_overrides)
        if args.seeds is not None:
            warnings.warn(
                "--seeds is deprecated when using YAML config. "
                "Use --set 'train.seeds=[...]' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            overrides.setdefault("train.seeds", args.seeds)
        cfg = load_experiment(config_path, overrides or None)

    default_ckpt_subdir = model_type  # "lstm" or "cnn_lstm"
    checkpoint_dir = args.checkpoint_dir or (
        Path(cfg.runtime.checkpoint_dir) / default_ckpt_subdir if cfg else Path(f"checkpoints/{default_ckpt_subdir}")
    )
    data_dir_path = args.data_dir or (cfg.data.data_dir if cfg else Path("data/processed"))
    seeds = (cfg.train.seeds if cfg else None) or args.seeds or [0, 17, 42, 123, 2024]
    window_size = int(cfg.data.window_size) if cfg else 60

    ckpt_dir = ROOT / checkpoint_dir
    data_dir = ROOT / data_dir_path
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    from sklearn.metrics import f1_score

    from src.data.labels import LABELLERS
    from src.rigor.threshold import (
        apply_thresholds,
        compute_pr_curves,
        find_f1_optimal_threshold,
    )

    device = torch.device("cpu")
    labeller = LABELLERS["fvg_valid"]()  # instantiate the labeller

    # Load val + test data once (guard: val for threshold search, test for final measurement only)
    print("Loading val and test parquet...")
    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")

    per_seed: dict[str, dict] = {}
    deltas: list[float] = []

    for seed in seeds:
        ckpt = ckpt_dir / f"{model_type}_seed{seed}.pt"
        if not ckpt.exists():
            print(f"WARNING: checkpoint not found: {ckpt} — skipping seed {seed}")
            continue

        print(f"\n--- Seed {seed} ---")

        # Get val probabilities (threshold search — val only, no test leakage)
        print("  Getting val probas...")
        val_proba, val_true = _get_probas(ckpt, val_df, labeller, device, model_type, window_size)

        # Get test probabilities (final measurement only, after thresholds are fixed)
        print("  Getting test probas...")
        test_proba, test_true = _get_probas(ckpt, test_df, labeller, device, model_type, window_size)

        # Threshold optimisation on val ONLY
        curves = compute_pr_curves(val_true, val_proba)
        optimal_thresholds: dict[int, float] = {}
        for cls in range(3):
            prec, rec, thr = curves[cls]
            opt_thr, opt_f1 = find_f1_optimal_threshold(prec, rec, thr)
            optimal_thresholds[cls] = opt_thr
            name = {0: "none", 1: "bull", 2: "bear"}[cls]
            print(f"  Class {cls} ({name}): threshold={opt_thr:.3f}, val F1={opt_f1:.4f}")

        # Apply thresholds to test (final measurement — executed exactly once, after threshold fixed)
        argmax_pred = test_proba.argmax(axis=1)
        tuned_pred = apply_thresholds(test_proba, optimal_thresholds)

        argmax_f1 = float(f1_score(test_true, argmax_pred, average="macro", zero_division=0.0))
        tuned_f1 = float(f1_score(test_true, tuned_pred, average="macro", zero_division=0.0))
        delta = tuned_f1 - argmax_f1

        print(f"  Argmax test macro F1: {argmax_f1:.4f}")
        print(f"  Tuned  test macro F1: {tuned_f1:.4f}")
        print(f"  Delta:                {delta:+.4f}")

        per_seed[str(seed)] = {
            "argmax_macro_f1": argmax_f1,
            "tuned_macro_f1": tuned_f1,
            "delta_macro_f1": delta,
            "thresholds": {str(c): float(t) for c, t in optimal_thresholds.items()},
        }
        deltas.append(delta)

    if not deltas:
        print("ERROR: no seeds processed")
        sys.exit(1)

    result = {
        "per_seed": per_seed,
        "mean_delta_macro_f1": float(np.mean(deltas)),
        "std_delta_macro_f1": float(np.std(deltas)),
        "n_seeds": len(deltas),
    }

    out_path = out_dir / "threshold_tuning.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== G4 Summary ===")
    print(f"Mean delta macro F1: {result['mean_delta_macro_f1']:+.4f} ± {result['std_delta_macro_f1']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
