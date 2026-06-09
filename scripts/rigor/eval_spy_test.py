"""eval_spy_test.py — Post-training SPY-only-test evaluation for multi-symbol models.

Loads each per-seed checkpoint trained on pooled multi-symbol data and scores it
on the FIXED SPY-only test set (data/processed/spy_h1_test.parquet).  This gives
the apples-to-apples headline metric vs SPY-only baselines:
  CNN-LSTM 0.639, LSTM 0.595, tuned Transformer 0.601, XGB 0.721.

Usage:
    python scripts/rigor/eval_spy_test.py [--models cnn_lstm lstm transformer xgb]

Design notes:
  - Reuses src.rigor.seed_sweep._eval_f1_all (line 718) and helpers so F1 numbers
    are computed identically to training (same averaging, same class order
    none=0 / bull=1 / bear=2).
  - DL inference: CPU-only (MPS gradient kernel bug on Apple Silicon, torch 2.11).
  - XGB predict-only: in-process load+predict is safe on macOS arm64 — the
    segfault documented in CLAUDE.md only triggers on fit() after torch import.
    predict() on a pre-loaded model does not call fit() so no subprocess needed.
  - Missing checkpoints are skipped with a warning; script evaluates whatever
    seeds exist and reports seeds_found / total.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse eval helper from seed_sweep — keeps F1 computation byte-for-byte identical.
# _eval_f1_all is a private helper; importing it here is intentional (internal
# rigor script, same package). See src/rigor/seed_sweep.py line 718.
from src.rigor.seed_sweep import _eval_f1_all  # noqa: PLC2701

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [0, 17, 42, 123, 2024]
DATA_DIR = ROOT / "data" / "processed"
SPY_TEST_PATH = DATA_DIR / "spy_h1_test.parquet"

# SPY-only baselines (5-seed runs, committed results from CLAUDE.md / models-status.md)
BASELINES: dict[str, float] = {
    "cnn_lstm": 0.639,
    "lstm": 0.595,
    "transformer": 0.601,
    "xgb": 0.721,
}

# Model config and checkpoint locations for multi-symbol runs
MODEL_META: dict[str, dict[str, Any]] = {
    "cnn_lstm": {
        "arch": "cnn_lstm",
        "yaml": ROOT / "experiments" / "cnn_lstm_multisym.yaml",
        "ckpt_dir": ROOT / "checkpoints" / "cnn_lstm_multisym" / "cnn_lstm",
        "ckpt_pattern": "cnn_lstm_seed{seed}.pt",
    },
    "lstm": {
        "arch": "lstm",
        "yaml": ROOT / "experiments" / "lstm_multisym.yaml",
        "ckpt_dir": ROOT / "checkpoints" / "lstm_multisym" / "lstm",
        "ckpt_pattern": "lstm_seed{seed}.pt",
    },
    "transformer": {
        "arch": "transformer",
        "yaml": ROOT / "experiments" / "transformer_multisym.yaml",
        "ckpt_dir": ROOT / "checkpoints" / "transformer_multisym" / "transformer",
        "ckpt_pattern": "transformer_seed{seed}.pt",
    },
    "xgb": {
        "arch": "xgb",
        "yaml": ROOT / "experiments" / "xgb_multisym.yaml",
        "ckpt_dir": ROOT / "checkpoints" / "xgb_multisym" / "xgboost",
        "ckpt_pattern": "xgb_seed{seed}.ubj",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_spy_test_df() -> pd.DataFrame:
    """Load the SPY-only test parquet (no symbol column)."""
    if not SPY_TEST_PATH.exists():
        raise FileNotFoundError(
            f"SPY test parquet not found: {SPY_TEST_PATH}\n"
            "Run `python -c 'from src.data.pipeline import build_pipeline; build_pipeline()'` first."
        )
    df = pd.read_parquet(SPY_TEST_PATH)
    log.info("SPY test parquet loaded: %d rows, cols=%s", len(df), list(df.columns))
    return df


def _load_class_weights() -> list[float]:
    weights_path = DATA_DIR / "class_weights.json"
    with weights_path.open() as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [float(w) for w in data]
    return [float(data[str(i)]) for i in range(3)]


def _build_dl_test_loader(test_df: pd.DataFrame, window_size: int = 60) -> DataLoader:
    """Build a DataLoader over the SPY-only test split."""
    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset

    labeller = LABELLERS["fvg_valid"]()
    test_ds = SMCWindowDataset(
        test_df, labeller, stride=1, window_size=window_size,
        drop_cross_session_windows=False,
    )
    return DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)


def _build_dl_model(arch: str, hp: dict[str, Any]) -> torch.nn.Module:
    """Instantiate a DL model from the registry, filtering to ctor-accepted kwargs."""
    import inspect
    from src.config.registry import MODELS  # triggers model registrations

    if arch not in MODELS:
        raise KeyError(f"Arch {arch!r} not in MODELS registry {sorted(MODELS)}")

    _NON_MODEL_HP = {
        "batch_size", "lr", "weight_decay", "max_epochs", "patience",
        "warmup_steps", "max_grad_norm", "window_size",
    }
    model_kwargs = {k: v for k, v in hp.items() if k not in _NON_MODEL_HP}
    valid = set(inspect.signature(MODELS[arch].__init__).parameters) - {"self"}
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid}
    return MODELS[arch](**model_kwargs)


def _eval_dl_seed(
    arch: str,
    ckpt_path: Path,
    hp: dict[str, Any],
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Load one DL checkpoint and score it on the pre-built test loader.

    Returns (metrics_dict, y_true, y_pred) so callers can save preds for bootstrap CI.
    """
    model = _build_dl_model(arch, hp)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    macro_f1, per_class_f1, (y_true, y_pred) = _eval_f1_all(model, test_loader, device)
    metrics = {
        "macro_f1": macro_f1,
        "none_f1": per_class_f1[0],
        "bull_f1": per_class_f1[1],
        "bear_f1": per_class_f1[2],
    }
    return metrics, np.array(y_true), np.array(y_pred)


def _eval_xgb_all_seeds(
    ckpt_dir: Path,
    pattern: str,
    seeds: list[int],
) -> list[dict]:
    """Evaluate all XGB seeds via a torch-free subprocess worker.

    XGB predict() segfaults on macOS arm64 after torch has been imported in the
    parent process — same root cause as fit().  We isolate the XGB call in a
    fresh subprocess (_xgb_eval_spy_worker.py) that never imports torch.

    Returns a list of per-seed result dicts (same schema as _eval_dl_seed).
    """
    import subprocess

    worker = Path(__file__).parent / "_workers" / "_xgb_eval_spy_worker.py"
    ckpt_paths = [str(ckpt_dir / pattern.format(seed=s)) for s in seeds]
    payload = {
        "ckpt_paths": ckpt_paths,
        "spy_test_parquet": str(SPY_TEST_PATH),
        "seeds": seeds,
    }
    proc = subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"XGB eval worker failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    if proc.stderr.strip():
        log.debug("XGB worker stderr: %s", proc.stderr.strip())
    return json.loads(proc.stdout)


def _load_hp_from_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load experiment YAML and return a merged HP dict (model + train fields)."""
    from src.config.loader import load_experiment

    cfg = load_experiment(yaml_path)
    hp: dict[str, Any] = {k: v for k, v in cfg.model.__dict__.items() if k != "arch"}
    hp["batch_size"] = cfg.train.batch_size
    hp["lr"] = cfg.train.lr
    hp["weight_decay"] = cfg.train.weight_decay
    hp["max_epochs"] = cfg.train.max_epochs
    hp["patience"] = cfg.train.patience
    hp["window_size"] = cfg.data.window_size
    if hasattr(cfg.train, "max_grad_norm") and cfg.train.max_grad_norm is not None:
        hp["max_grad_norm"] = cfg.train.max_grad_norm
    return hp


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def eval_model(
    model_name: str,
    test_df: pd.DataFrame,
    preds_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all available seeds for one model on SPY-only test set.

    Args:
        model_name: Key in MODEL_META.
        test_df: SPY-only test DataFrame.
        preds_dir: If given, per-seed <model>_seed<N>_spytest_preds.npz files
                   (keys y_true, y_pred) are written here for bootstrap CI.

    Returns a dict with per-seed results + aggregate mean/std.
    """
    meta = MODEL_META[model_name]
    arch = meta["arch"]
    ckpt_dir: Path = meta["ckpt_dir"]
    pattern: str = meta["ckpt_pattern"]
    device = torch.device("cpu")

    seed_results: list[dict[str, Any]] = []
    missing: list[int] = []

    # Load HP from YAML once (DL) or as needed (XGB)
    hp: dict[str, Any] = {}
    test_loader = None

    if arch != "xgb":
        hp = _load_hp_from_yaml(meta["yaml"])
        window_size = int(hp.get("window_size", 60))
        log.info("[%s] Building SPY test loader (window_size=%d)...", model_name, window_size)
        test_loader = _build_dl_test_loader(test_df, window_size=window_size)

    if arch == "xgb":
        # XGB: batch subprocess eval (torch-free worker) — avoids macOS arm64
        # segfault that occurs when predict() is called after torch import.
        log.info("[xgb] Running all seeds via subprocess worker...")
        try:
            worker_results = _eval_xgb_all_seeds(ckpt_dir, pattern, SEEDS)
        except Exception as exc:
            log.error("[xgb] Subprocess worker failed: %s", exc)
            worker_results = []

        for wr in worker_results:
            seed = wr["seed"]
            if wr.get("missing"):
                log.warning("[xgb] Seed %s checkpoint not found — skipping.", seed)
                missing.append(seed)
            elif "error" in wr:
                log.error("[xgb] Seed %s eval error: %s", seed, wr["error"])
                missing.append(seed)
            else:
                result = {k: wr[k] for k in ("macro_f1", "none_f1", "bull_f1", "bear_f1")}
                result["seed"] = seed
                seed_results.append(result)
                log.info(
                    "[xgb] Seed %s — macro_f1=%.4f  bull_f1=%.4f  bear_f1=%.4f",
                    seed, result["macro_f1"], result["bull_f1"], result["bear_f1"],
                )
                # Save per-seed preds for bootstrap CI
                if preds_dir is not None and "y_true" in wr:
                    preds_dir.mkdir(parents=True, exist_ok=True)
                    npz_name = f"xgb_seed{seed}_spytest_preds.npz"
                    np.savez(
                        preds_dir / npz_name,
                        y_true=np.array(wr["y_true"]),
                        y_pred=np.array(wr["y_pred"]),
                    )
                    log.info("[xgb] Preds saved: %s", npz_name)
    else:
        for seed in SEEDS:
            ckpt_name = pattern.format(seed=seed)
            ckpt_path = ckpt_dir / ckpt_name

            if not ckpt_path.exists():
                log.warning("[%s] Seed %s checkpoint not found — skipping: %s", model_name, seed, ckpt_path)
                missing.append(seed)
                continue

            log.info("[%s] Evaluating seed %s from %s", model_name, seed, ckpt_path.name)
            try:
                assert test_loader is not None
                result, y_true, y_pred = _eval_dl_seed(arch, ckpt_path, hp, test_loader, device)
            except Exception as exc:
                log.error("[%s] Seed %s eval failed: %s", model_name, seed, exc)
                missing.append(seed)
                continue

            result["seed"] = seed
            seed_results.append(result)

            # Save per-seed preds for bootstrap CI (bootstrap_ci_multiseed.py contract)
            if preds_dir is not None:
                preds_dir.mkdir(parents=True, exist_ok=True)
                npz_name = f"{arch}_seed{seed}_spytest_preds.npz"
                np.savez(preds_dir / npz_name, y_true=y_true, y_pred=y_pred)
                log.info("[%s] Preds saved: %s", model_name, npz_name)
            log.info(
                "[%s] Seed %s — macro_f1=%.4f  bull_f1=%.4f  bear_f1=%.4f",
                model_name, seed,
                result["macro_f1"], result["bull_f1"], result["bear_f1"],
            )

    seeds_found = len(seed_results)
    agg: dict[str, Any] = {}
    if seed_results:
        for key in ("macro_f1", "none_f1", "bull_f1", "bear_f1"):
            vals = [r[key] for r in seed_results]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals, ddof=0))
    else:
        log.warning("[%s] No seeds evaluated — all checkpoints missing.", model_name)

    return {
        "model": model_name,
        "arch": arch,
        "seeds_found": seeds_found,
        "seeds_total": len(SEEDS),
        "seeds_missing": missing,
        "per_seed": seed_results,
        "aggregate": agg,
        "baseline_macro": BASELINES.get(model_name),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'model':<12}  {'seeds':>6}  "
        f"{'macro_f1':>14}  {'bull_f1':>14}  {'bear_f1':>14}  {'baseline':>9}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print("  SPY-only-test evaluation — multi-symbol checkpoints")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        agg = r["aggregate"]
        if not agg:
            row = (
                f"{r['model']:<12}  {r['seeds_found']:>3}/{r['seeds_total']:<2}  "
                f"{'n/a':>14}  {'n/a':>14}  {'n/a':>14}  "
                f"{r['baseline_macro'] or 'n/a':>9}"
            )
        else:
            macro = f"{agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}"
            bull = f"{agg['bull_f1_mean']:.4f}±{agg['bull_f1_std']:.4f}"
            bear = f"{agg['bear_f1_mean']:.4f}±{agg['bear_f1_std']:.4f}"
            row = (
                f"{r['model']:<12}  {r['seeds_found']:>3}/{r['seeds_total']:<2}  "
                f"{macro:>14}  {bull:>14}  {bear:>14}  "
                f"{r['baseline_macro'] or 'n/a':>9.3f}"
            )
        print(row)
    print(sep)
    print()


def _save_json(results: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Results written to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-symbol model checkpoints on SPY-only test set."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_META),
        default=list(MODEL_META),
        help="Which models to evaluate (default: all four).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "rigor" / "09-Jun-26" / "spy_test_eval.json",
        help="Path for JSON results output.",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=ROOT / "reports" / "rigor" / "09-Jun-26" / "spytest_preds",
        help="Directory to write per-seed *_spytest_preds.npz for bootstrap CI.",
    )
    args = parser.parse_args()

    if not SPY_TEST_PATH.exists():
        log.error("SPY test parquet missing: %s", SPY_TEST_PATH)
        sys.exit(1)

    log.info("Loading SPY-only test set from %s", SPY_TEST_PATH)
    test_df = _load_spy_test_df()

    preds_dir: Path = args.preds_dir
    log.info("Per-seed preds will be written to %s", preds_dir)

    all_results: list[dict[str, Any]] = []
    for model_name in args.models:
        log.info("=" * 60)
        log.info("Evaluating model: %s", model_name)
        result = eval_model(model_name, test_df, preds_dir=preds_dir)
        all_results.append(result)

    _print_table(all_results)
    _save_json(all_results, args.output)
    log.info("Done.")


if __name__ == "__main__":
    main()
