"""train_utils.py — Shared utilities for Phase 4 training.

Functions:
  set_seed(seed)          — deterministic seeds (no deterministic mode on MPS: 8x slowdown)
  log_run_metadata(...)   — git sha, torch/python versions, device, timestamp
  eval_epoch(...)         — one evaluation pass, returns F1 metrics + confusion data
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.training.loss import WeightedCE


def set_seed(seed: int) -> None:
    """Set random, numpy, torch seeds for reproducibility.

    Does NOT enable torch.use_deterministic_algorithms(True) — 8x slowdown on MPS per CLAUDE.md.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Note: MPS has limited determinism support — seed helps but is not fully deterministic


def log_run_metadata(seed: int, device: torch.device) -> dict[str, Any]:
    """Return dict with run metadata for reproducibility logging."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    return {
        "git_sha": git_sha,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "device": str(device),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: WeightedCE,
    device: torch.device,
) -> dict[str, Any]:
    """Run one full evaluation pass.

    Returns:
        {
            "loss": float,
            "macro_f1": float,
            "per_class_f1": list[float],   # [none_f1, bull_f1, bear_f1]
            "y_true": np.ndarray,
            "y_pred": np.ndarray,
        }
    """
    model.eval()
    all_loss: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            if isinstance(y_batch, torch.Tensor):
                y_batch = y_batch.to(device)
            else:
                y_batch = torch.tensor(y_batch, device=device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            all_loss.append(loss.item())

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_y_pred.extend(preds.tolist())
            all_y_true.extend(y_batch.cpu().numpy().tolist())

    y_true = np.array(all_y_true, dtype=np.int64)
    y_pred = np.array(all_y_pred, dtype=np.int64)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    per_class_f1 = [
        float(v) for v in f1_score(y_true, y_pred, average=None, zero_division=0.0)
    ]

    return {
        "loss": float(np.mean(all_loss)),
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }
