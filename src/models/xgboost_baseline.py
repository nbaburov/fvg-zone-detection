"""xgboost_baseline.py — XGBoost FVG 3-class classifier wrapper.

Thin wrapper around XGBClassifier providing:
  - Default hyperparameters for SPY H1 FVG classification
  - fit() with optional sample_weight + early stopping on val set
  - predict() / predict_proba() pass-through
  - save() / load() with metadata sidecar (.ubj + .ubj.meta.json)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


class XGBoostFVGClassifier:
    """XGBClassifier wrapper for FVG 3-class prediction (none / bull / bear)."""

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": ["mlogloss", "merror"],
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 30,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        # early_stopping_rounds is a constructor param in XGBoost 3.x
        self._early_stopping_rounds: int = merged.get("early_stopping_rounds", 30)
        self._params = merged
        self._model = xgb.XGBClassifier(**self._params)
        self._n_estimators_used: int | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "XGBoostFVGClassifier":
        """Train with early stopping on val mlogloss."""
        self._model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Record actual rounds used
        self._n_estimators_used = self._model.best_iteration + 1 if hasattr(self._model, "best_iteration") and self._model.best_iteration is not None else self._params["n_estimators"]
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions in {0, 1, 2}."""
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix (n_samples, 3). Rows sum to 1."""
        return self._model.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save model to `path` (.ubj) and sidecar `<path>.meta.json`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

        meta = {
            "params": self._params,
            "early_stopping_rounds": self._early_stopping_rounds,
            "n_estimators_used": self._n_estimators_used,
            "git_sha": _git_sha(),
            "python_version": sys.version,
            "xgboost_version": xgb.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
        }
        meta_path = Path(str(path) + ".meta.json")
        with meta_path.open("w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "XGBoostFVGClassifier":
        """Load model from .ubj file. Reads sidecar for params if available."""
        path = Path(path)
        meta_path = Path(str(path) + ".meta.json")

        instance = cls.__new__(cls)
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(str(path))

        if meta_path.exists():
            with meta_path.open() as fh:
                meta = json.load(fh)
            instance._params = meta.get("params", {})
            instance._early_stopping_rounds = meta.get("early_stopping_rounds", 30)
            instance._n_estimators_used = meta.get("n_estimators_used")
        else:
            instance._params = dict(cls.DEFAULT_PARAMS)
            instance._early_stopping_rounds = 30
            instance._n_estimators_used = None

        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_estimators_used(self) -> int | None:
        return self._n_estimators_used

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def evals_result(self) -> dict:
        return self._model.evals_result()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"
