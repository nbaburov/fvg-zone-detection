"""xgboost_adapter.py — ModelAdapter wrapping XGBoostFVGClassifier.

Python 3.14+ note: loading xgboost model files after PyTorch is imported
causes a segfault (incompatible libstdc++ / jemalloc interaction). On Python
3.14+, this adapter automatically routes inference through a spawned subprocess
that imports XGBoost before PyTorch. On Python 3.12 and below, inference runs
in-process as normal.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from src.inspect.base import ModelAdapter

_DEFAULT_CHECKPOINT = "xgb_seed42.ubj"

# Warn if windows look normalised (likely wrong input for XGBoost)
_NORM_RANGE_THRESHOLD = 5.0

# Force subprocess isolation unconditionally — on Apple Silicon (arm64) xgboost
# segfaults when loaded in-process after PyTorch has been imported (libgomp clash).
# Subprocess avoids the issue at the cost of ~0.5 s process-startup overhead per batch.
_NEED_SUBPROCESS = True

_WORKER_PATH = Path(__file__).resolve().parent / "_xgb_worker.py"


class XGBoostAdapter(ModelAdapter):
    """Wraps a trained XGBoostFVGClassifier for inspection inference.

    XGBoost operates on RAW OHLCV windows (not normalised). The runner passes
    raw windows to this adapter. If normalised windows are accidentally passed
    (values outside [-5, 5] heuristic check), a warning is issued.

    ``extract_window_features`` is called internally to convert (N, 60, 5)
    raw windows into the 35-feature tabular matrix used at training time.

    On Python 3.14+: inference is routed through a subprocess to avoid the
    xgboost+torch segfault. Subprocess is spawned fresh per predict_proba call
    (stateless, clean process). Each call incurs ~0.5 s overhead for process
    startup, acceptable for the inspection use case.
    """

    name = "xgboost"

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_file: str = _DEFAULT_CHECKPOINT,
        checkpoint_path: Path | None = None,
        **_kwargs,
    ) -> None:
        if checkpoint_path is not None:
            self._checkpoint_path = Path(checkpoint_path)
        else:
            checkpoint_dir = Path(checkpoint_dir)
            self._checkpoint_path = checkpoint_dir / "xgboost" / checkpoint_file
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"XGBoost checkpoint not found: {self._checkpoint_path}"
            )

        if not _NEED_SUBPROCESS:
            # In-process: load model now and verify class ordering
            from src.models.xgboost_baseline import XGBoostFVGClassifier
            self._clf = XGBoostFVGClassifier.load(self._checkpoint_path)
            classes = list(self._clf._model.classes_)
            if classes != [0, 1, 2]:
                raise ValueError(
                    f"XGBoost model classes_ = {classes}; expected [0, 1, 2]. "
                    "Re-train the model with encoded labels {0, 1, 2}."
                )
        else:
            self._clf = None
            if not _WORKER_PATH.exists():
                raise FileNotFoundError(
                    f"XGBoost subprocess worker not found: {_WORKER_PATH}. "
                    "Ensure _xgb_worker.py is present in src/inspect/adapters/."
                )

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Run XGBoost inference on raw OHLCV windows.

        Parameters
        ----------
        windows : np.ndarray, shape (N, 60, 5), dtype float32
            RAW (un-normalised) OHLCV windows.

        Returns
        -------
        np.ndarray, shape (N, 3), dtype float32
            Class probabilities [none, bullish, bearish].
        """
        if windows.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        self._check_normalised_input(windows)

        if _NEED_SUBPROCESS:
            return self._predict_via_subprocess(windows)
        return self._predict_inprocess(windows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_normalised_input(self, windows: np.ndarray) -> None:
        ohlc_vals = windows[:, :, :4]
        if ohlc_vals.size > 0:
            max_abs = float(np.abs(ohlc_vals).max())
            if max_abs < _NORM_RANGE_THRESHOLD:
                warnings.warn(
                    f"XGBoostAdapter received windows with max |OHLC| = {max_abs:.3f}. "
                    "This looks like normalised input. XGBoost expects raw OHLCV. "
                    "Predictions may be wrong.",
                    stacklevel=3,
                )

    def _predict_inprocess(self, windows: np.ndarray) -> np.ndarray:
        X = _windows_to_features(windows)
        probas = self._clf.predict_proba(X)
        return probas.astype(np.float32)

    def _predict_via_subprocess(self, windows: np.ndarray) -> np.ndarray:
        """Route inference through a subprocess to avoid the xgboost+torch segfault."""
        n = windows.shape[0]
        windows_f32 = windows.astype(np.float32)

        # Encode: 4-byte N + flat float32 array
        payload = struct.pack("<i", n) + windows_f32.tobytes()

        result = subprocess.run(
            [sys.executable, str(_WORKER_PATH), str(self._checkpoint_path)],
            input=payload,
            capture_output=True,
            timeout=120,
        )

        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"XGBoost subprocess worker failed (exit {result.returncode}): {stderr_msg}"
            )

        # Decode output: n*3 float32 values
        raw_out = result.stdout
        expected_bytes = n * 3 * 4
        if len(raw_out) != expected_bytes:
            raise RuntimeError(
                f"XGBoost worker returned {len(raw_out)} bytes, expected {expected_bytes}."
            )

        probas = np.frombuffer(raw_out, dtype=np.float32).reshape(n, 3).copy()
        return probas


def _windows_to_features(windows: np.ndarray) -> np.ndarray:
    """Convert (N, 60, 5) raw windows to (N, 35) feature matrix."""
    from src.features.window_features import _compute_features  # type: ignore[reportPrivateUsage]

    n = windows.shape[0]
    X = np.zeros((n, 35), dtype=np.float32)

    for j in range(n):
        w = windows[j]  # (60, 5)
        o = w[:, 0].astype(np.float64)
        h = w[:, 1].astype(np.float64)
        lo = w[:, 2].astype(np.float64)
        c = w[:, 3].astype(np.float64)
        v = w[:, 4].astype(np.float64)
        X[j] = _compute_features(c, h, lo, o, v)

    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return X
