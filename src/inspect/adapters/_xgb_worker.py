"""_xgb_worker.py — Subprocess worker for XGBoost inference.

This module is executed as __main__ in a spawned subprocess to isolate
XGBoost from PyTorch on Python 3.14+ where loading both segfaults.

Protocol (stdio-based, binary):
  stdin  <- 4-byte little-endian int32 N  (number of windows)
           + N*60*5*4 bytes float32 raw OHLCV windows (C-order)
  stdout -> N*3*4 bytes float32 probabilities (C-order)
  stderr -> error message on failure

The subprocess receives the checkpoint path as argv[1].
"""

import sys
import struct
import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: _xgb_worker.py <checkpoint_path>\n")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    # Import xgboost here (before torch is ever imported in this process)
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from src.models.xgboost_baseline import XGBoostFVGClassifier
    from src.features.window_features import _compute_features

    clf = XGBoostFVGClassifier.load(checkpoint_path)

    # Read N from stdin
    n_bytes = sys.stdin.buffer.read(4)
    if len(n_bytes) < 4:
        sys.stderr.write("Failed to read N\n")
        sys.exit(1)
    n = struct.unpack("<i", n_bytes)[0]

    if n == 0:
        # Write empty result
        sys.stdout.buffer.write(b"")
        sys.stdout.buffer.flush()
        return

    # Read windows
    total_floats = n * 60 * 5
    raw = sys.stdin.buffer.read(total_floats * 4)
    windows = np.frombuffer(raw, dtype=np.float32).reshape(n, 60, 5).copy()

    # Compute features
    X = np.zeros((n, 35), dtype=np.float32)
    for j in range(n):
        w = windows[j]
        o = w[:, 0].astype(np.float64)
        h = w[:, 1].astype(np.float64)
        lo = w[:, 2].astype(np.float64)
        c = w[:, 3].astype(np.float64)
        v = w[:, 4].astype(np.float64)
        X[j] = _compute_features(c, h, lo, o, v)

    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    probas = clf.predict_proba(X).astype(np.float32)  # (n, 3)
    sys.stdout.buffer.write(probas.tobytes())
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
