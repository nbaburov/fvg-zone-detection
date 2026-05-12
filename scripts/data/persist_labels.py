"""persist_labels.py — Ensure spy_h1_labeled.parquet is current.

Copies spy_h1.parquet → spy_h1_labeled.parquet if:
  - spy_h1_labeled.parquet does not exist, OR
  - spy_h1_labeled.parquet is older than src/data/labels/valid_fvg.py

Idempotent: running twice produces the same result.

Usage:
    python scripts/data/persist_labels.py

Exit codes:
    0 — success (created or confirmed up-to-date)
    1 — source parquet missing
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "data" / "processed" / "spy_h1.parquet"
DST = REPO / "data" / "processed" / "spy_h1_labeled.parquet"
LABELLER_SRC = REPO / "src" / "data" / "labels" / "valid_fvg.py"


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source parquet not found: {SRC}", file=sys.stderr)
        return 1

    # Staleness check
    if DST.exists():
        dst_mtime = DST.stat().st_mtime
        src_mtime = SRC.stat().st_mtime
        labeller_mtime = LABELLER_SRC.stat().st_mtime if LABELLER_SRC.exists() else 0.0
        newest_src = max(src_mtime, labeller_mtime)
        if dst_mtime >= newest_src:
            print("spy_h1_labeled.parquet is up-to-date, skipping.")
            _print_stats(DST)
            return 0

    print(f"Copying {SRC.name} → {DST.name} ...")
    shutil.copy2(str(SRC), str(DST))
    # shutil.copy2 preserves src mtime — force DST mtime to NOW so idempotency check
    # (`dst_mtime >= newest_src`) reliably skips on re-run.
    now = time.time()
    os.utime(DST, (now, now))
    print("Done.")
    _print_stats(DST)
    return 0


def _print_stats(path: Path) -> None:
    df = pd.read_parquet(path)
    total = len(df)
    counts = df["label"].value_counts().sort_index()
    pos = total - counts.get(0, 0)
    print(f"  Total rows:     {total}")
    print(f"  Label counts:   {dict(counts)}")
    print(f"  Positive rows:  {pos} ({100.0 * pos / total:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
