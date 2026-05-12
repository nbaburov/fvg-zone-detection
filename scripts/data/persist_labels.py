"""persist_labels.py — DEPRECATED.

spy_h1_labeled.parquet is no longer used. The canonical labelled dataset is
data/processed/spy_h1.parquet, produced by build_pipeline(labeller_name='fvg_valid').

This script is kept as a stub to avoid import errors. It does nothing.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "DEPRECATED: spy_h1_labeled.parquet is no longer used. "
        "The canonical labelled dataset is data/processed/spy_h1.parquet, "
        "produced by build_pipeline(labeller_name='fvg_valid')."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
