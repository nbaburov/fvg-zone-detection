"""report_utils.py — Markdown + HTML report generation for rigor sprint."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def timestamped_dir(base: Path) -> Path:
    """Create and return base/<YYYY-MM-DD_HHMMSS>/."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_rigor_report(
    results_df: pd.DataFrame,
    output_dir: Path,
    title: str,
) -> Path:
    """Write markdown summary table to output_dir/<slug>.md.

    Expects columns: [seed, macro_f1, none_f1, bull_f1, bear_f1]
    Appends mean±std row.

    Returns path to written markdown file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = title.lower().replace(" ", "_").replace("/", "_")
    out_path = output_dir / f"{slug}.md"

    lines: list[str] = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| seed | macro_f1 | none_f1 | bull_f1 | bear_f1 |",
        "|------|----------|---------|---------|---------|",
    ]

    numeric_cols = ["macro_f1", "none_f1", "bull_f1", "bear_f1"]

    for _, row in results_df.iterrows():
        seed_val = int(row["seed"]) if "seed" in row else "-"
        lines.append(
            f"| {seed_val} "
            f"| {row['macro_f1']:.4f} "
            f"| {row['none_f1']:.4f} "
            f"| {row['bull_f1']:.4f} "
            f"| {row['bear_f1']:.4f} |"
        )

    # Mean ± std row
    means = results_df[numeric_cols].mean()
    stds = results_df[numeric_cols].std()
    lines.append(
        f"| **mean±std** "
        f"| **{means['macro_f1']:.4f}±{stds['macro_f1']:.4f}** "
        f"| {means['none_f1']:.4f}±{stds['none_f1']:.4f} "
        f"| {means['bull_f1']:.4f}±{stds['bull_f1']:.4f} "
        f"| {means['bear_f1']:.4f}±{stds['bear_f1']:.4f} |"
    )

    lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path
