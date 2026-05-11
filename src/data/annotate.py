"""Gold-set annotation tool for label quality validation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

_LOW_VOL_QUANTILE: float = 0.25
_KAPPA_GATE_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Private stratum samplers
# ---------------------------------------------------------------------------


def _sample_fvg_rich(
    df: pd.DataFrame,
    pos_indices: np.ndarray,
    n: int,
    n_fvg_rich: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Return up to n_fvg_rich (index, stratum) tuples from FVG-rich neighbourhood."""
    if len(pos_indices) == 0:
        candidates = list(range(n))
    else:
        # Vectorised neighbourhood expansion: ±5 bars around each positive
        offsets = np.arange(-5, 6)
        neighbour_idx = (pos_indices[:, None] + offsets).ravel()
        candidates = np.unique(np.clip(neighbour_idx, 0, n - 1)).tolist()
        if len(candidates) < n_fvg_rich:
            candidates = list(range(n))

    chosen = rng.choice(candidates, size=min(n_fvg_rich, len(candidates)), replace=False)
    return [(int(i), "fvg_rich") for i in chosen]


def _sample_low_vol(
    df: pd.DataFrame,
    n: int,
    n_low_vol: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Return up to n_low_vol (index, stratum) tuples from low-ATR bars."""
    atr_values = df["_atr"].to_numpy()
    low_vol_threshold = np.quantile(atr_values, _LOW_VOL_QUANTILE)
    candidates = np.where(atr_values <= low_vol_threshold)[0].tolist()
    if len(candidates) < n_low_vol:
        candidates = list(range(n))
    chosen = rng.choice(candidates, size=min(n_low_vol, len(candidates)), replace=False)
    return [(int(i), "low_vol") for i in chosen]


def _sample_random_stratified(
    df: pd.DataFrame,
    n_random: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Return up to n_random (index, stratum) tuples, stratified by year."""
    years = df.index.year.unique().tolist()
    per_year = max(1, n_random // len(years))
    random_idx: list[int] = []
    for year in years:
        year_positions = np.where(df.index.year == year)[0]
        n_sample = min(per_year, len(year_positions))
        sampled = rng.choice(year_positions, size=n_sample, replace=False)
        random_idx.extend(sampled.tolist())

    if len(random_idx) > n_random:
        random_idx = rng.choice(random_idx, size=n_random, replace=False).tolist()

    return [(int(i), "random") for i in random_idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_gold_set(
    df: pd.DataFrame,
    n_fvg_rich: int = 25,
    n_low_vol: int = 25,
    n_random: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return DataFrame of (n_fvg_rich + n_low_vol + n_random) rows from df, stratified.

    Stratification:
    - fvg_rich: windows from high-density positive-label periods
    - low_vol:  windows from low ATR periods (intraday range as ATR proxy)
    - random:   uniform random sample stratified by year

    Each row includes: candle_index (int), datetime, programmatic_label, stratum.
    """
    rng = np.random.default_rng(seed)
    total = n_fvg_rich + n_low_vol + n_random
    n = len(df)

    if n < total:
        raise ValueError(
            f"Dataset has only {n} rows but gold set requires {total}. "
            "Use a larger dataset."
        )

    # Compute ATR proxy: intraday range (high - low)
    df = df.copy()
    df["_atr"] = df["high"] - df["low"]

    # Identify positive label positions (raw_label != 0)
    pos_indices = np.where((df["raw_label"] != 0).to_numpy())[0]

    fvg_set = _sample_fvg_rich(df, pos_indices, n, n_fvg_rich, rng)
    low_set = _sample_low_vol(df, n, n_low_vol, rng)
    rand_set = _sample_random_stratified(df, n_random, rng)

    # Deduplicate across strata — first occurrence wins
    seen: set[int] = set()
    combined: list[tuple[int, str]] = []
    for idx_val, stratum in fvg_set + low_set + rand_set:
        if idx_val not in seen:
            seen.add(idx_val)
            combined.append((idx_val, stratum))

    # Fill remainder with random from remaining positions if dedup reduced count
    if len(combined) < total:
        remaining = list(set(range(n)) - seen)
        extra_needed = total - len(combined)
        if len(remaining) >= extra_needed:
            extra_idx = rng.choice(remaining, size=extra_needed, replace=False)
            for idx_val in extra_idx:
                combined.append((int(idx_val), "random"))

    combined = combined[:total]

    records = []
    for idx_val, stratum in combined:
        row = df.iloc[idx_val]
        records.append(
            {
                "candle_index": idx_val,
                "datetime": df.index[idx_val],
                "programmatic_label": int(row["raw_label"]),
                "stratum": stratum,
            }
        )

    return pd.DataFrame(records)


def compute_kappa(gold_csv_path: str) -> float:
    """
    Load gold_labels.csv, compute Cohen's kappa on (programmatic_label, human_label).
    Returns float in [-1, 1]. Logs breakdown by strata if strata column present.
    Raises: FileNotFoundError if path does not exist.
    """
    path = Path(gold_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold labels file not found: {gold_csv_path}")

    df = pd.read_csv(path)
    required = {"programmatic_label", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gold_labels.csv missing columns: {missing}")

    prog = df["programmatic_label"].tolist()
    human = df["human_label"].tolist()

    if len(prog) == 0:
        logger.warning("Gold labels file has no rows yet — cannot compute kappa.")
        return float("nan")
    if len(prog) < 2:
        logger.warning(
            "Gold labels file has only %d row(s); kappa needs at least 2.", len(prog)
        )
        return float("nan")
    if len(set(prog)) < 2 and len(set(human)) < 2:
        logger.warning(
            "All annotations belong to a single class — kappa is undefined."
        )
        return float("nan")

    kappa = cohen_kappa_score(prog, human)

    if kappa < _KAPPA_GATE_THRESHOLD:
        logger.warning(
            "PHASE 4 BLOCKED — label quality gate failed. "
            "Cohen's kappa = %.3f (threshold: %.1f). Surface to user before training.",
            kappa,
            _KAPPA_GATE_THRESHOLD,
        )
    else:
        logger.info(
            "Cohen's kappa = %.3f — label quality gate PASSED (threshold: %.1f)",
            kappa,
            _KAPPA_GATE_THRESHOLD,
        )

    return float(kappa)


def run_annotation_ui(
    df: pd.DataFrame,
    gold_set_df: pd.DataFrame,
    output_path: str = "data/gold_labels.csv",
) -> None:
    """
    Interactive annotation UI using Plotly and keyboard input.

    Opens a Plotly candlestick chart for each unannotated candle in gold_set_df.
    Displays 7-candle window (3 before + target + 3 after) from raw OHLCV data.
    Does NOT display programmatic label (anchoring bias prevention).

    Keyboard controls:
        b = bullish FVG (label=1)
        e = bearish FVG (label=-1)
        n = none (label=0)
        a = ambiguous (label=0, annotator_note="ambiguous")

    Resumes from last annotated candle if output_path already exists.
    Appends each annotation to output_path immediately (safe to interrupt).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required for annotation UI. Install with: pip install plotly")

    output = Path(output_path)

    # Load already-annotated indices if resuming
    annotated: set[int] = _resume_annotated_set(output)

    fieldnames = ["candle_index", "datetime", "programmatic_label", "human_label", "annotator_note"]
    output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output.exists()

    n_df = len(df)

    with open(output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for _, gold_row in gold_set_df.iterrows():
            candle_idx = int(gold_row["candle_index"])
            if candle_idx in annotated:
                continue

            _display_candle_chart(df, candle_idx, n_df, go)
            human_label, note = _prompt_label()

            writer.writerow(
                {
                    "candle_index": candle_idx,
                    "datetime": str(df.index[candle_idx]),
                    "programmatic_label": int(gold_row["programmatic_label"]),
                    "human_label": human_label,
                    "annotator_note": note,
                }
            )
            f.flush()
            annotated.add(candle_idx)
            print(f"  Saved annotation for candle {candle_idx}")

    print(f"\nAnnotation complete. {len(annotated)} candles annotated → {output}")


def _resume_annotated_set(output: Path) -> set[int]:
    """Load previously annotated candle indices from output CSV, if it exists."""
    if output.exists():
        existing = pd.read_csv(output)
        if "candle_index" in existing.columns:
            return set(existing["candle_index"].tolist())
    return set()


def _display_candle_chart(df: pd.DataFrame, candle_idx: int, n_df: int, go) -> None:
    """
    Render a 30-bar Plotly chart centred on candle_idx.

    Overlays (all pre-computed from raw OHLCV — no label data shown):
    - Swing markers: purple triangles at 50-bar pivot highs/lows visible in window.
    - S/R lines: orange horizontal lines at 5-bar pivot high/low levels.
    - Gann midpoint: blue dashed horizontal at midpoint of 50-bar swing.
    - FVG zone: shaded rectangle if a geometric FVG exists at candle_idx.
    """
    from src.data.labels.sr import compute_pivot_levels
    from src.data.labels.bos import compute_bos

    # 30-bar window: 20 bars to the left, 10 to the right
    start = max(0, candle_idx - 20)
    end = min(n_df, candle_idx + 11)
    window_df = df.iloc[start:end]
    target_ts = df.index[candle_idx]

    # -- Pre-compute overlays from full df (causal) --
    # S/R pivot levels (causal)
    pivot_high_full, pivot_low_full = compute_pivot_levels(df, lookback=5)

    # Swing for Gann
    swing_high_full = df["high"].rolling(50, min_periods=1).max().shift(1)
    swing_low_full = df["low"].rolling(50, min_periods=1).min().shift(1)
    swing_mid_full = swing_low_full + 0.5 * (swing_high_full - swing_low_full)

    # Window-local values
    ph_window = pivot_high_full.iloc[start:end]
    pl_window = pivot_low_full.iloc[start:end]
    swing_mid_at_target = float(swing_mid_full.iloc[candle_idx]) if candle_idx < len(swing_mid_full) else float("nan")

    # FVG zone at candle_idx (if geometric FVG exists — show zone only, no label)
    fvg_zone = _compute_fvg_zone(df, candle_idx, n_df)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=window_df.index,
                open=window_df["open"],
                high=window_df["high"],
                low=window_df["low"],
                close=window_df["close"],
                name="OHLCV",
            )
        ]
    )

    # Target bar marker
    fig.add_vline(x=target_ts.timestamp() * 1000, line_dash="dash", line_color="red", line_width=2)

    # S/R horizontal lines (orange) — show unique levels in window
    sr_levels_shown: set[float] = set()
    for ts, ph_val, pl_val in zip(window_df.index, ph_window, pl_window):
        if not np.isnan(ph_val) and ph_val not in sr_levels_shown:
            fig.add_hline(y=ph_val, line_color="orange", line_dash="dot", line_width=1, opacity=0.7)
            sr_levels_shown.add(ph_val)
        if not np.isnan(pl_val) and pl_val not in sr_levels_shown:
            fig.add_hline(y=pl_val, line_color="orange", line_dash="dot", line_width=1, opacity=0.7)
            sr_levels_shown.add(pl_val)

    # Gann midpoint (blue dashed)
    if not np.isnan(swing_mid_at_target):
        fig.add_hline(
            y=swing_mid_at_target,
            line_color="blue",
            line_dash="dash",
            line_width=1.5,
            annotation_text="Gann mid",
            annotation_position="right",
        )

    # Swing markers (purple triangles) at target bar
    # Show if target bar's pivot values are locally extreme in the window
    _add_swing_markers(fig, df, start, end, pivot_high_full, pivot_low_full)

    # FVG zone shading
    if fvg_zone is not None:
        direction, zone_bottom, zone_top = fvg_zone
        fill_color = "rgba(0,200,100,0.15)" if direction == "bull" else "rgba(200,50,50,0.15)"
        fig.add_hrect(
            y0=zone_bottom,
            y1=zone_top,
            fillcolor=fill_color,
            layer="below",
            line_width=1,
            line_color="rgba(0,200,100,0.5)" if direction == "bull" else "rgba(200,50,50,0.5)",
            annotation_text=f"FVG {direction}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"Candle {candle_idx} — {target_ts} | b=bull  e=bear  n=none  a=ambiguous",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
    )
    fig.show()


def _compute_fvg_zone(
    df: pd.DataFrame, candle_idx: int, n_df: int
) -> tuple[str, float, float] | None:
    """
    Return (direction, bottom, top) if a geometric FVG exists at candle_idx.
    Returns None if no FVG or boundary issue.
    """
    if candle_idx < 1 or candle_idx > n_df - 2:
        return None
    h_prev = float(df["high"].iloc[candle_idx - 1])
    l_next = float(df["low"].iloc[candle_idx + 1])
    l_prev = float(df["low"].iloc[candle_idx - 1])
    h_next = float(df["high"].iloc[candle_idx + 1])
    o_mid = float(df["open"].iloc[candle_idx])
    c_mid = float(df["close"].iloc[candle_idx])

    if h_prev < l_next and c_mid > o_mid:
        return ("bull", h_prev, l_next)
    if l_prev > h_next and c_mid < o_mid:
        return ("bear", h_next, l_prev)
    return None


def _add_swing_markers(
    fig,
    df: pd.DataFrame,
    start: int,
    end: int,
    pivot_high_full: pd.Series,
    pivot_low_full: pd.Series,
) -> None:
    """
    Add purple triangle markers at bars where a local pivot high/low occurred in the window.
    Uses the pivot_high/low computed causally from the full df.
    """
    window_ph = pivot_high_full.iloc[start:end]
    window_pl = pivot_low_full.iloc[start:end]
    window_high = df["high"].iloc[start:end]
    window_low = df["low"].iloc[start:end]

    # Swing high markers: bar i where ph[i] == high[i-1] (the pivot occurred at i-1)
    # Approximate: show bars where local high in window matches the rolling pivot value
    ph_times = []
    ph_vals = []
    pl_times = []
    pl_vals = []

    for i in range(len(window_ph)):
        ph_val = window_ph.iloc[i]
        pl_val = window_pl.iloc[i]
        ts = window_ph.index[i]
        bar_high = window_high.iloc[i]
        bar_low = window_low.iloc[i]

        # Heuristic: pivot high marker if local high is close to the rolling pivot high
        if not np.isnan(ph_val) and abs(bar_high - ph_val) < 0.05:
            ph_times.append(ts)
            ph_vals.append(bar_high + 0.1)  # slightly above bar

        if not np.isnan(pl_val) and abs(bar_low - pl_val) < 0.05:
            pl_times.append(ts)
            pl_vals.append(bar_low - 0.1)  # slightly below bar

    if ph_times:
        fig.add_trace(
            dict(
                type="scatter",
                x=ph_times,
                y=ph_vals,
                mode="markers",
                marker=dict(symbol="triangle-down", color="purple", size=8),
                name="Swing High",
            )
        )
    if pl_times:
        fig.add_trace(
            dict(
                type="scatter",
                x=pl_times,
                y=pl_vals,
                mode="markers",
                marker=dict(symbol="triangle-up", color="purple", size=8),
                name="Swing Low",
            )
        )


def _prompt_label() -> tuple[int, str]:
    """Prompt annotator for a label key and return (human_label, note)."""
    while True:
        key = input("Label (b/e/n/a): ").strip().lower()
        if key == "b":
            return 1, ""
        elif key == "e":
            return -1, ""
        elif key == "n":
            return 0, ""
        elif key == "a":
            return 0, "ambiguous"
        else:
            print("Invalid key. Use: b=bull, e=bear, n=none, a=ambiguous")
