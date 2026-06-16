"""fleet_replay.py — FleetReplayer: authoritative H1-grain P&L for a captured session.

Design
------
§4.2 of the plan: replay is the **authoritative** P&L number.  It
re-resolves each logged ``OrderIntent`` on forward **H1** bars built from
the captured 1-min bars (``bars_1m_fleet.jsonl``), then calls
``compute_exit`` — the single exit engine (§0 keystone).  No new exit
math here.

Resolution grain:  H1 (same as inspect/trading-simulation.md).
Live sim-fill grain: 1-min (preview only — see ``sim_executor.py``).

Self-contained
--------------
``RestBackfiller.warm`` writes warm-up bars to ``bars_1m_fleet.jsonl``
(tagged ``source="backfill"``) so replay needs no network.  Replaying the
same session twice always produces identical numbers (determinism).

Sparsity warning
----------------
When ``n_resolved < 30`` for any cell the report carries a ``SPARSITY
WARNING`` banner (per §11 pre-mortem: noise risk).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.live.fleet_state import make_intent_id
from src.strategy.exits import ExitConfig, compute_exit

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """Aggregated P&L for one (ticker, model, strategy) cell.

    Attributes
    ----------
    ticker, model, strategy : str
    n_intents : int
        Total intents logged for this cell (including skipped/degenerate).
    n_resolved : int
        Intents that reached a tp/sl/undecided outcome (i.e. filled +
        outcome != no_fill/no_future).
    n_tp : int
    n_sl : int
    n_undecided : int
    n_no_fill : int
        Limit intents that never filled within the forward window.
    after_cost_total_r : float
        Sum of ``r_multiple`` from ``compute_exit`` for resolved trades only
        (undecided trades contribute r_multiple=0 by convention).
    win_rate : float
        n_tp / max(n_tp + n_sl, 1).  Excludes undecided.
    undecided_rate : float
        n_undecided / max(n_resolved, 1).
    sparsity_warning : bool
        True when n_resolved < 30.
    """

    ticker: str
    model: str
    strategy: str
    n_intents: int = 0
    n_resolved: int = 0
    n_tp: int = 0
    n_sl: int = 0
    n_undecided: int = 0
    n_no_fill: int = 0
    after_cost_total_r: float = 0.0

    @property
    def win_rate(self) -> float:
        denom = self.n_tp + self.n_sl
        return self.n_tp / denom if denom > 0 else float("nan")

    @property
    def undecided_rate(self) -> float:
        return self.n_undecided / self.n_resolved if self.n_resolved > 0 else float("nan")

    @property
    def sparsity_warning(self) -> bool:
        return self.n_resolved < 30


@dataclass
class FleetReplayResult:
    """Aggregated result of a fleet replay run.

    Attributes
    ----------
    session_id : str
    fleet_id : str
    cells : list[CellResult]
    """

    session_id: str
    fleet_id: str
    cells: list[CellResult] = field(default_factory=list)

    @property
    def any_sparse(self) -> bool:
        return any(c.sparsity_warning for c in self.cells)


def merge_results(
    results: list["FleetReplayResult"],
    fleet_id: str,
) -> "FleetReplayResult":
    """Aggregate per-session ``FleetReplayResult`` objects into one (H3).

    Cells are merged by ``(ticker, model, strategy)``: all integer counts and
    ``after_cost_total_r`` are summed across sessions, so ``--fleet-id`` emits a
    single merged ``realistic_pnl.md`` instead of clobbering it per session.
    ``win_rate`` / ``undecided_rate`` / ``sparsity_warning`` recompute from the
    merged counts via their existing properties.

    The merged ``session_id`` lists the contributing sessions (comma-joined).
    """
    merged: dict[tuple[str, str, str], CellResult] = {}
    session_ids: list[str] = []

    for res in results:
        session_ids.append(res.session_id)
        for c in res.cells:
            key = (c.ticker, c.model, c.strategy)
            if key not in merged:
                merged[key] = CellResult(ticker=c.ticker, model=c.model, strategy=c.strategy)
            agg = merged[key]
            agg.n_intents += c.n_intents
            agg.n_resolved += c.n_resolved
            agg.n_tp += c.n_tp
            agg.n_sl += c.n_sl
            agg.n_undecided += c.n_undecided
            agg.n_no_fill += c.n_no_fill
            agg.after_cost_total_r += c.after_cost_total_r

    return FleetReplayResult(
        session_id=",".join(session_ids),
        fleet_id=fleet_id,
        cells=sorted(merged.values(), key=lambda c: (c.ticker, c.model, c.strategy)),
    )


# ---------------------------------------------------------------------------
# FleetReplayer
# ---------------------------------------------------------------------------

# How many forward H1 bars to scan per intent.  Set very high so no
# artificial cap cuts off resolution — the replay window is the full
# captured bar history after the intent's h1_timestamp.
_HIGH_TIMEOUT = 10_000


class FleetReplayer:
    """Re-resolves every logged ``OrderIntent`` on forward H1 bars.

    Parameters
    ----------
    session_dir : Path
        Path to ``logs/paper/<session-id>/`` (must contain
        ``intents.jsonl`` and ``bars_1m_fleet.jsonl``).
    exit_configs : dict[str, ExitConfig], optional
        Per-strategy ``ExitConfig`` overrides.  If a strategy is not in
        this dict a default ``ExitConfig(strategy=strategy,
        fill_mode="conservative")`` is used.
    fleet_id : str, optional
        Used for the report path.  Defaults to the session directory name.
    """

    def __init__(
        self,
        session_dir: Path,
        exit_configs: Optional[dict[str, ExitConfig]] = None,
        fleet_id: Optional[str] = None,
    ) -> None:
        self._session_dir = Path(session_dir)
        self._exit_configs: dict[str, ExitConfig] = exit_configs or {}
        self._fleet_id = fleet_id or self._session_dir.name

        if not self._session_dir.exists():
            raise FileNotFoundError(f"Session dir not found: {self._session_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> FleetReplayResult:
        """Execute replay and return aggregated results.

        Deterministic: repeated calls on the same session directory
        produce identical ``FleetReplayResult`` objects.
        """
        intents = self._load_intents()
        bars_1m = self._load_bars_1m()

        if not intents:
            _log.warning("No intents found in %s", self._session_dir)
            return FleetReplayResult(
                session_id=self._session_dir.name,
                fleet_id=self._fleet_id,
                cells=[],
            )

        # Build per-symbol sorted 1-min bar DataFrame
        symbol_bars: dict[str, pd.DataFrame] = self._build_symbol_bars(bars_1m)

        # Build H1 bars per symbol from 1-min bars
        symbol_h1: dict[str, list[tuple[pd.Timestamp, np.ndarray]]] = {
            sym: self._resample_to_h1(df)
            for sym, df in symbol_bars.items()
        }

        # Accumulate per-cell results
        cell_results: dict[tuple[str, str, str], CellResult] = {}

        for intent in intents:
            ticker = intent["ticker"]
            model = intent["model"]
            strategy = intent["strategy"]
            key = (ticker, model, strategy)

            if key not in cell_results:
                cell_results[key] = CellResult(ticker=ticker, model=model, strategy=strategy)
            cr = cell_results[key]
            cr.n_intents += 1

            # Skip degenerate intents
            if intent.get("skip_reason"):
                _log.debug(
                    "Skipping intent %s (skip_reason=%s)",
                    make_intent_id(ticker, model, strategy, pd.Timestamp(intent["h1_timestamp"])),
                    intent["skip_reason"],
                )
                continue

            # Resolve this intent on forward H1 bars
            self._resolve_intent(intent, symbol_h1.get(ticker, []), cr)

        result = FleetReplayResult(
            session_id=self._session_dir.name,
            fleet_id=self._fleet_id,
            cells=sorted(cell_results.values(), key=lambda c: (c.ticker, c.model, c.strategy)),
        )
        return result

    # ------------------------------------------------------------------
    # Report emitter
    # ------------------------------------------------------------------

    @staticmethod
    def emit_report(
        result: FleetReplayResult,
        reports_dir: Path,
    ) -> Path:
        """Write ``realistic_pnl.md`` to ``reports_dir/<fleet_id>/``.

        Returns the path of the written file.  Static so a merged
        cross-session result (H3) can be emitted without a session directory.
        """
        out_dir = Path(reports_dir) / result.fleet_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "realistic_pnl.md"

        lines: list[str] = []
        lines.append(f"# Fleet Replay — Realistic P&L")
        lines.append(f"")
        lines.append(f"**Session:** `{result.session_id}`  ")
        lines.append(f"**Fleet ID:** `{result.fleet_id}`  ")
        lines.append(f"**Resolution grain:** H1 (authoritative — matches inspect)")
        lines.append(f"**Live sim-fill grain:** 1-min (preview only; may differ by design)")
        lines.append(f"")

        if result.any_sparse:
            lines.append(
                "> **SPARSITY WARNING** — one or more cells have `n_resolved < 30`.  "
                "Results are statistically unreliable noise; do not quote as a verdict."
            )
            lines.append("")

        # Table header
        lines.append(
            "| ticker | model | strategy | after_cost_total_R | win_rate | "
            "n_trades | n_resolved | undecided_rate | sparse? |"
        )
        lines.append(
            "|--------|-------|----------|--------------------|----------|"
            "---------|------------|----------------|---------|"
        )

        for c in result.cells:
            win_pct = f"{c.win_rate:.1%}" if not (c.win_rate != c.win_rate) else "—"
            undec_pct = f"{c.undecided_rate:.1%}" if not (c.undecided_rate != c.undecided_rate) else "—"
            sparse_flag = "YES ⚠" if c.sparsity_warning else "no"
            lines.append(
                f"| {c.ticker} | {c.model} | {c.strategy} "
                f"| {c.after_cost_total_r:+.3f} "
                f"| {win_pct} "
                f"| {c.n_intents} "
                f"| {c.n_resolved} "
                f"| {undec_pct} "
                f"| {sparse_flag} |"
            )

        lines.append("")
        lines.append(
            "_Replay is deterministic: running again on the same session produces "
            "identical numbers._"
        )
        lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        _log.info("Wrote realistic_pnl.md → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_intents(self) -> list[dict]:
        """Load all intents from the root ``intents.jsonl``."""
        path = self._session_dir / "intents.jsonl"
        if not path.exists():
            _log.warning("intents.jsonl not found at %s", path)
            return []
        intents = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        intents.append(json.loads(line))
                    except json.JSONDecodeError:
                        _log.warning("Skipping malformed line in intents.jsonl")
        return intents

    def _load_bars_1m(self) -> list[dict]:
        """Load all 1-min bars from ``bars_1m_fleet.jsonl``."""
        path = self._session_dir / "bars_1m_fleet.jsonl"
        if not path.exists():
            _log.warning("bars_1m_fleet.jsonl not found at %s", path)
            return []
        bars = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        bars.append(json.loads(line))
                    except json.JSONDecodeError:
                        _log.warning("Skipping malformed line in bars_1m_fleet.jsonl")
        return bars

    def _build_symbol_bars(self, bars_1m: list[dict]) -> dict[str, pd.DataFrame]:
        """Group 1-min bars by symbol and sort chronologically."""
        grouped: dict[str, list[dict]] = {}
        for b in bars_1m:
            sym = b.get("symbol", "")
            if sym not in grouped:
                grouped[sym] = []
            grouped[sym].append(b)

        result = {}
        for sym, rows in grouped.items():
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
                "America/New_York"
            )
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            df = df.set_index("timestamp")
            result[sym] = df

        return result

    def _resample_to_h1(self, df: pd.DataFrame) -> list[tuple[pd.Timestamp, np.ndarray]]:
        """Resample a symbol's 1-min DataFrame to RTH-anchored H1 bars.

        Canonical params: closed='left', label='left', offset='30min'
        (same as ``src.data.download._resample_minute_to_h1`` and training).

        Returns a list of (h1_timestamp, ohlcv_array) sorted ascending, where
        ohlcv_array has shape (5,) with columns [open, high, low, close, volume].
        """
        if df.empty:
            return []

        # RTH filter: 09:30–15:59
        df = df.between_time("09:30", "15:59")
        if df.empty:
            return []

        # Ensure required columns exist
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                _log.warning("Column %s missing from bars; skipping resample", col)
                return []

        h1 = df.resample("1h", closed="left", label="left", offset="30min").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        h1 = h1.dropna(subset=["open"])
        h1 = h1[~h1.index.duplicated(keep="first")]

        result = []
        for ts, row in h1.iterrows():
            arr = np.array(
                [row["open"], row["high"], row["low"], row["close"], row["volume"]],
                dtype=np.float64,
            )
            result.append((ts, arr))

        return result

    def _resolve_intent(
        self,
        intent: dict,
        h1_bars: list[tuple[pd.Timestamp, np.ndarray]],
        cr: CellResult,
    ) -> None:
        """Resolve one intent against forward H1 bars, updating ``cr`` in-place."""
        try:
            h1_ts = pd.Timestamp(intent["h1_timestamp"])
        except (KeyError, ValueError) as exc:
            _log.warning("Bad h1_timestamp in intent: %s", exc)
            return

        if h1_ts.tzinfo is None:
            h1_ts = h1_ts.tz_localize("America/New_York")

        # Rebuild the decision window (60 H1 bars UP TO AND INCLUDING h1_ts)
        window_raw = self._build_decision_window(h1_ts, h1_bars)
        if window_raw is None:
            _log.debug("Could not rebuild 60-bar window for intent at %s", h1_ts)
            return

        # Collect FORWARD H1 bars (strictly after h1_ts)
        forward_rows = [
            ohlcv[:4]  # OHLC only — compute_exit expects (L, 4)
            for ts, ohlcv in h1_bars
            if ts > h1_ts
        ]

        direction = int(intent.get("direction", 1))
        strategy = intent.get("strategy", "fixed_2r")
        cfg = self._get_config(strategy)

        if forward_rows:
            future_ohlcv = np.array(forward_rows, dtype=np.float64)
        else:
            future_ohlcv = np.empty((0, 4), dtype=np.float64)

        # Override fill_timeout_bars so the full captured history is used
        import dataclasses
        cfg_full = dataclasses.replace(cfg, fill_timeout_bars=_HIGH_TIMEOUT)

        outcome = compute_exit(
            window_raw=window_raw,
            future_ohlcv=future_ohlcv,
            direction=direction,
            config=cfg_full,
        )

        # Tally outcome
        if outcome.outcome == "no_fill":
            cr.n_no_fill += 1
            # no_fill: not resolved (limit never triggered)
            return

        if outcome.outcome == "no_future":
            # No forward bars at all — treat as unresolved
            return

        # Filled trade
        cr.n_resolved += 1
        if outcome.outcome == "tp":
            cr.n_tp += 1
        elif outcome.outcome == "sl":
            cr.n_sl += 1
        else:  # "undecided"
            cr.n_undecided += 1

        # Per-trade cost drag — IDENTICAL formula to inspect's summarise_trades
        # (src/inspect/outcomes.py): round-trip slippage+commission expressed in
        # R units via risk_dollars = |entry - sl|.  Without this, the
        # after_cost_total_R column would be a pre-cost total and overstate vs
        # the inspect realistic sweep it is compared against.
        _TICK = 0.01
        risk_dollars = (
            abs(outcome.entry - outcome.sl)
            if not (np.isnan(outcome.entry) or np.isnan(outcome.sl))
            else 0.0
        )
        cost_per_rt = (
            cfg_full.slippage_ticks * _TICK * 2 + cfg_full.commission_per_share * 2
        )
        cost_drag_r = cost_per_rt / risk_dollars if risk_dollars > 0 else 0.0
        cr.after_cost_total_r += outcome.r_multiple - cost_drag_r

    def _build_decision_window(
        self,
        h1_ts: pd.Timestamp,
        h1_bars: list[tuple[pd.Timestamp, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Extract the 60-bar H1 window ending at (and including) h1_ts.

        Returns shape (60, 5) float64, or None if insufficient bars.
        """
        # Find bars with timestamp <= h1_ts, sorted ascending
        eligible = [ohlcv for ts, ohlcv in h1_bars if ts <= h1_ts]
        if len(eligible) < 60:
            return None

        window = np.array(eligible[-60:], dtype=np.float64)  # (60, 5)
        return window

    def _get_config(self, strategy: str) -> ExitConfig:
        """Return ExitConfig for strategy, using override if provided."""
        if strategy in self._exit_configs:
            return self._exit_configs[strategy]
        return ExitConfig(strategy=strategy, fill_mode="conservative")
