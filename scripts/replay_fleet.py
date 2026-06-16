#!/usr/bin/env python3
"""replay_fleet.py — Re-resolve a captured fleet session into a realistic P&L table.

Usage
-----
::

    # Replay a single session
    python scripts/replay_fleet.py --session 20260610_093500

    # Replay all sessions for a fleet-id
    python scripts/replay_fleet.py --fleet-id abc123

    # Override realism params
    python scripts/replay_fleet.py --session 20260610_093500 \\
        --fill-mode conservative --tp-rr 2.0

    # Output to custom dir
    python scripts/replay_fleet.py --session 20260610_093500 \\
        --reports-dir /tmp/reports

Output
------
``reports/fleet/<fleet-id>/realistic_pnl.md`` — markdown table per
(ticker, model, strategy) with after_cost_total_R, win_rate, n_trades,
n_resolved, undecided_rate, and a SPARSITY WARNING banner when
n_resolved < 30 for any cell.

Determinism
-----------
Replaying the same session twice produces identical numbers (no
randomness; all inputs read from session logs).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.live.fleet_replay import (
    FleetReplayer,
    FleetReplayResult,
    merge_results,
)
from src.strategy.exits import ExitConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
_log = logging.getLogger("replay_fleet")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-resolve a captured fleet session into a realistic P&L table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--session",
        metavar="SESSION_ID",
        help="Single session ID (directory name under logs/paper/).",
    )
    source.add_argument(
        "--fleet-id",
        metavar="FLEET_ID",
        help=(
            "Fleet ID — replay ALL sessions whose logs/paper/<session-id> "
            "matches this fleet-id prefix, then aggregate.  Currently replays "
            "each session independently and merges cell-level counts."
        ),
    )

    p.add_argument(
        "--base-dir",
        default="logs/paper",
        metavar="DIR",
        help="Root directory for session logs.",
    )
    p.add_argument(
        "--reports-dir",
        default="reports/fleet",
        metavar="DIR",
        help="Root directory for P&L reports.",
    )
    p.add_argument(
        "--fill-mode",
        choices=["conservative", "optimistic"],
        default="conservative",
        help="Fill mode for limit entries.",
    )
    p.add_argument(
        "--tp-rr",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="Take-profit reward-to-risk multiple (used by fixed_2r and swing fallback).",
    )
    p.add_argument(
        "--realistic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply realism guards (ATR min-stop floor, commission, slippage) so "
            "after_cost_total_R reflects costs — matches the live builder + "
            "inspect realistic sweep. Default: on (use --no-realistic to disable)."
        ),
    )
    p.add_argument(
        "--forward-bars-from",
        metavar="PARQUET",
        default=None,
        help=(
            "Optional path to a parquet file containing additional forward bars "
            "(appended to session bars before replay).  Useful for offline testing."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STRATEGIES = ["fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"]


def _build_exit_configs(
    fill_mode: str,
    tp_rr: float,
    realistic: bool = True,
) -> dict[str, ExitConfig]:
    """Build per-strategy ExitConfigs.

    H1: when ``realistic`` (default ON) the same realism guards used by the
    live fleet builder and the inspect realistic sweep are threaded in —
    ATR min-stop floor (0.5), commission (0.005/share), slippage (1 tick) —
    so ``after_cost_total_R`` reflects costs.  ``--no-realistic`` zeroes them.
    """
    return {
        s: ExitConfig(
            strategy=s,
            fill_mode=fill_mode,
            tp_rr=tp_rr,
            min_stop_atr_k=0.5 if realistic else None,
            commission_per_share=0.005 if realistic else 0.0,
            slippage_ticks=1 if realistic else 0,
        )
        for s in _STRATEGIES
    }


def _run_session(
    session_dir: Path,
    fleet_id: str,
    exit_configs: dict[str, ExitConfig],
) -> FleetReplayResult:
    """Run replay for one session and return its result (no report emitted)."""
    if not session_dir.exists():
        _log.error("Session directory not found: %s", session_dir)
        sys.exit(1)
    _log.info("Replaying session: %s", session_dir.name)
    replayer = FleetReplayer(
        session_dir=session_dir,
        exit_configs=exit_configs,
        fleet_id=fleet_id,
    )
    return replayer.run()


def _emit_and_print(
    result: FleetReplayResult,
    reports_dir: Path,
) -> None:
    """Emit the (possibly merged) report and print a stdout summary."""
    report_path = FleetReplayer.emit_report(result, reports_dir)

    print(f"\nSession: {result.session_id}  Fleet: {result.fleet_id}")
    if result.cells:
        print(f"{'ticker':<6} {'model':<12} {'strategy':<12} {'total_R':>10} {'win_rate':>9} {'n_resolved':>11} {'sparse':>7}")
        print("-" * 70)
        for c in result.cells:
            win_str = f"{c.win_rate:.1%}" if c.win_rate == c.win_rate else "  —  "
            print(
                f"{c.ticker:<6} {c.model:<12} {c.strategy:<12} "
                f"{c.after_cost_total_r:>+10.3f} {win_str:>9} "
                f"{c.n_resolved:>11}  {'⚠' if c.sparsity_warning else ' '}"
            )
    else:
        print("  (no cells resolved)")

    print(f"\nReport written: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    exit_configs = _build_exit_configs(args.fill_mode, args.tp_rr, args.realistic)
    base_dir = Path(args.base_dir)
    reports_dir = Path(args.reports_dir)

    if args.session:
        session_dir = base_dir / args.session
        fleet_id = args.session  # default fleet_id = session name
        result = _run_session(session_dir, fleet_id, exit_configs)
        _emit_and_print(result, reports_dir)

    elif args.fleet_id:
        # Aggregate ALL matching sessions into ONE merged report (H3).
        fleet_id = args.fleet_id
        if not base_dir.exists():
            _log.error("Base dir not found: %s", base_dir)
            sys.exit(1)
        matching = sorted(
            d for d in base_dir.iterdir()
            if d.is_dir() and fleet_id in d.name
        )
        if not matching:
            _log.error(
                "No session directories found under %s matching fleet-id '%s'",
                base_dir, fleet_id,
            )
            sys.exit(1)
        _log.info("Found %d session(s) for fleet-id '%s'", len(matching), fleet_id)
        results = [
            _run_session(session_dir, fleet_id, exit_configs)
            for session_dir in matching
        ]
        merged = merge_results(results, fleet_id)
        _emit_and_print(merged, reports_dir)


if __name__ == "__main__":
    main()
