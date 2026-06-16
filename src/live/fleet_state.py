"""fleet_state.py — Durable open-position + resolved-trade ledger for a fleet run.

Keyed by a stable ``fleet_id``; state lives in
``logs/fleet/<fleet_id>/state/``.

Design decisions
----------------
- Open positions: ``open_positions.jsonl`` — one JSON object per line,
  one line per open position (rewritten on every ``persist()``).
- Resolved ledger: ``resolved_ledger.jsonl`` — append-only, one line per
  resolved ``SimTrade``.
- Reload is **idempotent**: re-applying a ``SimTrade`` already in the ledger
  by ``intent_id`` is a silent no-op.
- Schema-versioned: every line carries ``"schema_version": 1`` so future
  migrations can detect old files.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OpenSimPosition:
    """An open simulated position waiting for SL/TP resolution.

    Parameters
    ----------
    intent_id : str
        Stable unique identifier derived from the ``OrderIntent`` that spawned
        this position (ticker:model:strategy:h1_timestamp).
    ticker : str
    model : str
    strategy : str
    direction : int
        1 = bull, 2 = bear.
    fill_price : float
        Actual fill price (next 1-min open for market; limit price for limit).
    sl : float
        Stop-loss price.
    tp : float
        Take-profit price.
    fill_ts : pd.Timestamp
        Timestamp of the fill bar.
    accumulated_bars : list[dict]
        1-min OHLCV dicts accumulated since fill, used by ``SimFillExecutor``
        when calling ``compute_exit``.  Stored as plain dicts (JSON-safe).
    cell_key : str
        ``"ticker:model:strategy"`` — the fleet cell this position belongs to.
    """

    intent_id: str
    ticker: str
    model: str
    strategy: str
    direction: int
    fill_price: float
    sl: float
    tp: float
    fill_ts: pd.Timestamp
    accumulated_bars: list = field(default_factory=list)
    cell_key: str = ""
    # 60×5 H1 window that produced the signal, stored as a flat list for
    # JSON serialisation.  Used by SimFillExecutor when calling compute_exit
    # over the accumulated 1-min slice so that ATR-floor / geometry is correct.
    # Optional — when absent (e.g. reloaded from old state), falls back to a
    # flat approximation.
    window_raw_flat: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fill_ts"] = self.fill_ts.isoformat() if pd.notna(self.fill_ts) else None
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OpenSimPosition":
        d = dict(d)
        d.pop("schema_version", None)
        ts_raw = d.pop("fill_ts", None)
        fill_ts = pd.Timestamp(ts_raw) if ts_raw is not None else pd.NaT
        return cls(**d, fill_ts=fill_ts)


@dataclass
class SimTrade:
    """A fully resolved simulated trade (ledger entry).

    Parameters
    ----------
    intent_id : str
        Same id as the ``OpenSimPosition`` that was resolved.
    ticker, model, strategy, direction : str/int
        Metadata forwarded from the position.
    fill_price : float
        Entry fill price.
    sl, tp : float
    fill_ts : pd.Timestamp
    close_price : float
        Actual exit price from ``TradeOutcome.exit_price``.
    close_ts : pd.Timestamp
        Timestamp of the exit bar (may be NaT for undecided).
    outcome : str
        One of ``"tp"``, ``"sl"``, ``"undecided"``, ``"session_close"``.
    r_multiple : float
        Signed R-multiple from ``TradeOutcome.r_multiple``.
    cell_key : str
    """

    intent_id: str
    ticker: str
    model: str
    strategy: str
    direction: int
    fill_price: float
    sl: float
    tp: float
    fill_ts: pd.Timestamp
    close_price: float
    close_ts: pd.Timestamp
    outcome: str
    r_multiple: float
    cell_key: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fill_ts"] = self.fill_ts.isoformat() if pd.notna(self.fill_ts) else None
        d["close_ts"] = self.close_ts.isoformat() if pd.notna(self.close_ts) else None
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SimTrade":
        d = dict(d)
        d.pop("schema_version", None)
        for k in ("fill_ts", "close_ts"):
            raw = d.pop(k, None)
            d[k] = pd.Timestamp(raw) if raw is not None else pd.NaT
        return cls(**d)


# ---------------------------------------------------------------------------
# FleetState
# ---------------------------------------------------------------------------


class FleetState:
    """Durable store of open sim-positions and the resolved-trade ledger.

    Parameters
    ----------
    fleet_id : str
        Stable identifier for this fleet run (short hash of cell matrix or
        user-supplied).
    base_dir : Path, optional
        Root directory under which ``logs/fleet/<fleet_id>/state/`` is
        created.  Defaults to ``Path("logs")``.

    Attributes
    ----------
    state_dir : Path
        ``<base_dir>/fleet/<fleet_id>/state/``

    Notes
    -----
    - On construction the state dir is created if absent and existing state
      is loaded.
    - ``resolve()`` is idempotent: re-resolving an ``intent_id`` already in
      the ledger is a no-op (no duplicate appended, no error raised).
    - ``persist()`` rewrites ``open_positions.jsonl`` (full snapshot) and
      is called automatically by ``add_open`` / ``resolve``.  Callers may
      also call it explicitly at session close.
    """

    _OPEN_FILE = "open_positions.jsonl"
    _LEDGER_FILE = "resolved_ledger.jsonl"

    def __init__(
        self,
        fleet_id: str,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.fleet_id = fleet_id
        _base = Path(base_dir) if base_dir is not None else Path("logs")
        self.state_dir = _base / "fleet" / fleet_id / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._open: dict[str, OpenSimPosition] = {}  # intent_id -> position
        self._ledger_ids: set[str] = set()  # intent_ids already resolved

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_open(self, position: OpenSimPosition) -> None:
        """Register a new open position.

        Positions are keyed by ``(cell_key, intent_id)`` so that two cells
        (e.g. same ticker, different strategy) never collide and each cell's
        ``SimFillExecutor`` only ever sees its own book.  If a position with
        the same key is already open it is silently replaced (re-fill edge
        case on session restart).
        """
        self._open[self._key(position.cell_key, position.intent_id)] = position
        self._persist_open()

    def resolve(self, trade: SimTrade) -> None:
        """Move a position from open to the resolved ledger.

        Idempotent: if this ``(cell_key, intent_id)`` is already in the ledger
        the call is a no-op.  Keying by cell_key keeps two cells that happen to
        share an ``intent_id`` (different strategy on the same ticker/ts) from
        resolving each other.
        """
        lid = self._key(trade.cell_key, trade.intent_id)
        if lid in self._ledger_ids:
            return  # already resolved — idempotent

        # Remove from open positions (may already be absent on reload)
        self._open.pop(lid, None)
        self._persist_open()

        # Append to ledger
        self._ledger_ids.add(lid)
        ledger_path = self.state_dir / self._LEDGER_FILE
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trade.to_dict()) + "\n")

    def open_positions(self, cell_key: Optional[str] = None) -> list[OpenSimPosition]:
        """Return a snapshot of open positions.

        Parameters
        ----------
        cell_key : str, optional
            When given, only positions belonging to that cell are returned.
            ``None`` (default) returns every cell's positions — used by
            reporting / replay, never by a single cell's executor.
        """
        if cell_key is None:
            return list(self._open.values())
        return [p for p in self._open.values() if p.cell_key == cell_key]

    def resolved_trades(self) -> list[SimTrade]:
        """Return all resolved trades from the ledger (full load)."""
        ledger_path = self.state_dir / self._LEDGER_FILE
        if not ledger_path.exists():
            return []
        trades = []
        with ledger_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    trades.append(SimTrade.from_dict(json.loads(line)))
        return trades

    def persist(self) -> None:
        """Explicit persist of open-positions snapshot (e.g. at session close)."""
        self._persist_open()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(cell_key: str, intent_id: str) -> str:
        """Composite open/ledger key — ``cell_key`` disambiguates intents that
        two cells share (same ticker/ts, different strategy)."""
        return f"{cell_key}|{intent_id}"

    def _load(self) -> None:
        """Load open positions and build the set of resolved keys."""
        # Load resolved ledger keys first (for idempotency check in resolve())
        ledger_path = self.state_dir / self._LEDGER_FILE
        if ledger_path.exists():
            with ledger_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            self._ledger_ids.add(
                                self._key(d.get("cell_key", ""), d["intent_id"])
                            )
                        except (json.JSONDecodeError, KeyError):
                            pass  # corrupt line — skip

        # Load open positions
        open_path = self.state_dir / self._OPEN_FILE
        if open_path.exists():
            with open_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            pos = OpenSimPosition.from_dict(d)
                            key = self._key(pos.cell_key, pos.intent_id)
                            # Skip positions that were already resolved in a
                            # previous session (idempotent reload)
                            if key not in self._ledger_ids:
                                self._open[key] = pos
                        except (json.JSONDecodeError, KeyError):
                            pass  # corrupt line — skip

    def _persist_open(self) -> None:
        """Rewrite open_positions.jsonl as a full snapshot."""
        open_path = self.state_dir / self._OPEN_FILE
        with open_path.open("w", encoding="utf-8") as fh:
            for pos in self._open.values():
                fh.write(json.dumps(pos.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Utility: stable intent_id from OrderIntent fields
# ---------------------------------------------------------------------------


def make_intent_id(
    ticker: str,
    model: str,
    strategy: str,
    h1_timestamp: pd.Timestamp,
) -> str:
    """Build a stable, human-readable intent identifier.

    Format: ``<ticker>:<model>:<strategy>:<h1_ts_isoformat>``.
    Deterministic for the same inputs — no UUID randomness so that replay
    and cross-session reload can match on the same id.
    """
    ts_str = h1_timestamp.isoformat() if pd.notna(h1_timestamp) else "NaT"
    return f"{ticker}:{model}:{strategy}:{ts_str}"
