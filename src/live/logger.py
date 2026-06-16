"""logger.py — Session logger: parquet + sqlite + jsonl.

Writes per-session data to logs/paper/<session-id>/. Three stores:
- Parquet (bars_1m, bars_h1, predictions) — buffered, flushed on H1 close or close().
- SQLite (orders, fills) — written immediately per event.
- JSONL (equity, events) — written immediately per event.

``FleetSessionLogger`` extends ``SessionLogger`` with:
- Namespace isolation per ``(ticker, model, strategy)`` — each cell writes to
  its own sub-directory under the session dir.
- ``log_bar(bar, source)`` — logs a ``MinuteBar`` tagged with a ``source`` field
  (``"live"`` or ``"backfill"``); used by ``RestBackfiller`` (WS-B).
- ``log_intent(intent)`` — appends an ``OrderIntent`` to a per-cell JSONL file
  (``intents.jsonl``) for deterministic offline replay (WS-G).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.live.slippage import SlippageModel
from src.live.stream import MinuteBar
from src.live.window_builder import H1Bar, WindowEvent
from src.live.decision import TradeAction

# Imported lazily inside methods to avoid circular dependency at module level.
# from src.live.order_plan import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class FillEvent:
    """Fill or partial_fill notification from TradingStream."""

    order_id: str
    filled_at: pd.Timestamp
    fill_price: float
    fill_qty: int
    fill_type: str  # "fill" | "partial_fill"
    side: str       # "buy" | "sell"
    symbol: Optional[str] = None  # set in the fleet path so FillRouter can route by symbol


class SessionLogger:
    """Logs all session events to logs/paper/<session-id>/."""

    def __init__(self, session_id: str, base_dir: str = "logs/paper") -> None:
        self._session_id = session_id
        self._dir = Path(base_dir) / session_id
        self._dir.mkdir(parents=True, exist_ok=True)

        # In-memory parquet buffers
        self._bars_1m: list[dict] = []
        self._bars_h1: list[dict] = []
        self._predictions: list[dict] = []

        # SQLite connection
        self._conn = sqlite3.connect(str(self._dir / "orders.sqlite"))
        self._fills_conn = sqlite3.connect(str(self._dir / "fills.sqlite"))
        self._init_sqlite()

        # JSONL handles
        self._equity_fh = open(self._dir / "equity.jsonl", "a", encoding="utf-8")
        self._events_fh = open(self._dir / "events.jsonl", "a", encoding="utf-8")

        logger.info("SessionLogger initialised: %s", self._dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_bar_1m(self, bar: MinuteBar) -> None:
        self._bars_1m.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "is_update": bar.is_update,
        })

    def log_bar_h1(self, bar: H1Bar, window_valid: bool) -> None:
        self._bars_h1.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "window_valid": window_valid,
        })
        # Flush parquet on each H1 close (low frequency, manageable size)
        self._flush_parquet()

    def log_prediction(self, event: WindowEvent, action: TradeAction) -> None:
        pred_class = int({"none": 0, "bull": 1, "bear": 2}[action.signal])
        self._predictions.append({
            "timestamp": event.h1_timestamp,
            "proba_none": float(action.proba[0]),
            "proba_bull": float(action.proba[1]),
            "proba_bear": float(action.proba[2]),
            "pred_class": pred_class,
            "signal": action.signal,
            "confidence": action.confidence,
            "skip_reason": action.skip_reason or "",
        })

    def log_order(
        self,
        order_id: str,
        action: TradeAction,
        qty: int,
        sl: float,
        tp: float,
        symbol: str = "SPY",
    ) -> None:
        side = "buy" if action.signal == "bull" else "sell"
        submitted_at = pd.Timestamp.now(tz="America/New_York").isoformat()
        self._conn.execute(
            """INSERT OR IGNORE INTO orders
               (order_id, submitted_at, symbol, side, qty, sl_price, tp_price, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, submitted_at, symbol, side, qty, sl, tp, "new"),
        )
        self._conn.commit()

    def log_fill(self, fill: FillEvent, slippage_model: SlippageModel) -> None:
        adj_pnl: Optional[float] = None
        # P&L is only calculable on a closing fill — track in executor, just record here
        commission = slippage_model.commission(fill.fill_qty)
        self._fills_conn.execute(
            """INSERT INTO fills
               (order_id, filled_at, fill_price, fill_qty, fill_type, adj_pnl, commission)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                fill.order_id,
                fill.filled_at.isoformat(),
                fill.fill_price,
                fill.fill_qty,
                fill.fill_type,
                adj_pnl,
                commission,
            ),
        )
        self._fills_conn.commit()

    def log_equity(self, timestamp: pd.Timestamp, account_snapshot: dict) -> None:
        record = {"timestamp": timestamp.isoformat(), **account_snapshot}
        self._equity_fh.write(json.dumps(record) + "\n")
        self._equity_fh.flush()

    def log_event(self, event_type: str, payload: dict) -> None:
        record = {
            "timestamp": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self._events_fh.write(json.dumps(record) + "\n")
        self._events_fh.flush()

    def close(self) -> None:
        """Flush all buffers and close file handles."""
        self._flush_parquet(force=True)
        self._conn.close()
        self._fills_conn.close()
        self._equity_fh.close()
        self._events_fh.close()
        logger.info("SessionLogger closed: %s", self._dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                submitted_at TEXT,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                sl_price REAL,
                tp_price REAL,
                status TEXT
            )
        """)
        self._conn.commit()

        self._fills_conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                filled_at TEXT,
                fill_price REAL,
                fill_qty INTEGER,
                fill_type TEXT,
                adj_pnl REAL,
                commission REAL
            )
        """)
        self._fills_conn.commit()

    def _flush_parquet(self, force: bool = False) -> None:
        """Write buffered rows to parquet files."""
        if self._bars_1m:
            _append_parquet(self._dir / "bars_1m.parquet", self._bars_1m)
            self._bars_1m = []

        if self._bars_h1:
            _append_parquet(self._dir / "bars_h1.parquet", self._bars_h1)
            self._bars_h1 = []

        if self._predictions:
            _append_parquet(self._dir / "predictions.parquet", self._predictions)
            self._predictions = []


def _append_parquet(path: Path, rows: list[dict]) -> None:
    """Append rows to a parquet file (or create it if it doesn't exist)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pq.read_table(str(path)).to_pandas()
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    pq.write_table(pa.Table.from_pandas(combined), str(path))


# ---------------------------------------------------------------------------
# FleetSessionLogger
# ---------------------------------------------------------------------------

class FleetSessionLogger:
    """Namespaced session logger for multi-model/strategy fleet sessions.

    Wraps one ``SessionLogger`` per ``(ticker, model, strategy)`` cell, stored
    in ``<base_dir>/<session_id>/<ticker>/<model>/<strategy>/``.

    Additionally provides:
    - ``log_bar(bar, source)`` — logs a ``MinuteBar`` tagged with a ``source``
      field (``"live"`` or ``"backfill"``).  Used by ``RestBackfiller`` (WS-B)
      and the live feed; written to a shared ``bars_1m_fleet.jsonl`` under the
      session root so the replay script can reconstruct decision windows.
    - ``log_intent(intent)`` — appends the ``OrderIntent`` dict (plus its cell
      key) to ``intents.jsonl`` under the cell sub-directory.  WS-G reads this
      file to replay-resolve P&L without a network connection.

    Parameters
    ----------
    session_id : str
        Unique session identifier.
    base_dir : str
        Root directory for all session logs.  Defaults to ``"logs/paper"``.

    Intent JSONL schema (one JSON object per line in ``intents.jsonl``)
    -------------------------------------------------------------------
    ::

        {
          "ticker": "SPY",
          "model": "cnn_lstm",
          "strategy": "ict_iofed",
          "h1_timestamp": "<ISO-8601>",
          "direction": 1,
          "signal": "bull",
          "confidence": 0.72,
          "entry": 452.30,
          "sl": 451.00,
          "tp": 454.90,
          "entry_type": "limit",
          "skip_reason": null
        }

    Bar JSONL schema (one JSON object per line in ``bars_1m_fleet.jsonl``)
    ----------------------------------------------------------------------
    ::

        {
          "symbol": "SPY",
          "timestamp": "<ISO-8601>",
          "open": 452.10,
          "high": 452.50,
          "low": 451.90,
          "close": 452.30,
          "volume": 12345.0,
          "is_update": false,
          "source": "backfill"
        }
    """

    def __init__(self, session_id: str, base_dir: str = "logs/paper") -> None:
        self._session_id = session_id
        self._root = Path(base_dir) / session_id
        self._root.mkdir(parents=True, exist_ok=True)

        # Cell loggers keyed by (ticker, model, strategy)
        self._cell_loggers: dict[tuple[str, str, str], SessionLogger] = {}

        # Shared bar JSONL (all tickers, tagged by source)
        self._bars_fh = open(self._root / "bars_1m_fleet.jsonl", "a", encoding="utf-8")

        # Shared intent JSONL at root level (mirrors per-cell intents for easy grep)
        self._intents_fh = open(self._root / "intents.jsonl", "a", encoding="utf-8")

        logger.info("FleetSessionLogger initialised: %s", self._root)

    # ------------------------------------------------------------------
    # Cell-namespace helpers
    # ------------------------------------------------------------------

    def _cell_key(self, ticker: str, model: str, strategy: str) -> tuple[str, str, str]:
        return (ticker, model, strategy)

    def cell_logger(self, ticker: str, model: str, strategy: str) -> SessionLogger:
        """Return (or lazily create) the ``SessionLogger`` for this cell."""
        key = self._cell_key(ticker, model, strategy)
        if key not in self._cell_loggers:
            cell_dir = str(self._root / ticker / model / strategy)
            self._cell_loggers[key] = SessionLogger(
                session_id=f"{ticker}__{model}__{strategy}",
                base_dir=str(self._root),
            )
            # Override the internal dir to the cell sub-path
            cell_path = self._root / ticker / model / strategy
            cell_path.mkdir(parents=True, exist_ok=True)
            self._cell_loggers[key]._dir = cell_path  # type: ignore[attr-defined]
        return self._cell_loggers[key]

    # ------------------------------------------------------------------
    # Fleet-level bar logging (shared across all tickers)
    # ------------------------------------------------------------------

    def log_bar(self, bar: MinuteBar, source: str = "live") -> None:
        """Log a ``MinuteBar`` tagged with ``source`` (``"live"`` or ``"backfill"``).

        Written to ``bars_1m_fleet.jsonl`` under the session root so that
        ``replay_fleet`` can reconstruct every decision window offline.

        Parameters
        ----------
        bar : MinuteBar
            The 1-minute bar to record.
        source : str
            ``"live"`` for bars arriving from the WebSocket stream;
            ``"backfill"`` for bars fetched by ``RestBackfiller`` at startup.
        """
        record = {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "is_update": bar.is_update,
            "source": source,
        }
        self._bars_fh.write(json.dumps(record) + "\n")
        self._bars_fh.flush()

    # ------------------------------------------------------------------
    # Intent persistence (replay input for WS-G)
    # ------------------------------------------------------------------

    def log_intent(self, intent: "Any") -> None:  # Any = OrderIntent (lazy import)
        """Append an ``OrderIntent`` to both the root ``intents.jsonl`` and the
        per-cell ``intents.jsonl`` so ``replay_fleet`` can locate them by cell.

        Parameters
        ----------
        intent : OrderIntent
            The fully-resolved intent from ``StrategyOrderPlanner``.
        """
        import math

        record = {
            "ticker": intent.ticker,
            "model": intent.model,
            "strategy": intent.strategy,
            "h1_timestamp": intent.h1_timestamp.isoformat(),
            "direction": intent.direction,
            "signal": intent.signal,
            "confidence": intent.confidence,
            "entry": None if (isinstance(intent.entry, float) and math.isnan(intent.entry)) else intent.entry,
            "sl": None if (isinstance(intent.sl, float) and math.isnan(intent.sl)) else intent.sl,
            "tp": None if (isinstance(intent.tp, float) and math.isnan(intent.tp)) else intent.tp,
            "entry_type": intent.entry_type,
            "skip_reason": intent.skip_reason,
        }
        line = json.dumps(record) + "\n"

        # Root-level JSONL (all cells)
        self._intents_fh.write(line)
        self._intents_fh.flush()

        # Per-cell JSONL
        cell_dir = self._root / intent.ticker / intent.model / intent.strategy
        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_intents = cell_dir / "intents.jsonl"
        with open(cell_intents, "a", encoding="utf-8") as fh:
            fh.write(line)

    # ------------------------------------------------------------------
    # Delegating wrappers for common per-cell operations
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        ticker: str,
        model: str,
        strategy: str,
        event: "Any",
        action: "Any",
    ) -> None:
        """Delegate to the cell logger's ``log_prediction``."""
        self.cell_logger(ticker, model, strategy).log_prediction(event, action)

    def log_order(
        self,
        ticker: str,
        model: str,
        strategy: str,
        order_id: str,
        action: "Any",
        qty: int,
        sl: float,
        tp: float,
    ) -> None:
        """Delegate to the cell logger's ``log_order``."""
        self.cell_logger(ticker, model, strategy).log_order(
            order_id, action, qty, sl, tp, symbol=ticker
        )

    def log_event(self, event_type: str, payload: dict) -> None:
        """Write a fleet-level event to the session root ``events.jsonl``."""
        events_path = self._root / "events.jsonl"
        record = {
            "timestamp": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush all cell loggers and close fleet-level file handles."""
        for cell_log in self._cell_loggers.values():
            cell_log.close()
        self._bars_fh.close()
        self._intents_fh.close()
        logger.info("FleetSessionLogger closed: %s", self._root)
