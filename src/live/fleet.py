"""fleet.py — FleetRouter: cell-matrix construction and bar fan-out.

§4b configurability contract
-----------------------------
The run is defined by three independent set-flags whose cartesian product
is the cell matrix:

    --models      subset of {lstm, cnn_lstm, transformer, xgboost}
    --tickers     subset of {SPY, QQQ, IWM, DIA}
    --strategies  subset of {fixed_2r, ict_iofed, ce_50pct, tradinglab}

Every (model, ticker, strategy) triple in the product is a *cell*.
Cells in ``--live-subset`` are backed by a real ``PaperExecutor``; all
others get a ``SimFillExecutor``.  The two executor types are mutually
exclusive per cell (real XOR sim — never double-counted).

Legality rules (hard-validated BEFORE any network call)
--------------------------------------------------------
1. No set may be empty → ``ValueError``.
2. Every live-subset triple must be in the resolved matrix → ``ValueError``.
3. Real cells must each own a unique ticker (paper account nets per symbol)
   → ``ValueError`` naming the duplicate ticker.

Open/Closed principle
---------------------
Adding a new model or strategy = a config entry.  ``FleetRouter.on_bar``
iterates the registry; no ``if model == "x"`` branching.

Pure, testable matrix function
-------------------------------
``build_cell_matrix`` is a pure function with no I/O.  It raises
``ValueError`` on illegal inputs and returns an ``ExecutorRegistry`` that
maps ``cell_key -> "real" | "sim"``.  The ``FleetRouter`` constructor
takes an already-built registry plus live objects; it never validates
legality itself.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import json
import pathlib

from src.data.timeframe import H1, Timeframe
from src.data.timeframe import warmup_days as _tf_warmup_days
from src.inspect.base import ModelAdapter
from src.live.execution import PaperExecutor
from src.live.decision import SingleModelDecision, TradeAction
from src.live.executor_base import Executor
from src.live.fleet_state import FleetState
from src.live.logger import FleetSessionLogger
from src.live.order_plan import OrderIntent, StrategyOrderPlanner
from src.live.stream import MinuteBar
from src.live.window_builder import LiveWindowBuilder, WindowEvent
from src.strategy.exits import ExitConfig

if TYPE_CHECKING:
    from src.live.fill_router import FillRouter

_log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------

# cell_key = "ticker:model:strategy"
CellKey = str

# ExecutorRegistry: cell_key -> Executor instance
ExecutorRegistry = dict[CellKey, Executor]


# -----------------------------------------------------------------------
# Pure cell-matrix builder (unit-testable, no I/O)
# -----------------------------------------------------------------------


def make_cell_key(ticker: str, model: str, strategy: str) -> CellKey:
    """Canonical cell key format: ``"ticker:model:strategy"``."""
    return f"{ticker}:{model}:{strategy}"


@dataclass
class CellSpec:
    """Metadata for one resolved cell (used by --list-cells and tests)."""

    ticker: str
    model: str
    strategy: str
    executor_type: str  # "real" | "sim"
    tf: str = "h1"     # timeframe token resolved from checkpoint meta.json

    @property
    def key(self) -> CellKey:
        return make_cell_key(self.ticker, self.model, self.strategy)


def resolve_tf_from_meta(checkpoint_path: "str | pathlib.Path | None") -> str:
    """Read the timeframe token from a checkpoint's ``.meta.json`` file.

    The meta.json is expected at ``<checkpoint_path>.meta.json`` or, if
    *checkpoint_path* is a directory, at ``<dir>/<stem>.meta.json`` (not
    supported; directories → "h1").

    Missing file, missing key, or any read error → "h1" (plan §3.10).
    Validates the token against the Timeframe registry; invalid token → "h1".

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to the checkpoint file (or None).

    Returns
    -------
    str
        Timeframe token, e.g. ``"h1"``, ``"15m"``, ``"5m"``.
    """
    if checkpoint_path is None:
        return "h1"
    try:
        p = pathlib.Path(checkpoint_path)
        meta_path = p.with_suffix(".meta.json") if p.suffix else p / ".meta.json"
        if not meta_path.exists():
            # Also try <name>.meta.json in same directory
            meta_path = p.parent / (p.stem + ".meta.json")
        if not meta_path.exists():
            return "h1"
        with open(meta_path) as f:
            data = json.load(f)
        token = data.get("timeframe", "h1")
        # Validate against registry
        Timeframe.from_token(token)
        return token
    except Exception:
        return "h1"


def build_cell_matrix(
    models: list[str],
    tickers: list[str],
    strategies: list[str],
    live_subset: list[str],  # each entry: "model:ticker:strategy"
    model_tfs: Optional[dict[str, str]] = None,  # model_name -> tf_token; missing → "h1"
) -> list[CellSpec]:
    """Resolve and validate the cell matrix.

    This is a **pure function** (no I/O, no network, no executor
    instantiation).  It only validates legality and returns the resolved
    list of ``CellSpec`` objects.  Callers build the actual executor
    objects from the returned specs.

    Parameters
    ----------
    models : list[str]
        Model names to include.
    tickers : list[str]
        Ticker symbols to include.
    strategies : list[str]
        Strategy names to include.
    live_subset : list[str]
        Triples ``"model:ticker:strategy"`` that should use real
        ``PaperExecutor`` rather than ``SimFillExecutor``.  Empty list
        means all-sim.

    Returns
    -------
    list[CellSpec]
        One spec per cell in the cartesian product, each tagged with
        ``executor_type = "real" | "sim"``.

    Raises
    ------
    ValueError
        On any legality violation (empty set, out-of-matrix triple,
        duplicate real ticker).
    """
    # ---- Rule 1: no empty set ----------------------------------------
    for label, items in (("models", models), ("tickers", tickers), ("strategies", strategies)):
        if not items:
            raise ValueError(
                f"Cell matrix requires at least one {label}; got empty list."
            )

    # ---- Build full cartesian product -----------------------------------
    full_matrix: set[CellKey] = {
        make_cell_key(t, m, s)
        for m, t, s in itertools.product(models, tickers, strategies)
    }

    # ---- Parse and validate live-subset triples -------------------------
    real_keys: set[CellKey] = set()
    for triple in live_subset:
        parts = triple.strip().split(":")
        if len(parts) != 3:
            raise ValueError(
                f"live-subset entry must be 'model:ticker:strategy'; got {triple!r}"
            )
        model, ticker, strategy = parts
        key = make_cell_key(ticker, model, strategy)

        # Rule 2: triple must be in the matrix
        if key not in full_matrix:
            raise ValueError(
                f"live-subset triple '{triple}' is not in the resolved cell matrix "
                f"(models={models}, tickers={tickers}, strategies={strategies})."
            )
        real_keys.add(key)

    # ---- Rule 3: real cells must have unique tickers --------------------
    real_tickers: list[str] = []
    seen_real_tickers: set[str] = set()
    for key in real_keys:
        ticker = key.split(":")[0]
        if ticker in seen_real_tickers:
            raise ValueError(
                f"Duplicate real ticker '{ticker}' among live-subset cells. "
                "The paper account nets per symbol — each ticker may have at most "
                "one real executor. Use --live-subset to select a unique ticker per cell."
            )
        seen_real_tickers.add(ticker)
        real_tickers.append(ticker)

    # ---- Assemble CellSpec list -----------------------------------------
    _model_tfs = model_tfs or {}
    specs: list[CellSpec] = []
    # Stable ordering: ticker, model, strategy
    for ticker, model, strategy in itertools.product(tickers, models, strategies):
        key = make_cell_key(ticker, model, strategy)
        executor_type = "real" if key in real_keys else "sim"
        tf_token = _model_tfs.get(model, "h1")
        # Validate; fall back to "h1" on unknown token
        try:
            Timeframe.from_token(tf_token)
        except ValueError:
            tf_token = "h1"
        specs.append(CellSpec(
            ticker=ticker,
            model=model,
            strategy=strategy,
            executor_type=executor_type,
            tf=tf_token,
        ))

    return specs


def format_cell_list(specs: list[CellSpec]) -> str:
    """Format the cell matrix as a human-readable table for ``--list-cells``."""
    lines: list[str] = [
        f"{'TICKER':<8} {'MODEL':<14} {'STRATEGY':<14} {'TF':<6} {'EXECUTOR':<8}",
        "-" * 56,
    ]
    for spec in specs:
        lines.append(
            f"{spec.ticker:<8} {spec.model:<14} {spec.strategy:<14} {spec.tf:<6} {spec.executor_type:<8}"
        )
    lines.append(f"\nTotal cells: {len(specs)}")
    real_count = sum(1 for s in specs if s.executor_type == "real")
    sim_count = len(specs) - real_count
    lines.append(f"  real (PaperExecutor):   {real_count}")
    lines.append(f"  sim  (SimFillExecutor): {sim_count}")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# FleetRouter
# -----------------------------------------------------------------------


class FleetRouter:
    """Routes live 1-min bars across all (ticker, model, strategy) cells.

    One ``LiveWindowBuilder`` per ``(ticker, tf_token)`` pair; one
    ``SingleModelDecision`` per ``(ticker, model)``.  On each valid
    ``WindowEvent``, every model for the matching ticker+tf decides →
    for every strategy a ``StrategyOrderPlanner`` plans an ``OrderIntent``
    dispatched to the cell's ``Executor``.

    Open/Closed: adding a model or strategy means adding an entry to the
    constructor's adapter/config dicts, not editing this class.

    Backward-compat guarantee: a pure-H1 fleet (all cells tf="h1") has
    exactly one builder per ticker, keyed ``(ticker, "h1")``, and all
    routing paths are byte-identical to the original H1-only implementation.

    Parameters
    ----------
    tickers : list[str]
        All tickers in the fleet.
    adapters : dict[str, ModelAdapter]
        Model name → adapter instance.
    strategy_configs : dict[str, ExitConfig]
        Strategy name → ``ExitConfig``.
    executor_registry : ExecutorRegistry
        ``cell_key -> Executor`` (pre-built; real or sim per cell).
    logger : FleetSessionLogger
        Fleet-level session logger.
    specs : list[CellSpec], optional
        Full cell spec list (used to build per-cell TF routing table).
        When None, all cells default to H1.
    threshold : float
        Decision confidence threshold.  Default 0.5.
    fill_router : FillRouter, optional
        When provided, REAL cells are gated through
        ``fill_router.can_enter(...)`` before dispatch and
        ``fill_router.record_entry(...)`` after a confirmed submit, enforcing
        the fleet-level ``--max-real-exposure`` cap (C2).  Sim cells are never
        gated.  ``None`` (default, all-sim runs) bypasses the gate entirely.
    """

    def __init__(
        self,
        tickers: list[str],
        adapters: dict[str, ModelAdapter],
        strategy_configs: dict[str, ExitConfig],
        executor_registry: ExecutorRegistry,
        logger: FleetSessionLogger,
        specs: Optional[list[CellSpec]] = None,
        threshold: float = 0.5,
        fill_router: "FillRouter | None" = None,
    ) -> None:
        self._tickers = list(tickers)
        self._adapters = dict(adapters)
        self._strategy_configs = dict(strategy_configs)
        self._registry = executor_registry
        self._logger = logger
        self._threshold = threshold
        self._fill_router = fill_router

        # Build per-cell TF routing table: cell_key -> tf_token.
        # H2: before a cell joins routing, assert its declared tf matches the
        # timeframe stamped in its model's checkpoint meta.json — a model must
        # only ever trade on the TF it was trained on (missing meta → "h1").
        self._cell_tf: dict[str, str] = {}
        if specs:
            for spec in specs:
                adapter = self._adapters.get(spec.model)
                ckpt = getattr(adapter, "checkpoint_path", None) if adapter else None
                if ckpt is not None:
                    meta_tf = resolve_tf_from_meta(ckpt)
                    if meta_tf != spec.tf:
                        raise ValueError(
                            f"Fleet cell {spec.key!r}: model {spec.model!r} declares "
                            f"tf={spec.tf!r} but its checkpoint meta says tf={meta_tf!r}. "
                            "A model must trade only on the timeframe it was trained on."
                        )
                self._cell_tf[spec.key] = spec.tf

        # Collect unique (ticker, tf_token) pairs needed across all cells.
        # For H1-only fleets this is one pair per ticker — builder keyed (ticker,"h1").
        unique_ticker_tfs: set[tuple[str, str]] = set()
        for ticker in tickers:
            # Gather all TFs used by any model for this ticker
            tfs_for_ticker: set[str] = set()
            if specs:
                for spec in specs:
                    if spec.ticker == ticker:
                        tfs_for_ticker.add(spec.tf)
            if not tfs_for_ticker:
                tfs_for_ticker = {"h1"}  # default
            for tf_tok in tfs_for_ticker:
                unique_ticker_tfs.add((ticker, tf_tok))

        # One LiveWindowBuilder per (ticker, tf_token)
        self._builders: dict[tuple[str, str], LiveWindowBuilder] = {}
        for ticker, tf_tok in unique_ticker_tfs:
            tf = Timeframe.from_token(tf_tok)
            self._builders[(ticker, tf_tok)] = LiveWindowBuilder(timeframe=tf)

        # Per-ticker set of tf tokens (for on_bar fan-out)
        self._ticker_tfs: dict[str, set[str]] = {}
        for ticker, tf_tok in unique_ticker_tfs:
            self._ticker_tfs.setdefault(ticker, set()).add(tf_tok)

        # One SingleModelDecision per (ticker, model)
        self._decisions: dict[tuple[str, str], SingleModelDecision] = {
            (t, m): SingleModelDecision(adapter, threshold)
            for t in tickers
            for m, adapter in adapters.items()
        }

        # One StrategyOrderPlanner per strategy
        self._planners: dict[str, StrategyOrderPlanner] = {
            s: StrategyOrderPlanner(cfg)
            for s, cfg in strategy_configs.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def builders(self) -> dict[tuple[str, str], LiveWindowBuilder]:
        """Builders keyed by (ticker, tf_token) — exposed for backfill warm-up."""
        return self._builders

    def on_bar(self, bar: MinuteBar) -> None:
        """Route a 1-min bar through the full cell matrix.

        Steps:
        1. Feed bar to every executor's ``on_bar`` (uniform ABC contract).
        2. Feed bar to EVERY builder whose ticker matches (all TFs for that ticker).
        3. For each WindowEvent emitted, for each model+strategy cell keyed to
           that (ticker, tf) decide → plan an intent → dispatch to executor + log.
        """
        ticker = bar.symbol

        # Log live bar (tagged "live")
        self._logger.log_bar(bar, source="live")

        # Step 1: feed every executor (for limit timeout + sim intrabar resolution)
        for executor in self._registry.values():
            executor.on_bar(bar)

        # Step 2: feed the ticker's builders (all TFs)
        if ticker not in self._ticker_tfs:
            _log.debug("FleetRouter.on_bar: unknown ticker %s — ignored", ticker)
            return

        for tf_tok in self._ticker_tfs[ticker]:
            builder = self._builders[(ticker, tf_tok)]
            event: Optional[WindowEvent] = builder.on_bar(bar)
            if event is not None:
                # Step 3: valid (or skip-reason) window event — fan out
                self._on_window_event(ticker, tf_tok, event)

    def on_session_close(self) -> None:
        """Propagate session-close to all executors and flush the logger."""
        for executor in self._registry.values():
            executor.on_session_close()
        self._logger.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch(self, executor: Executor, intent: OrderIntent, ticker: str) -> bool:
        """Dispatch an intent to one executor, applying the real-cell gate.

        Sim cells (or when no ``FillRouter`` is wired) dispatch unconditionally.
        Real cells (``PaperExecutor``) are gated through
        ``FillRouter.can_enter`` using the per-trade dollar risk
        (``equity * risk_pct``); a confirmed submit (open_count increment) then
        calls ``record_entry`` so the cap accumulates (C2).

        Returns ``True`` if the intent was dispatched, ``False`` if the gate
        blocked it.
        """
        if self._fill_router is None or not isinstance(executor, PaperExecutor):
            executor.on_intent(intent)
            return True

        if intent.skip_reason is not None:
            # Degenerate intents are dropped by the executor anyway; no gate.
            executor.on_intent(intent)
            return True

        equity = executor.current_equity()
        if equity is None:
            _log.warning("Exposure gate: no equity for %s — blocking entry", ticker)
            return False

        risk_amount = equity * executor.risk_pct
        if not self._fill_router.can_enter(ticker, risk_amount, equity):
            _log.info("Exposure gate BLOCKED real entry for %s", ticker)
            return False

        before = executor.open_count()
        executor.on_intent(intent)
        # Record only when the submit actually opened a position.
        if executor.open_count() > before:
            self._fill_router.record_entry(ticker, risk_amount)
        return True

    def _on_window_event(self, ticker: str, tf_tok: str = "h1", event: Optional[WindowEvent] = None) -> None:
        """Fan out a WindowEvent to all (model, strategy) cells for this (ticker, tf).

        Backward-compat: existing tests call ``_on_window_event(ticker, event)``
        with 2 positional args.  The overload is detected by checking whether
        ``tf_tok`` is a ``WindowEvent`` instance (old calling convention).
        """
        # Detect old two-arg call: _on_window_event(ticker, event)
        if isinstance(tf_tok, WindowEvent):
            event = tf_tok
            tf_tok = "h1"
        assert event is not None
        for model_name, decision in [
            (m, d) for (t, m), d in self._decisions.items() if t == ticker
        ]:
            # Only dispatch to cells whose tf matches the builder that fired
            # (models may have different TFs on the same ticker): check if this
            # model on this ticker has any cell with this TF.
            model_has_tf = any(
                self._cell_tf.get(make_cell_key(ticker, model_name, s), "h1") == tf_tok
                for s in self._planners
            )
            if not model_has_tf:
                continue
            action: TradeAction = decision.decide(event)

            # Skip if no signal (none direction)
            if action.signal == "none":
                continue

            direction = 1 if action.signal == "bull" else 2

            for strategy_name, planner in self._planners.items():
                cell_key = make_cell_key(ticker, model_name, strategy_name)
                executor = self._registry.get(cell_key)
                if executor is None:
                    continue

                intent: OrderIntent = planner.plan(
                    window_raw=event.raw_window,
                    direction=direction,
                    h1_timestamp=event.h1_timestamp,
                    ticker=ticker,
                    model=model_name,
                    signal=action.signal,
                    confidence=action.confidence,
                )

                # Persist intent for offline replay (WS-G)
                self._logger.log_intent(intent)

                # Dispatch to the cell's executor — REAL cells pass the
                # fleet-level exposure gate first (C2).
                if not self._dispatch(executor, intent, ticker):
                    continue

                _log.debug(
                    "FleetRouter: dispatched intent cell=%s ts=%s signal=%s skip=%s",
                    cell_key,
                    event.h1_timestamp,
                    action.signal,
                    intent.skip_reason,
                )
