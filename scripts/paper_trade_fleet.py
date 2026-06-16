"""paper_trade_fleet.py — Multi-model × multi-ticker × multi-strategy fleet runner.

Implements the §4b configurability contract: all three axis flags define the
cartesian cell matrix; --live-subset overlays real PaperExecutor cells.
Default (no flags) = full matrix (4 models × 4 tickers × 4 strategies = 64 cells),
all sim.

Usage examples
--------------
# List cells without connecting
python scripts/paper_trade_fleet.py --list-cells \\
    --models lstm,cnn_lstm --tickers SPY,QQQ --strategies fixed_2r,tradinglab

# All-sim dry run (2 minutes, offline-safe)
python scripts/paper_trade_fleet.py \\
    --models lstm,cnn_lstm --tickers SPY,QQQ --strategies fixed_2r \\
    --max-minutes 2

# Real overlay on two cells (different tickers — unique-ticker rule)
python scripts/paper_trade_fleet.py \\
    --models lstm --tickers SPY,QQQ --strategies fixed_2r \\
    --live-subset "lstm:SPY:fixed_2r,lstm:QQQ:fixed_2r"

Defaults
--------
--models       lstm,cnn_lstm,transformer,xgboost
--tickers      SPY,QQQ,IWM,DIA
--strategies   fixed_2r,ict_iofed,ce_50pct,tradinglab
--fill-mode    conservative
--realistic    on (ATR floor + costs)
--warmup-days  12
--threshold    0.5
--tp-rr        2.0
--max-real-exposure  0.04
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path

# ---- project root on sys.path -----------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Load .env so ALPACA_* credentials are available (mirrors paper_trade.py).
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# ---- defaults ----------------------------------------------------------
_DEFAULT_MODELS = ["lstm", "cnn_lstm", "transformer", "xgboost"]
_DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]
_DEFAULT_STRATEGIES = ["fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"]
_CHECKPOINT_DIR = _ROOT / "checkpoints"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("fleet")


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_trade_fleet.py",
        description="Multi-model × multi-ticker × multi-strategy live fleet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- axis flags (§4b) -------------------------------------------
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names (bare name → multisym seed0, or "
            "'name:path' for explicit checkpoint). "
            f"Default: {','.join(_DEFAULT_MODELS)}"
        ),
    )
    p.add_argument(
        "--tickers",
        default=None,
        help=f"Comma-separated tickers. Default: {','.join(_DEFAULT_TICKERS)}",
    )
    p.add_argument(
        "--strategies",
        default=None,
        help=f"Comma-separated strategies. Default: {','.join(_DEFAULT_STRATEGIES)}",
    )

    # ---- live-subset overlay -----------------------------------------
    p.add_argument(
        "--live-subset",
        default="",
        dest="live_subset",
        help=(
            "Comma-separated 'model:ticker:strategy' triples for real paper "
            "orders. Empty (default) = all sim, no real orders placed."
        ),
    )

    # ---- cell listing ------------------------------------------------
    p.add_argument(
        "--list-cells",
        action="store_true",
        dest="list_cells",
        help="Print resolved cell matrix + executor type, then exit (no connect).",
    )

    # ---- fleet identity ----------------------------------------------
    p.add_argument(
        "--fleet-id",
        default=None,
        dest="fleet_id",
        help=(
            "Stable fleet identifier for durable state dir "
            "(default: short hash of resolved cell matrix)."
        ),
    )

    # ---- run parameters ----------------------------------------------
    p.add_argument(
        "--max-real-exposure",
        type=float,
        default=0.04,
        dest="max_real_exposure",
        help="Fleet-level cap on summed open real risk as fraction of equity. Default 0.04.",
    )
    p.add_argument(
        "--warmup-days",
        type=int,
        default=12,
        dest="warmup_days",
        help="Days of 1-min history to fetch for window warm-up. Default 12.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision confidence threshold. Default 0.5.",
    )
    p.add_argument(
        "--tp-rr",
        type=float,
        default=2.0,
        dest="tp_rr",
        help="Take-profit R-multiple. Default 2.0.",
    )
    p.add_argument(
        "--fill-mode",
        default="conservative",
        choices=["conservative", "optimistic"],
        dest="fill_mode",
        help="Sim fill mode (conservative = close-based). Default conservative.",
    )
    p.add_argument(
        "--realistic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ATR floor + cost realism guards (default: on).",
    )
    p.add_argument(
        "--session",
        default=None,
        help=(
            "Session identifier for log namespacing. "
            "Default: fleet-id + UTC timestamp."
        ),
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        dest="max_minutes",
        help="Graceful stop after N minutes (useful for smoke tests; default: run until interrupted).",
    )

    return p


# -----------------------------------------------------------------------
# Model token parsing: "name" or "name:path"
# -----------------------------------------------------------------------


def _normalize_live_subset_triple(triple: str) -> str:
    """Upper-case the ticker component of a ``model:ticker:strategy`` triple.

    Matches the upper-casing applied to ``--tickers`` (M5) so a lowercase
    live-subset ticker still resolves into the matrix.  Malformed triples are
    returned unchanged so ``build_cell_matrix`` raises its descriptive error.
    """
    parts = triple.split(":")
    if len(parts) != 3:
        return triple
    model, ticker, strategy = parts
    return f"{model}:{ticker.upper()}:{strategy}"


def _parse_model_tokens(raw: list[str]) -> tuple[list[str], dict[str, str]]:
    """Split model tokens into (names, checkpoint_overrides).

    Parameters
    ----------
    raw : list[str]
        E.g. ["lstm", "cnn_lstm:checkpoints/x.pt"]

    Returns
    -------
    (names, overrides)
        names       — list of arch names in order
        overrides   — dict[name -> explicit checkpoint path str]
    """
    names: list[str] = []
    overrides: dict[str, str] = {}
    for token in raw:
        if ":" in token:
            parts = token.split(":", 1)
            name, path = parts[0], parts[1]
            names.append(name)
            overrides[name] = path
        else:
            names.append(token)
    return names, overrides


# -----------------------------------------------------------------------
# Adapter loading with multisym resolution
# -----------------------------------------------------------------------


def _load_adapters(
    model_names: list[str],
    checkpoint_overrides: dict[str, str],
) -> dict[str, "ModelAdapter"]:  # type: ignore[type-arg]
    """Load one adapter per arch name (multisym seed0 default)."""
    from src.inspect.multisym import default_checkpoint_path
    from src.inspect.registry import load_adapters

    per_model: dict[str, str] = {}
    for name in model_names:
        if name in checkpoint_overrides:
            path = checkpoint_overrides[name]
            if not Path(path).is_absolute():
                path = str(_ROOT / path)
            per_model[name] = path
        else:
            cp = default_checkpoint_path(name, _CHECKPOINT_DIR)
            if cp is not None:
                per_model[name] = str(cp)
            else:
                _log.warning(
                    "No multisym seed0 checkpoint found for '%s' under %s — "
                    "adapter will use default checkpoint_dir discovery.",
                    name,
                    _CHECKPOINT_DIR,
                )

    adapter_list = load_adapters(
        model_names,
        checkpoint_dir=_CHECKPOINT_DIR,
        checkpoint_paths=per_model if per_model else None,
    )
    return {a.name: a for a in adapter_list}


# -----------------------------------------------------------------------
# Fleet-id derivation
# -----------------------------------------------------------------------


def _derive_fleet_id(models: list[str], tickers: list[str], strategies: list[str]) -> str:
    """Stable short hash of the resolved cell matrix."""
    raw = ",".join(sorted(models)) + "|" + ",".join(sorted(tickers)) + "|" + ",".join(sorted(strategies))
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


# -----------------------------------------------------------------------
# Build executor registry from cell specs
# -----------------------------------------------------------------------


def _build_executor_registry(
    specs: list["CellSpec"],  # type: ignore[type-arg]
    strategy_configs: dict[str, "ExitConfig"],  # type: ignore[type-arg]
    fleet_state: "FleetState",  # type: ignore[type-arg]
    fleet_logger: "FleetSessionLogger",  # type: ignore[type-arg]
    trading_client: "TradingClient | None",  # type: ignore[type-arg]
    fill_router: "FillRouter",  # type: ignore[type-arg]
) -> "ExecutorRegistry":  # type: ignore[type-arg]
    from src.live.execution import PaperExecutor
    from src.live.fleet import ExecutorRegistry, make_cell_key
    from src.live.sim_executor import SimFillExecutor

    registry: ExecutorRegistry = {}
    for spec in specs:
        key = make_cell_key(spec.ticker, spec.model, spec.strategy)
        cfg = strategy_configs[spec.strategy]

        if spec.executor_type == "real":
            if trading_client is None:
                raise RuntimeError(
                    f"Real executor requested for cell '{key}' but no TradingClient "
                    "is available. Set ALPACA_PAPER=true and provide credentials."
                )
            executor = PaperExecutor(
                trading_client=trading_client,
                symbol=spec.ticker,
                tp_rr=cfg.tp_rr,
                logger_sink=fleet_logger,
            )
            fill_router.register(spec.ticker, executor)
        else:
            executor = SimFillExecutor(
                realism=cfg,
                state=fleet_state,
                cell_key=key,
            )

        registry[key] = executor

    return registry


# -----------------------------------------------------------------------
# Warm-up all tickers
# -----------------------------------------------------------------------


def _warmup_tickers(
    tickers: list[str],
    builders: "dict[tuple[str, str], LiveWindowBuilder]",  # type: ignore[type-arg]
    api_key: str,
    secret_key: str,
    warmup_days: "int | None",
    fleet_logger: "FleetSessionLogger",  # type: ignore[type-arg]
) -> list[str]:
    """Warm up each (ticker, tf) builder; return tickers where all TFs passed.

    For each unique (ticker, tf_token) builder pair, a separate backfill
    is run.  The warmup depth is derived from the builder's own timeframe
    (warmup_days arg is an optional override).  A ticker is considered
    ready only when ALL its TF builders have h1_count >= window_size.
    """
    from src.data.timeframe import Timeframe
    from src.live.backfill import RestBackfiller

    backfiller = RestBackfiller(api_key=api_key, secret_key=secret_key, feed="iex")
    # Track per-ticker pass/fail across TFs
    ticker_ok: dict[str, bool] = {t: True for t in tickers}

    for (ticker, tf_tok), builder in builders.items():
        if ticker not in ticker_ok:
            continue
        tf = Timeframe.from_token(tf_tok)
        ws = builder._window_size
        try:
            h1_count = backfiller.warm(
                symbol=ticker,
                builder=builder,
                warmup_days=warmup_days,
                logger=fleet_logger,
                tf=tf,
                window_size=ws,
            )
        except AssertionError as exc:
            _log.warning("Ticker %s [%s]: warm-up check failed: %s", ticker, tf_tok, exc)
            ticker_ok[ticker] = False
            continue

        if h1_count < ws:
            _log.warning(
                "Ticker %s [%s]: warm-up returned only %d bars (need %d). "
                "Skipping for this session.",
                ticker, tf_tok, h1_count, ws,
            )
            ticker_ok[ticker] = False
        else:
            _log.info("Ticker %s [%s]: warm-up complete (%d bars).", ticker, tf_tok, h1_count)

    ok_tickers = [t for t in tickers if ticker_ok.get(t, False)]
    return ok_tickers


# -----------------------------------------------------------------------
# Real-fill stream (C3) — only opened when the matrix has real cells
# -----------------------------------------------------------------------


async def _run_fill_stream(
    api_key: str,
    secret_key: str,
    fill_router: "FillRouter",  # type: ignore[type-arg]
) -> None:
    """Subscribe to the Alpaca TradingStream and route fills to the FillRouter.

    Mirrors ``scripts/paper_trade.py::_run_fill_stream`` but fans every fill
    into ``FillRouter.on_fill`` (symbol-routed), which decrements the owning
    PaperExecutor's open count, runs its kill-switch check, and releases the
    fleet exposure bucket on closing fills.
    """
    try:
        import pandas as pd
        from alpaca.trading.stream import TradingStream

        from src.live.logger import FillEvent

        trading_stream = TradingStream(api_key, secret_key, paper=True)

        async def fill_handler(data) -> None:
            if data.event in ("fill", "partial_fill"):
                fill = FillEvent(
                    order_id=str(data.order.id),
                    filled_at=pd.Timestamp.now(tz="America/New_York"),
                    fill_price=float(data.order.filled_avg_price or 0),
                    fill_qty=int(data.order.filled_qty or 0),
                    fill_type=data.event,
                    side=str(data.order.side.value),
                    symbol=str(getattr(data.order, "symbol", "") or "").upper() or None,
                )
                fill_router.on_fill(fill)

        trading_stream.subscribe_trade_updates(fill_handler)
        await trading_stream._run_forever()
    except Exception as exc:  # pragma: no cover - network/SDK guard
        _log.warning("Fleet fill stream error: %s", exc)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Parse axis flags -------------------------------------------
    raw_models = (
        [t.strip() for t in args.models.split(",") if t.strip()]
        if args.models
        else _DEFAULT_MODELS
    )
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else _DEFAULT_TICKERS
    )
    strategies = (
        [s.strip() for s in args.strategies.split(",") if s.strip()]
        if args.strategies
        else _DEFAULT_STRATEGIES
    )
    live_subset_raw = (
        [
            _normalize_live_subset_triple(t.strip())
            for t in args.live_subset.split(",")
            if t.strip()
        ]
        if args.live_subset
        else []
    )

    # ---- Parse model tokens -----------------------------------------
    model_names, checkpoint_overrides = _parse_model_tokens(raw_models)

    # ---- Validate cell matrix (pure, no network) --------------------
    from src.live.fleet import build_cell_matrix, format_cell_list, resolve_tf_from_meta
    from src.inspect.multisym import default_checkpoint_path

    # Resolve each model's timeframe from its checkpoint .meta.json so cells
    # adopt the TF the model was trained on (otherwise they default to "h1"
    # and the FleetRouter TF-guard rejects a 5m/15m checkpoint).
    model_tfs: dict[str, str] = {}
    for name in model_names:
        path = checkpoint_overrides.get(name)
        if path is None:
            cp = default_checkpoint_path(name, _CHECKPOINT_DIR)
            path = str(cp) if cp is not None else None
        elif not Path(path).is_absolute():
            path = str(_ROOT / path)
        model_tfs[name] = resolve_tf_from_meta(path)

    specs = build_cell_matrix(
        models=model_names,
        tickers=tickers,
        strategies=strategies,
        live_subset=live_subset_raw,
        model_tfs=model_tfs,
    )

    # ---- --list-cells exit ------------------------------------------
    if args.list_cells:
        print(format_cell_list(specs))
        return

    # ---- Fleet-id and session id ------------------------------------
    fleet_id = args.fleet_id or _derive_fleet_id(model_names, tickers, strategies)
    import datetime
    ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = args.session or f"{fleet_id}_{ts_str}"
    _log.info("Fleet-id: %s  Session: %s", fleet_id, session_id)

    # ---- Assert ALPACA_PAPER=true before any real submit ------------
    has_real = any(s.executor_type == "real" for s in specs)
    if has_real:
        alpaca_paper = os.environ.get("ALPACA_PAPER", "").lower()
        if alpaca_paper != "true":
            raise RuntimeError(
                "Real cells are in the matrix but ALPACA_PAPER is not set to 'true'. "
                "Set ALPACA_PAPER=true in your environment before using --live-subset."
            )

    # ---- Credentials ------------------------------------------------
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        _log.warning(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
            "Warm-up and stream will fail unless stubs are injected."
        )

    # ---- Strategy configs -------------------------------------------
    from src.strategy.exits import ExitConfig

    strategy_configs: dict[str, ExitConfig] = {}
    for s in strategies:
        strategy_configs[s] = ExitConfig(
            strategy=s,
            tp_rr=args.tp_rr,
            fill_mode=args.fill_mode,
            min_stop_atr_k=0.5 if args.realistic else None,
            commission_per_share=0.005 if args.realistic else 0.0,
            slippage_ticks=1 if args.realistic else 0,
        )

    # ---- Load adapters ----------------------------------------------
    _log.info("Loading adapters: %s", model_names)
    adapters = _load_adapters(model_names, checkpoint_overrides)

    # ---- Fleet state ------------------------------------------------
    from src.live.fleet_state import FleetState

    fleet_state = FleetState(fleet_id=fleet_id, base_dir=_ROOT / "logs")

    # ---- Fleet logger -----------------------------------------------
    from src.live.logger import FleetSessionLogger

    fleet_logger = FleetSessionLogger(session_id=session_id, base_dir=str(_ROOT / "logs" / "paper"))

    # ---- Trading client (only if real cells) ------------------------
    trading_client = None
    if has_real:
        from alpaca.trading.client import TradingClient

        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )

    # ---- Fill router ------------------------------------------------
    from src.live.fill_router import FillRouter

    fill_router = FillRouter(max_real_exposure=args.max_real_exposure)

    # ---- Build executor registry ------------------------------------
    registry = _build_executor_registry(
        specs=specs,
        strategy_configs=strategy_configs,
        fleet_state=fleet_state,
        fleet_logger=fleet_logger,
        trading_client=trading_client,
        fill_router=fill_router,
    )

    # ---- Build FleetRouter ------------------------------------------
    from src.live.fleet import FleetRouter

    router = FleetRouter(
        tickers=tickers,
        adapters=adapters,
        strategy_configs=strategy_configs,
        executor_registry=registry,
        logger=fleet_logger,
        specs=specs,
        threshold=args.threshold,
        fill_router=fill_router if has_real else None,
    )

    # ---- Warm up tickers --------------------------------------------
    ok_tickers = _warmup_tickers(
        tickers=tickers,
        builders=router.builders,
        api_key=api_key,
        secret_key=secret_key,
        warmup_days=args.warmup_days,
        fleet_logger=fleet_logger,
    )
    if not ok_tickers:
        _log.error("No tickers passed warm-up. Aborting.")
        fleet_logger.close()
        return

    _log.info(
        "Matrix: %d cells | tickers: %s | models: %s | strategies: %s",
        len(specs),
        ok_tickers,
        model_names,
        strategies,
    )

    # ---- Stream + run loop ------------------------------------------
    from src.live.stream import AlpacaBarStream

    async def _run() -> None:
        stream = AlpacaBarStream(
            api_key=api_key,
            secret_key=secret_key,
            symbols=ok_tickers,
            on_bar=router.on_bar,
            paper=True,
        )

        # C3: open a trade-updates stream so real PaperExecutors receive fills
        # (open-count decrement + kill switch + exposure release).  Sim-only
        # runs never open it.
        tasks: list[asyncio.Task] = [
            asyncio.create_task(stream.start(), name="bar-stream"),
        ]
        if has_real:
            tasks.append(
                asyncio.create_task(
                    _run_fill_stream(api_key, secret_key, fill_router),
                    name="fill-stream",
                )
            )

        async def _stopper() -> None:
            """Wait for max_minutes then cancel all sibling tasks."""
            await asyncio.sleep(args.max_minutes * 60)  # type: ignore[operator]
            _log.info("--max-minutes reached; stopping stream.")
            # Stop the SDK WebSocket so bar-stream's _run_forever returns.
            await stream.stop()
            # Cancel all other tasks (e.g. fill-stream) so the event loop exits.
            for t in tasks:
                if not t.done():
                    t.cancel()

        if args.max_minutes is not None:
            stopper_task = asyncio.create_task(_stopper(), name="stopper")
            all_tasks = tasks + [stopper_task]
        else:
            all_tasks = tasks

        try:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        finally:
            # Cancel any still-running tasks (e.g. on KeyboardInterrupt) so
            # the event loop drains cleanly before on_session_close runs.
            for t in all_tasks:
                if not t.done():
                    t.cancel()
            # Drain cancelled tasks — swallow CancelledError.
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        _log.info("Keyboard interrupt — shutting down fleet.")
    finally:
        router.on_session_close()
        _log.info("Fleet session %s complete.", session_id)


if __name__ == "__main__":
    main()
