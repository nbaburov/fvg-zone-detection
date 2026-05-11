# Plan: Live Paper-Trading Harness
**Date:** 11-May-26
**Status:** DRAFT
**Prior research:** `.nb-suite/research/11-May-26/paper-trading-harness.md`

---

## Context

Single-model-per-process paper trading harness for the SMC FVG project. No voting. No consensus. One process = one `ModelAdapter` trading one symbol. Parallel deployment = user runs the script N times in separate terminals. The harness reuses `ModelAdapter` ABC from `src/inspect/base.py` unchanged — `predict_proba((1, 60, 5))` interface is the contract.

All decisions from research are locked in. This plan adds implementation detail only.

---

## File Map

```
src/live/
  __init__.py
  stream.py           # Alpaca WebSocket 1-min bar subscriber
  window_builder.py   # RTH H1 accumulator → 60-bar window emitter
  decision.py         # single-model confidence threshold → TradeAction
  execution.py        # bracket order submit + position tracking + kill switch
  slippage.py         # 1-tick slippage overlay for P&L accounting
  logger.py           # session logger: parquet + sqlite + jsonl
  replay.py           # offline replay from saved bars

scripts/
  paper_trade.py      # CLI entrypoint
```

---

## Module Specifications

### `src/live/stream.py` — `AlpacaBarStream`

Wraps `alpaca_trade_api.stream.Stream` (or `alpaca-py` `StockDataStream`). Connects to Alpaca WebSocket, subscribes to 1-min bars for one symbol.

**Interface:**
```python
class AlpacaBarStream:
    def __init__(self, api_key: str, secret_key: str, symbol: str, on_bar: Callable, paper: bool = True): ...
    async def start(self) -> None: ...  # blocks; call in asyncio.run()
    async def stop(self) -> None: ...
```

**Behaviour:**
- On each `bar` event: call `on_bar(bar: MinuteBar)` synchronously in the event loop.
- On `updated_bar` event: call `on_bar` again with `is_update=True` flag — the window builder handles in-place replacement.
- WebSocket disconnect: exponential backoff reconnect (delays: 1s, 2s, 4s, 8s, 16s, cap 60s). After 5 consecutive failures, emit `STREAM_FATAL` event and halt. On reconnect, call `on_reconnect()` callback so `WindowBuilder` can trigger REST gap-fill.
- Timezone: all timestamps normalised to `America/New_York` on receipt.

**`MinuteBar` dataclass:**
```python
@dataclass
class MinuteBar:
    symbol: str
    timestamp: pd.Timestamp   # tz=America/New_York
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_update: bool = False
```

---

### `src/live/window_builder.py` — `LiveWindowBuilder`

Accumulates 1-min bars, resamples to RTH-anchored H1, emits 60-bar windows.

**RTH definition (non-negotiable — must match training pipeline):**
- Include: `09:30 <= bar.timestamp.time() < 16:00` ET
- Reject pre-market and AH bars before inserting into buffer.
- H1 boundaries anchored at `:30`: `09:30`, `10:30`, `11:30`, `12:30`, `13:30`, `14:30`, `15:30` (6 H1 bars per full session, 3 on NYSE half-day).
- H1 close fires when the first 1-min bar with `timestamp >= next_h1_boundary` arrives. At that point, the completed H1 is assembled from all 1-min bars in `[boundary - 1h, boundary)`.

**H1 assembly:**
```
h1_open   = first 1m bar in window: open
h1_high   = max(high) over all 1m bars
h1_low    = min(low) over all 1m bars
h1_close  = last 1m bar in window: close
h1_volume = sum(volume) over all 1m bars
h1_timestamp = boundary start (label='left', closed='left') — matches training resample
```

This is the IDENTICAL semantics as `pd.resample('H', closed='left', label='left', offset='30min')` used in training. The live code computes it manually bar-by-bar for streaming compatibility — the math is equivalent.

**Session gap detection:**
```python
def _has_session_gap(h1_buffer: deque[H1Bar]) -> bool:
    # Same logic as src/data/window.py _has_session_gap
    # Max gap between consecutive H1 timestamps > 90 min → True
```
If `True`, the window is poisoned (cross-session contamination). `build()` returns `None` and logs `SKIP_CROSS_SESSION_GAP`.

**Interface:**
```python
@dataclass
class H1Bar:
    timestamp: pd.Timestamp
    open: float; high: float; low: float; close: float; volume: float

class LiveWindowBuilder:
    WINDOW_SIZE = 60

    def __init__(self): ...

    def on_bar(self, bar: MinuteBar) -> Optional[WindowEvent]:
        """Feed a 1-min bar. Returns WindowEvent on H1 close, else None."""

    def gap_fill(self, bars: list[MinuteBar]) -> None:
        """Insert REST-fetched historical 1-min bars in order (warm-up or reconnect)."""

    @property
    def h1_count(self) -> int:
        """Number of complete H1 bars in buffer."""
```

**`WindowEvent` dataclass:**
```python
@dataclass
class WindowEvent:
    h1_timestamp: pd.Timestamp          # boundary start of the completed H1 bar
    h1_bar: H1Bar                       # the just-closed H1 bar
    window: Optional[np.ndarray]        # (60, 5) float32 normalised, or None if buffer < 60 or gap
    raw_window: Optional[np.ndarray]    # (60, 5) float64 raw OHLCV (for XGBoost adapter raw passthrough)
    skip_reason: Optional[str]          # e.g. "WARMUP", "CROSS_SESSION_GAP"
```

**Warm-up:**
While `h1_count < 60`, `window` and `raw_window` are `None`, `skip_reason = "WARMUP"`. The runner checks this and skips inference.

**Cold start backfill (startup warm-up):**
On process start, before subscribing to stream, `LiveWindowBuilder.gap_fill()` is called with historical 1-min bars from `StockHistoricalDataClient` for today's session so far. This pre-fills the H1 buffer and accelerates warm-up past bar 1 of the day. If market is not open yet (pre-session start), backfill yesterday's last 60 H1 bars instead.

**Reconnect gap-fill:**
On `on_reconnect()`, fetch 1-min bars for the missed interval via REST and call `gap_fill()`. Bars already in buffer are not re-inserted (deduplication by timestamp). This handles the case where WSS drops for 5–30 minutes.

**DST / half-day handling:**
On session start, query `exchange_calendars.get_calendar("NYSE").schedule(start=today, end=today)` to get actual open/close times. Override `session_close_time` for the day. If early close (13:00 ET), the H1 boundaries are `09:30`, `10:30`, `11:30`, `12:30` — only 3.5 hours of bars.

---

### `src/live/decision.py` — `SingleModelDecision`

Takes `WindowEvent` → `TradeAction`. One model. No voting.

**Interface:**
```python
@dataclass
class TradeAction:
    signal: Literal["bull", "bear", "none"]
    confidence: float        # max proba of bull/bear class
    proba: np.ndarray        # (3,) raw probas [none, bull, bear]
    h1_timestamp: pd.Timestamp
    gap_low: float           # from last 3 bars of window (for SL)
    gap_high: float          # from last 3 bars of window (for SL)
    skip_reason: Optional[str]

class SingleModelDecision:
    def __init__(self, adapter: ModelAdapter, threshold: float = 0.5): ...

    def decide(self, event: WindowEvent) -> TradeAction:
        """
        1. If event.window is None → return signal="none", skip_reason=event.skip_reason.
        2. Call adapter.predict_proba(event.window[np.newaxis])  → (1, 3) → squeeze to (3,).
        3. bull_p = proba[1], bear_p = proba[2].
        4. If max(bull_p, bear_p) < threshold → signal="none", skip_reason="BELOW_THRESHOLD".
        5. Else → signal = "bull" if bull_p > bear_p else "bear".
        6. Extract gap boundaries from raw_window[-3:] for SL/TP calc.
           gap_low  = min(raw_window[-2, 2], raw_window[-1, 2])  # candles N-1 and N low
           gap_high = max(raw_window[-2, 1], raw_window[-1, 1])  # candles N-1 and N high
           (Conservative: use the outer edges of the 3-candle FVG structure.)
        """
```

**FVG gap boundary extraction:**
The 3-candle FVG pattern: bars[-3]=candle N-2 (impulse candle), bars[-2]=candle N-1 (gap candle), bars[-1]=candle N (label bar, already closed = N+1 per ValidFVGLabeller). Gap is between bar[-3].high and bar[-1].low (bull) or bar[-3].low and bar[-1].high (bear). Extracting from raw (unnormalised) window:
- Bull FVG gap: `gap_low = raw_window[-1, 2]` (bar N low), `gap_high = raw_window[-3, 1]` (bar N-2 high)
- Bear FVG gap: `gap_low = raw_window[-3, 2]` (bar N-2 low), `gap_high = raw_window[-1, 1]` (bar N high)

---

### `src/live/execution.py` — `PaperExecutor`

Submits bracket orders, tracks positions, enforces safety rules.

**Interface:**
```python
class PaperExecutor:
    def __init__(
        self,
        trading_client: TradingClient,
        risk_pct: float = 0.01,       # fraction of equity per trade
        tp_rr: float = 2.0,           # take-profit R-multiple
        max_sl_pct: float = 0.03,     # skip if SL > 3% of price
        max_daily_dd: float = 0.02,   # kill switch at 2% daily drawdown
        max_shares: int = 50,         # absolute cap on qty
    ): ...

    def on_trade_action(self, action: TradeAction, logger: SessionLogger) -> None:
        """Main entry. Validates guards, sizes, submits bracket order."""

    def on_fill_event(self, fill: FillEvent) -> None:
        """Update position tracker. Trigger kill switch check."""

    def on_session_close(self) -> None:
        """Cancel all open orders. Flatten open positions."""

    @property
    def is_halted(self) -> bool:
        """True if kill switch has fired."""
```

**Safety guards (checked in order, first failure skips trade):**
1. `is_halted` → `SKIP_KILL_SWITCH`
2. `action.signal == "none"` → no-op (already logged by decision layer)
3. `_is_blackout_period(action.h1_timestamp)` → `SKIP_BLACKOUT`
   - Blackout: first H1 bar of day (09:30–10:30 open volatility) and last H1 bar (15:30–16:00 close)
   - Implemented as: skip if `h1_timestamp.hour == 9` (09:30 bar) or `h1_timestamp.hour == 15` (15:30 bar)
4. `open_positions > 0` → `SKIP_OPEN_POSITION`
5. SL distance check: `sl_dist_pct > max_sl_pct` → `SKIP_SL_TOO_WIDE`
6. Position sizing: `qty = floor(equity * risk_pct / sl_dist)`. If `qty < 1` → `SKIP_SIZING`

**Position sizing:**
```python
account = trading_client.get_account()
equity = float(account.portfolio_value)
entry_price = current_ask_or_bid  # fetched via REST snapshot
sl_dist = abs(entry_price - sl_price)
qty = min(int(equity * risk_pct / sl_dist), max_shares)
```
Entry price for sizing: fetch latest ask (buy) or bid (sell) via `StockLatestQuoteRequest` at decision time. This is the best available price before order submission. Actual fill will differ by 1 tick (slippage model accounts for this in logger).

**SL/TP calculation:**
```python
if action.signal == "bull":
    sl_price = action.gap_low - 0.01          # 1-tick buffer below gap low
    tp_price = entry_price + tp_rr * (entry_price - sl_price)
elif action.signal == "bear":
    sl_price = action.gap_high + 0.01         # 1-tick buffer above gap high
    tp_price = entry_price - tp_rr * (sl_price - entry_price)
```

**Bracket order submission:**
```python
order = MarketOrderRequest(
    symbol="SPY",
    qty=qty,
    side=OrderSide.BUY if signal == "bull" else OrderSide.SELL,
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    stop_loss=StopLossRequest(stop_price=round(sl_price, 2)),
    take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
)
result = trading_client.submit_order(order)
```

**Kill switch:**
Triggered when `(session_start_equity - current_equity) / session_start_equity >= max_daily_dd`. On trigger: cancel all orders, close position at market, set `is_halted = True`, log `KILL_SWITCH_TRIGGERED`.

**Partial fills:**
Track `filled_qty` per order separately. Do not re-submit for unfilled portion — the bracket order legs cover the actual filled qty. Log partial fills to `fills.sqlite`.

---

### `src/live/slippage.py` — `SlippageModel`

Adjusts P&L accounting for realistic entry/exit cost. Does not affect actual Alpaca orders.

**Interface:**
```python
class SlippageModel:
    def __init__(self, ticks: int = 1, tick_size: float = 0.01, commission_per_share: float = 0.005): ...

    def adjusted_entry(self, fill_price: float, side: str) -> float:
        """Return fill_price + ticks*tick_size (buy) or - ticks*tick_size (sell)."""

    def commission(self, qty: int) -> float:
        """Return qty * commission_per_share."""

    def adjusted_pnl(self, entry_fill: float, exit_fill: float, qty: int, side: str) -> float:
        """Compute P&L with slippage on both legs + round-trip commission."""
```

Slippage model is applied in `SessionLogger` when recording realised P&L — it never modifies the actual order prices sent to Alpaca.

---

### `src/live/logger.py` — `SessionLogger`

Three stores per session in `logs/paper/<session-id>/`.

**Interface:**
```python
class SessionLogger:
    def __init__(self, session_id: str, base_dir: str = "logs/paper"): ...

    def log_bar_1m(self, bar: MinuteBar) -> None: ...
    def log_bar_h1(self, bar: H1Bar, window_valid: bool) -> None: ...
    def log_prediction(self, event: WindowEvent, action: TradeAction) -> None: ...
    def log_order(self, order_id: str, action: TradeAction, qty: int, sl: float, tp: float) -> None: ...
    def log_fill(self, fill: FillEvent, slippage_model: SlippageModel) -> None: ...
    def log_equity(self, timestamp: pd.Timestamp, account_snapshot: dict) -> None: ...
    def log_event(self, event_type: str, payload: dict) -> None: ...
    def close(self) -> None: ...  # flush parquet, close sqlite, close jsonl
```

**File outputs:**

| File | Format | Schema |
|------|--------|--------|
| `bars_1m.parquet` | parquet | `timestamp, open, high, low, close, volume, is_update` |
| `bars_h1.parquet` | parquet | `timestamp, open, high, low, close, volume, window_valid` |
| `predictions.parquet` | parquet | `timestamp, proba_none, proba_bull, proba_bear, pred_class, signal, confidence, skip_reason` |
| `orders.sqlite` | sqlite | `order_id, submitted_at, symbol, side, qty, sl_price, tp_price, status` |
| `fills.sqlite` | sqlite | `order_id, filled_at, fill_price, fill_qty, fill_type, adj_pnl, commission` |
| `equity.jsonl` | jsonl | `{timestamp, cash, portfolio_value, unrealised_pnl, realised_pnl, drawdown_pct}` |
| `events.jsonl` | jsonl | `{timestamp, event_type, payload}` — catch-all debug log |

**Parquet buffering:** accumulate rows in-memory lists, flush to parquet on H1 bar close and on `close()`. Do not write per-minute — too many small writes.

**SQLite:** write immediately on each order/fill event (no buffering — orders are low frequency and correctness matters).

**Session ID format:** `YYYYMMDD_HHMMSS` — e.g. `20260511_093500`. Passed via `--session` CLI arg. Directory: `logs/paper/20260511_093500/`.

---

### `src/live/replay.py` — `SessionReplayer`

Offline replay from saved session bars. No streaming. No order submission.

**Interface:**
```python
class SessionReplayer:
    def __init__(self, session_id: str, adapter: ModelAdapter, threshold: float, base_dir: str = "logs/paper"): ...

    def run(self) -> ReplaySummary:
        """
        1. Load bars_h1.parquet from session.
        2. Load predictions.parquet from session (for comparison).
        3. Feed H1 bars into LiveWindowBuilder one by one.
        4. On each WindowEvent, run adapter.predict_proba → TradeAction.
        5. Compare to logged prediction — flag any divergence.
        6. Print bar-by-bar decision trace to stdout.
        7. Return ReplaySummary with match_rate, divergences list.
        """
```

Divergence = predicted class differs from logged class for the same bar. Expected match rate 100% if same adapter and threshold. Any divergence signals non-determinism bug.

`--replay` mode in `scripts/paper_trade.py` instantiates `SessionReplayer` directly — no streams, no orders.

---

### `scripts/paper_trade.py` — CLI Entrypoint

```
python scripts/paper_trade.py \
  --model <name>:<checkpoint-path> \
  --session <session-id> \
  --symbol SPY \
  --threshold 0.5 \
  --risk-pct 1.0 \
  --max-daily-dd 2.0 \
  --tp-rr 2.0 \
  --slippage-ticks 1

python scripts/paper_trade.py --replay <session-id> --model <name>:<checkpoint-path>
```

**`--model` parsing:**
```
lstm:checkpoints/lstm_seed42.pt  → LSTMAdapter("checkpoints/lstm_seed42.pt")
xgboost:checkpoints/xgb.json    → XGBoostAdapter("checkpoints/xgb.json")
```
Adapter class resolved from `src/inspect/registry.py` by name prefix. Inference always runs on CPU — not MPS (per CLAUDE.md MPS caveats for eval).

**Main async event loop:**
```
asyncio.run(main())
  ├── load adapter (CPU)
  ├── init SessionLogger(session_id)
  ├── init LiveWindowBuilder
  ├── backfill today's 1m bars via REST → gap_fill()
  ├── init PaperExecutor(trading_client)
  ├── init SingleModelDecision(adapter, threshold)
  ├── init AlpacaBarStream(on_bar=..., on_reconnect=...)
  ├── subscribe TradingStream(on_fill=...)
  └── await stream.start()  ← blocks until KeyboardInterrupt or STREAM_FATAL
  └── on exit: executor.on_session_close(); logger.close()
```

**`on_bar` callback (called per 1-min bar):**
```
logger.log_bar_1m(bar)
event = window_builder.on_bar(bar)
if event is None: return
logger.log_bar_h1(event.h1_bar, window_valid=event.window is not None)
action = decision.decide(event)
logger.log_prediction(event, action)
if action.signal != "none":
    executor.on_trade_action(action, logger)
logger.log_equity(event.h1_timestamp, trading_client.get_account())
```

**Environment variables required:**
```
ALPACA_API_KEY
ALPACA_SECRET_KEY
ALPACA_PAPER=true
```

---

## Live Window Equivalence Guarantee

This is the single most critical correctness constraint. The live pipeline must produce IDENTICAL (60, 5) windows to what training saw for the same underlying bars.

**Training pipeline path:**
```
Alpaca REST 1m bars → RTH filter (09:30–15:59) → resample H1 (closed='left', label='left', offset='30min') → normalise_window()
```

**Live pipeline path:**
```
Alpaca WSS 1m bars → RTH filter (same bounds) → manual H1 assembly (same OHLCV aggregation math) → normalise_window() (same function, imported directly)
```

**Equivalence checkpoints:**
1. RTH filter uses `09:30 <= t.time() < 16:00` — same as `process.py` RTH filter.
2. H1 boundaries are multiples of `:30` from `09:30` — same anchor as `offset='30min'` in resample.
3. H1 assembly: open=first, high=max, low=min, close=last, volume=sum — exact pandas resample semantics for OHLCV.
4. `normalise_window()` called from `src.data.normalize` — same import, same function, never duplicated.
5. Session gap detection uses `_MAX_INTRA_WINDOW_GAP_MINUTES = 90` — same constant from `src/data/window.py` (import it, don't copy it).
6. Windows dropped when `window is None` — never passed to adapter with missing bars.

**Verification:** the replay module confirms this empirically. After a live session, running `--replay` with the same adapter must produce 100% matching predictions against the live log.

---

## Entry/SL/TP Rules (Pinned Defaults)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Entry timing | Next H1 open = next bar after model fires | Model fires at H1 close (bar N). Entry is at open of bar N+1. Market order submitted immediately at close — fills within seconds, effectively at next bar open. |
| SL (bull) | `gap_low - 0.01` | Below the FVG gap's lower boundary + 1-tick buffer. Stop placed beyond the invalidation level. |
| SL (bear) | `gap_high + 0.01` | Above FVG gap's upper boundary + 1-tick buffer. |
| TP | `entry + R * tp_rr` (bull) / `entry - R * tp_rr` (bear) | R = abs(entry - sl). Default 2R. Configurable via `--tp-rr`. |
| Max SL distance | 3% of price | Skip trade if gap is too wide (abnormal market conditions). |
| Risk per trade | 1% of equity | `qty = equity * 0.01 / sl_dist`. Hard cap 50 shares. |

---

## Safety Rails Summary

| Guard | Trigger | Response |
|-------|---------|---------|
| Daily drawdown kill switch | `(start_equity - current_equity) / start_equity >= max_daily_dd` | Cancel all orders, flatten position, halt for session |
| Max concurrent positions | >0 open position | Skip new signal, log `SKIP_OPEN_POSITION` |
| Blackout window | First H1 bar (09:30) and last H1 bar (15:30) of session | Skip signal, log `SKIP_BLACKOUT` |
| SL too wide | SL distance > 3% of entry price | Skip signal, log `SKIP_SL_TOO_WIDE` |
| Sizing floor | Computed qty < 1 share | Skip signal, log `SKIP_SIZING` |
| Session close | NYSE close (or early close) - 5 min | Cancel all open orders, flatten if needed |
| DST transition | Detected via `exchange_calendars` | H1 boundaries auto-adjusted (they're relative to session open) |
| Half-day (13:00 close) | `exchange_calendars` schedule check at startup | Override `session_close_time`, reduce H1 boundary set |
| WebSocket fatal | 5 retries exhausted | Emit `STREAM_FATAL`, cancel orders, halt |

---

## Test Strategy

**Unit tests** (pytest, no live network):

| Test | Module | Description |
|------|--------|-------------|
| `test_window_builder_rth_filter` | `window_builder.py` | Pre-market and AH bars are rejected, RTH bars accepted |
| `test_window_builder_h1_assembly` | `window_builder.py` | Feed 60 synthetic 1m bars spanning one H1; assert H1 OHLCV equals manual calculation |
| `test_window_builder_warmup` | `window_builder.py` | First 59 H1 bars return `window=None` with `skip_reason="WARMUP"` |
| `test_window_builder_cross_session_gap` | `window_builder.py` | Insert a gap > 90 min; assert `window=None` with `skip_reason="CROSS_SESSION_GAP"` |
| `test_window_builder_normalise_identity` | `window_builder.py` | Output window from `LiveWindowBuilder` == `normalise_window(raw)` for same raw bars |
| `test_window_builder_gap_fill` | `window_builder.py` | Simulate reconnect: insert historical bars via `gap_fill()`, confirm no duplicates |
| `test_slippage_model` | `slippage.py` | Adjusted entry, commission, and P&L calculations are arithmetically correct |
| `test_decision_threshold` | `decision.py` | Mock adapter returning fixed probas; assert signal/skip at both sides of threshold |
| `test_decision_gap_extraction_bull` | `decision.py` | Synthetic raw window with known gap; assert correct `gap_low`/`gap_high` for bull |
| `test_decision_gap_extraction_bear` | `decision.py` | Same for bear |
| `test_executor_guards` | `execution.py` | Each skip guard fires correctly given mocked state |
| `test_executor_sizing` | `execution.py` | Position sizing math: equity=100k, risk=1%, sl_dist=0.50 → qty=2000 (capped 50) |
| `test_executor_sl_tp_calc` | `execution.py` | SL/TP prices correct for bull and bear signals |
| `test_logger_schema` | `logger.py` | Write a few rows to each store; read back and verify column names + types |

**Integration test** (offline, no live network):

`tests/test_live_integration.py` — `TestOfflineReplay`:
1. Load a fixture `tests/fixtures/session_sample/` (prerecorded bars_h1.parquet from a real paper session).
2. Feed bars through `LiveWindowBuilder` → `SingleModelDecision` using a deterministic mock adapter.
3. Assert decisions match expected fixture (`predictions_expected.parquet`).
4. Verifies end-to-end without touching network.

---

## Phases

### Phase 1 — Stream + Window + Logger (observe only)
**Deliverables:** `stream.py`, `window_builder.py`, `logger.py`, `paper_trade.py` (no trading)
**Scope:** Connect to Alpaca WSS, accumulate 1m bars, emit H1 bars, log all bars. No inference. No orders. Verify window equivalence offline.
**Done when:** Can run for a full NYSE session and produce valid `bars_1m.parquet` + `bars_h1.parquet`. Window content spot-checked against historical H1 from REST.

### Phase 2 — Single-model paper trading
**Deliverables:** `decision.py`, `execution.py`, `paper_trade.py` (full)
**Scope:** Load adapter, run inference on each H1 close, submit bracket orders, track fills.
**Done when:** Can complete a full session with at least one round-trip trade logged in `fills.sqlite`. Kill switch tested manually by adjusting threshold to 0.0.

### Phase 3 — Replay + Slippage + Safety Rails
**Deliverables:** `slippage.py`, `replay.py`, reconnect logic, blackout windows, half-day handling
**Scope:** Replay mode works and shows 100% match rate on Phase 1 session log. Slippage P&L differs from Alpaca raw fill. All safety guards verified.
**Done when:** `--replay` CLI works. Unit test suite passing. Integration test with fixture passing.

### Phase 4 — Polish + Docs
**Deliverables:** README in `src/live/`, env example file, docstrings, CLAUDE.md update
**Scope:** Operational runbook for running the harness. Document the window equivalence guarantee explicitly.
**Done when:** Another person could run the harness from the README alone.

---

## Open Questions (from research — resolved here)

1. **IEX vs SIP:** Accept IEX for thesis scope. Note in results that prices may differ slightly from SIP. Do not subscribe to Algo Trader Plus.
2. **Multi-model consensus:** Single-model per process. Decided. No multi-model code in this harness.
3. **Risk % per trade:** 1% of equity. Default $1,000 on $100k paper account. Accepted.
4. **Warm-up source:** Backfill from REST at session start (option b). Decided. `gap_fill()` runs on startup before stream opens.
5. **Deployment environment:** Local Mac M4 during market hours. Inference always on CPU (not MPS — per CLAUDE.md MPS eval caveats). No cloud deployment planned for this phase.
