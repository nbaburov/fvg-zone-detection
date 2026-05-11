# Research: Live Paper-Trading Harness
**Date:** 11-May-26  
**Depth:** Standard  
**Topic:** Alpaca paper trading + model-agnostic live inference harness for SMC FVG detection  

---

## 1. Recommendation

**Build custom, thin wrapper over `alpaca-py`.** Do not adopt Lumibot or Backtrader-live. Rationale: this project needs (a) minute-bar accumulation into RTH-anchored H1 windows — a custom constraint no framework handles cleanly — (b) direct reuse of `ModelAdapter` from `src/inspect/` which is already designed as a clean interface, and (c) a minimal, auditable codebase for academic submission. Lumibot adds ~15k LoC of opaque framework on top, hides the resample logic, and its backtest/live symmetry is not useful here (no backtest required — we have the inspector). Architecture: one async process runs `StockDataStream` (minute bars), accumulates into a rolling RTH buffer, fires H1 bar events at each `:30` (i.e., 09:30, 10:30 ... 15:30 ET), normalises the last 60 bars into a window, calls each registered `ModelAdapter.predict_proba`, maps probas to a trade decision, submits bracket orders via `TradingClient`, logs everything to `logs/paper/<session-id>/`.

---

## 2. Alpaca Capabilities and Caveats

### Authentication
Two sets of keys: market data keys and trading keys. Paper trading uses `paper-api.alpaca.markets`. Recommended env vars:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true          # switches TradingClient base URL
```
`TradingClient(api_key, secret_key, paper=True)` auto-targets `https://paper-api.alpaca.markets`.

### Free tier data
- **Feed:** `DataFeed.IEX` only (one exchange, ~10% of trades). SIP (consolidated tape, all exchanges) requires Algo Trader Plus ($99/mo).
- **WebSocket subscriptions:** capped at **30 symbols** on free tier. We only need SPY — not a constraint.
- **API rate limit:** 200 calls/min (free), 10,000/min (paid). Live harness makes ~6 REST calls/hour for order submission — negligible.
- **Historical on free tier:** last 15 minutes only via REST. For warm-up (first 60 bars of day) use `StockHistoricalDataClient` with `adjustment="raw"` — same pattern as training pipeline.

### Minute bar streaming (critical — same H1 issue as training)
`StockDataStream.subscribe_bars(handler, "SPY")` delivers 1-minute bars via WebSocket. Bars are labeled by **start time** and emitted ~2–3 seconds after the minute boundary. A bar labeled `09:30` covers trades `09:30:00–09:30:59`, arrives ~`09:31:03`. An `updated_bar` message may follow at the `:30` mark of the next minute for late trades — **must handle this** (update buffer in-place before H1 close). **Alpaca H1 bars remain clock-aligned (09:00–10:00 mix pre-market)** — same problem as training. Strategy confirmed: stream 1-minute bars, accumulate in RTH buffer (`09:30–15:59 ET`), resample at each full-hour boundary (09:30, 10:30, ... 15:30 — 6 H1 bars per full session).

### Known latency issues (from community forum)
- SPY bars reported arriving 30s late intermittently on SIP feed; IEX is thinner but generally on-time.
- Occasional 3-minute data gaps followed by volume spike — buffer must handle non-monotonic arrival by timestamp.
- Websocket connection can drop with no reconnect; must implement exponential-backoff reconnect logic.

### Paper trading fill realism
- Fill simulation: orders fill when marketable at NBBO. No slippage, no queue position, no market impact.
- Partial fills: random partial fills ~10% of the time — must handle `partial_fill` events from `TradingStream`.
- Pattern Day Trader (PDT) rules are enforced on paper account (< $25k equity).
- No borrow fees, no dividends, no regulatory fees simulated.
- **Conclusion:** Alpaca paper fill is unrealistically optimistic. Must layer own 1-tick slippage model (see §4).

### Supported order types
`TradingClient.submit_order()` supports: market, limit, stop, stop-limit, trailing-stop, bracket (OTO with TP+SL legs), OCO. For FVG strategy bracket order is the right primitive:
```python
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderClass, TimeInForce

order = MarketOrderRequest(
    symbol="SPY",
    qty=shares,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    stop_loss=StopLossRequest(stop_price=sl_price),
    take_profit=TakeProfitRequest(limit_price=tp_price),
)
trading_client.submit_order(order)
```

### Fill notifications
`TradingStream(api_key, secret_key, paper=True)` → `subscribe_trade_updates(handler)` delivers async events: `new`, `fill`, `partial_fill`, `canceled`, `rejected`. Run in same async event loop as `StockDataStream`.

### Rate limits
200 REST calls/min on free tier. Submit ~1 bracket order per H1 bar max → ~6/hour → not a concern.

---

## 3. Architecture Sketch

```
src/live/
  __init__.py
  session.py           # RTH session guard — is_rth_bar(), is_half_day(), session_close_time()
  buffer.py            # MinuteBarBuffer — accumulates 1m bars, emits H1 close events
  window_builder.py    # LiveWindowBuilder — mirrors src/data/window.py for streaming context
  trader.py            # PaperTrader — wraps TradingClient, applies slippage model, enforces risk rules
  logger.py            # SessionLogger — writes parquet/sqlite/jsonl (see §5)
  runner.py            # LiveRunner — wires streams + ModelAdapters + trader + logger

src/inspect/
  base.py              # ModelAdapter ABC — REUSED AS-IS (predict_proba interface)
  adapters/            # REUSED AS-IS
```

**Data flow:**

```
StockDataStream (1m bars)
  → MinuteBarBuffer.on_bar(bar)
      → if updated_bar: overwrite existing minute slot
      → at RTH H1 boundary: emit H1BarEvent(ohlcv_df: last 60 H1 bars + raw minutes)
          → LiveWindowBuilder.build(h1_df) → (60,5) normalised window (same normalise_window() as training)
              → for adapter in adapters:
                  probas = adapter.predict_proba(window)      # ModelAdapter.predict_proba — unchanged interface
                  pred = argmax(probas)
              → DecisionEngine.decide(probas_dict) → TradeAction(side, entry, sl, tp, qty)
                  → PaperTrader.submit(action) → TradingClient.submit_order(bracket)
                      → SessionLogger.log_prediction(...)
                      → SessionLogger.log_order(...)

TradingStream (order updates)
  → SessionLogger.log_fill(...)
  → EquityTracker.update(fill)
      → KillSwitch.check(drawdown)  → flatten all + halt if triggered
```

**ModelAdapter reuse:** `src/inspect/base.py` `ModelAdapter` ABC with `predict_proba(windows: np.ndarray) -> np.ndarray` where `windows` is `(N, 60, 5)`. Live harness always passes `N=1`. Same adapter classes work: `LSTMAdapter`, `XGBoostAdapter`, future `CNNLSTMAdapter`. No changes to adapter code required. Registry in `src/inspect/registry.py` used to load adapters at startup.

**LiveWindowBuilder vs SMCWindowDataset:** `SMCWindowDataset` operates on a complete DataFrame. `LiveWindowBuilder` wraps a rolling `deque(maxlen=60)` of H1 OHLCV rows. At each H1 close: converts deque to `(60,5)` numpy array, calls `normalise_window()` (same function from `src/data/normalize.py`), checks `_has_session_gap()` — if gap detected (holiday/weekend leaked into buffer), skip prediction. No cross-session contamination. This mirrors `_window_generator` logic exactly.

---

## 4. Realism Trade-offs

| Aspect | What Alpaca simulates | Gap / Mitigation |
|---|---|---|
| Fill price | NBBO best ask/bid at moment of marketability | No slippage. **Mitigation:** apply 1-tick slippage manually: entry_price = next_bar_open + 0.01 (buy) or - 0.01 (sell) for P&L accounting. Do NOT submit limit orders at next-open — use market order, then adjust P&L in logger. |
| Entry timing | Bracket fires at next market opportunity after submit | Model fires at H1 close (~15:30:03). Submit market order immediately. By the time it fills it will be a few seconds into the next minute — effectively "next open". Acceptable for H1 strategy. |
| SL/TP placement (FVG) | `stop_price` = below gap low (bull FVG) or above gap high (bear FVG). `take_profit` = entry + N×R (R = entry - SL). Start with N=2 (2R TP). Gap boundaries from last 3 bars of window (bars[-3], [-2], [-1]). SL = gap_low - 0.01 buffer tick. | FVG gap may be wide on SPY H1 — position size can be tiny. Check qty > 0 before submit. |
| Partial fills | Simulated randomly ~10% of time | Handle `partial_fill` events — track unfilled qty, do not double-submit. |
| PDT rules | Enforced | On H1 strategy (<4 round trips/day typically) this is unlikely to trigger but monitor. |
| Fees | Not simulated | Add $0.005/share commission model in `SessionLogger` for realistic P&L reporting. |
| Bid-ask spread | Not modelled separately | IEX feed has wider effective spreads than SIP. Accept as is; note in P&L caveat. |

**Position sizing default:** fixed-risk % per trade. `qty = floor((account_equity * 0.01) / (entry - sl_price))`. 1% risk per trade. Cap at 50 shares max on SPY to avoid PDT/PDT interaction. If `qty == 0`, skip trade and log `SKIP_SIZING`.

**Concurrent positions:** cap at 1 open position at a time. If a signal fires while a bracket is live, log `SKIP_OPEN_POSITION`. FVG signals can stack within the same H1 session on different bars, so this guard is required.

---

## 5. Logging Schema

Location: `logs/paper/<session-id>/` (already gitignored per `.gitignore`). Session ID = `YYYYMMDD_HHMMSS`.

### Files per session

| File | Format | Contents |
|---|---|---|
| `bars_1m.parquet` | parquet | All 1m bars received: timestamp, open, high, low, close, volume, is_updated |
| `bars_h1.parquet` | parquet | Reconstructed H1 bars: timestamp, open, high, low, close, volume, window_valid (bool) |
| `predictions.parquet` | parquet | Per-H1-bar: timestamp, model_name, proba_none, proba_bull, proba_bear, pred_class |
| `orders.sqlite` | sqlite | Per order: order_id, submitted_at, symbol, side, qty, order_class, entry_price_target, sl_price, tp_price, status |
| `fills.sqlite` | sqlite | Per fill event: order_id, filled_at, fill_price, fill_qty, fill_type (fill/partial_fill) |
| `equity.jsonl` | jsonl | Snapshot per H1 bar close: timestamp, cash, portfolio_value, unrealised_pnl, realised_pnl, drawdown_pct |
| `events.jsonl` | jsonl | Raw event log: timestamp, event_type, payload — everything for debugging |

### Replay capability
`scripts/replay_session.py <session-id>` loads `bars_h1.parquet` + `predictions.parquet` + `orders.sqlite` → steps bar-by-bar, prints decision at each step, reconstructs equity curve. No streaming required. Uses same `LiveWindowBuilder` in batch mode. This is purely offline — does not re-run inference, replays logged predictions.

---

## 6. Risk Register

| Risk | Likelihood | Blast radius | Mitigation |
|---|---|---|---|
| WebSocket disconnect mid-session | Medium | Missing bars → stale window → wrong/missed prediction | Exponential backoff reconnect (max 5 retries, then halt). On reconnect, backfill missed minutes via `StockHistoricalDataClient.get_stock_bars()` (free tier allows last 15 min). |
| H1 window contaminated by non-RTH bars | Medium | Lookahead + training distribution mismatch | `session.py` RTH guard: reject any bar with timestamp < 09:30 or >= 16:00 ET. Apply on every `on_bar()` call before buffer insert. |
| Half-day session (early close at 13:00) | Low | Model predicts on 5-bar H1 window, SL/TP dangle past close | Use `exchange_calendars` `NYSE.schedule()` at session start to detect early close. Override `session_close_time` for the day. Flatten all positions 5 min before session close. |
| Open bracket order not filled + market closes | Low | Dangling bracket order overnight | In `session.py` `on_session_close()`: cancel all open orders via `TradingClient.cancel_orders_for_symbol("SPY")`. |
| Model not yet warmed up (< 60 H1 bars since session start) | High (every session start) | `LiveWindowBuilder` returns `None` for first 59 H1 bars | `LiveWindowBuilder.build()` returns `None` until `len(buffer) == 60`. `runner.py` checks for `None` before calling adapters. Standard cross-session gap logic via `_has_session_gap()` handles the case where warm-up span crosses overnight gap. |

---

## 7. Open Questions for User

1. **IEX vs SIP feed:** Free tier IEX gives ~10% of SPY trades. H1 OHLCV should be representative enough for a thesis project, but prices may differ slightly from training data (which used Alpaca raw adjustment via historical API). Acceptable? Or worth subscribing to Algo Trader Plus ($99/mo)?

2. **Multi-model consensus:** When multiple adapters are loaded (e.g., LSTM + XGBoost), do we trade on any signal (union), majority vote, or only unanimous agreement? Needs a decision before `DecisionEngine` is designed. Suggestion: start with single-model mode (one adapter active), add consensus later.

3. **Risk % per trade:** 1% account equity per trade suggested above. Starting paper account equity is $100,000 by default on Alpaca. That's $1,000 risk per trade. Acceptable?

4. **Session warm-up source:** First 59 H1 bars of the day need to come from somewhere before the live stream can produce predictions. Options: (a) do nothing — predictions start at ~15:30 on day 1 (only last bar of first session), (b) backfill using `StockHistoricalDataClient` with today's 1m bars at session start and resample to H1. Option (b) preferred — is that assumption correct?

5. **Deployment environment:** Is this running on your local Mac M4 during market hours, or do you want to eventually run it on a cloud VM (Railway/Fly.io)? Architecture is the same but MPS vs CPU for torch inference matters (MPS unsafe for eval per CLAUDE.md — always CPU for live inference).
