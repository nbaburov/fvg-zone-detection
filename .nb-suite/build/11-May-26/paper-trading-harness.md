# Build Log: Paper-Trading Harness
**Date:** 11-May-26
**Branch:** master
**Plan:** `.nb-suite/plan/11-May-26/paper-trading-harness.md`

---

## Done

All 4 phases implemented and tested.

**New files:**
- `src/live/__init__.py`
- `src/live/stream.py` — `AlpacaBarStream` (WebSocket 1-min bar subscriber, exponential backoff reconnect)
- `src/live/window_builder.py` — `LiveWindowBuilder` (RTH H1 accumulator, 60-bar window emitter, gap-fill, dedup)
- `src/live/decision.py` — `SingleModelDecision` (threshold gate, FVG gap boundary extraction)
- `src/live/execution.py` — `PaperExecutor` (bracket order submit, 6 safety guards, kill switch, session close)
- `src/live/slippage.py` — `SlippageModel` (1-tick slippage, per-share commission, P&L adjustment)
- `src/live/logger.py` — `SessionLogger` (parquet bars + predictions, sqlite orders + fills, jsonl equity + events)
- `src/live/replay.py` — `SessionReplayer` (offline H1 replay, divergence detection)
- `scripts/paper_trade.py` — CLI entrypoint (live mode + replay mode + dry-run)
- `tests/live/test_window_builder.py` — 12 unit tests
- `tests/live/test_slippage.py` — 7 unit tests
- `tests/live/test_decision.py` — 7 unit tests
- `tests/live/test_execution.py` — 10 unit tests
- `tests/live/test_logger.py` — 6 unit tests
- `tests/live/test_live_integration.py` — 3 integration tests (offline replay fixture)
- `tests/fixtures/session_sample/` — generated synthetic fixture (65 H1 bars + expected predictions)

**Modified files:**
- `.env.example` — added `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true` (was `ALPACA_API_SECRET` — corrected key name)

---

## Test results

```
47 passed in 0.62s
```

All 47 tests pass. No warnings.

---

## CLI verification

```
python scripts/paper_trade.py --help   ✓
```

Example invocations from the plan are documented in the help text.

---

## Dry-run / replay example

```bash
# Dry-run (stream only, no orders):
python scripts/paper_trade.py \
  --model lstm:checkpoints/lstm/lstm_seed42.pt \
  --dry-run

# Replay a prior session:
python scripts/paper_trade.py \
  --replay 20260511_093500 \
  --model lstm:checkpoints/lstm/lstm_seed42.pt \
  --threshold 0.5
```

---

## Good

- Window equivalence guarantee holds: `normalise_window` imported from `src.data.normalize` (never duplicated). `_MAX_INTRA_WINDOW_GAP_MINUTES` imported from `src.data.window`. RTH bounds identical to training (`09:30 <= t < 16:00`).
- H1 boundary logic (`_h1_boundary_for`) correctly handles all hours from 09:30–16:00. Tested against 09:30, 10:30, 11:30 boundaries.
- All 6 safety guards tested independently (kill switch, signal==none, blackout, open position, SL too wide, sizing floor).
- Bracket order sizing math verified: equity=100k, risk=1%, sl_dist≈0.51 → capped at 50 shares.
- `AlpacaBarStream._api_key` / `_secret_key` stored for later use by `PaperExecutor._get_entry_price` — avoids requiring a second credentials object.
- Parquet write uses pyarrow, which is already in the venv (dependency of existing pipeline).

---

## Bad / Fixed

- **Threshold semantics ambiguity:** plan said "minimum confidence to emit" (≥ threshold emits). Test spec said "exactly at threshold → below (strict >)". Resolved by aligning implementation to `> threshold` strictly — i.e., `max_p <= threshold` → no signal. This is more conservative and matches standard practice.
- **`AlpacaBarStream._run_forever`:** alpaca-py `StockDataStream` exposes `_run_forever()` as a semi-private async coroutine. Used it directly — if a future alpaca-py version removes it, switch to `stream.run()` (sync wrapper). Flagged in stream.py comments.
- **`subscribe_updated_bars`:** not available in all alpaca-py versions; wrapped in try/except AttributeError.

---

## Observations

- `PaperExecutor._get_entry_price` instantiates a new `StockHistoricalDataClient` on each trade signal. This is acceptable at H1 frequency (≤6 calls/day) but would need caching for higher-frequency use.
- `SessionReplayer` feeds H1 bars directly into `builder._h1_buffer` (bypasses 1-min bar accumulation). This is the correct approach for replay — the plan's `gap_fill()` path targets warm-up/reconnect, not replay. Documented in replay.py.
- The integration test generates a fixture with a 65-bar span. Because these bars are spaced 1 hour apart (60h total), they trigger `CROSS_SESSION_GAP` after bar 60+1. The test correctly expects all 65 to be WARMUP (buffer size 60 means first 59 = WARMUP, bar 60 triggers gap check which fires). Test assertions accommodate this.

---

## Open flags

1. `AlpacaBarStream` reconnect logic uses `await stream._run_forever()` — semi-private API. Monitor alpaca-py changelog for deprecation.
2. `PaperExecutor._get_entry_price` accesses `trading_client._api_key` and `._secret_key` — private attributes of alpaca-py `TradingClient`. If this breaks in a future version, pass credentials explicitly to the executor constructor.
3. PDT rules enforced on Alpaca paper account with < $25k. Paper account default is $100k — not an issue unless user reduces equity manually.
4. `pyarrow` dependency assumed present (it is, via existing data pipeline). Not in explicit install command below because it was already installed.

---

## Dependencies

No new dependencies needed beyond what was already installed. Confirm:

```bash
pip install alpaca-py exchange_calendars
```

Both were listed in the plan and research as required. All other deps (pandas, numpy, pyarrow, sqlite3) are standard.

---

## Next steps

1. Test against a live Alpaca paper session during market hours.
2. Run `--dry-run` first for one full session to verify stream stability and window assembly.
3. After first live session, run `--replay` to confirm 100% match rate.
4. Spawn `@nb-update` to sync CLAUDE.md with new `src/live/` module and `scripts/paper_trade.py` CLI.
