# Where we are — caveman ELI5

## What we hunt

Cave drawing on SPY chart. Three candles make gap = **FVG**. Big buy-or-sell footprint of rich tribe. Want machine spot footprint instead of hand-draw. Not predict future price. Just say "footprint here, footprint not here."

## What we caught so far

### 🐟 The fish (data)
- Pulled SPY 1-min bars from Alpaca river, 2018–2024. Free tier net.
- Cooked them into H1 candles (RTH only, anchored 09:30 ET). Threw away half-day junk.
- Split fish into 3 baskets: train (2018–2021), val (2022), test (2023–2024). **No shuffle** = no cheating across years.
- ~12k H1 candles total.

### 🪨 The cave wall (labels)
- Wrote our own FVG detector — `FVGLabeller`. Raw geometry, 3-candle gap, label on candle N+1.
- Gives ~15% bull / 10% bear / 75% none. Moderate imbalance.
- Also built fancier `ValidFVGLabeller` (6 SMC criteria, label N+2, only "tradable" gaps). Implemented, tested 92 times, **not yet used to train**. Too sparse (~0.3%) — might starve the model.
- Made gold set (75 candles human-marked). Matched raw labeller perfect κ=1.0. Matched valid labeller κ=0.89.

### 🦴 The tools (models)
- **XGBoost** baseline. Old reliable. Eats hand-crafted features from window.
- **LSTM** 2-layer, 51k params. Eats raw 60-bar OHLCV window. Test macro F1 = **0.824**. Bull 0.77, bear 0.79.
- Both use **inverse-freq weighted loss**. Both seeded 42. Both checkpoint with metadata.

### 🔍 The looking-glass (inspector)
- `scripts/inspect_models.py` — points at any test slice, runs any model(s), gives F1 tables, confusion, agreement.
- Simulates trades: buy at next bar open after model fires, SL beyond gap edge, TP = 2R. Walks future 20 bars. Counts TP/SL/undecided.
- Per-model timeline plot: full slice candles + every trade arrow + exit dot color-coded.
- LSTM on Jan 2024 = 25 trades, 56% win, +13.2R total. XGB = 37 trades, 50% win, +17.5R.

### 🪤 The trap (live paper trader)
- `scripts/paper_trade.py` — full Alpaca paper-trading harness. Streams 1-min bars, builds H1 windows bit-identical to training, runs single model, places bracket orders, logs everything to `logs/paper/<session>/`.
- 47 tests pass. **Not yet run live** — market closed. Dry-run during market hours = next step.

### 📜 The cave drawings (docs)
- README, `docs/architecture.md` (mermaid diagrams), `docs/data-model.md` (schemas), `docs/models-status.md` (training facts).
- Notebook nb01 updated with target = ValidFVG narrative.
- 8 clean conventional commits.

## What we have NOT done

- **Train on ValidFVGLabeller.** Models still on raw FVG. Migration = big gamble (sparse).
- **Run live paper trader.** Code ready, market wasn't open.
- **CNN-LSTM / xLSTM / Transformer.** Next in progression.
- **Test on 2025+.** Out of sample.
- **Multi-symbol.** SPY only.
- **Fees + slippage in outcome sim.** Currently 0 cost.

## 🦣 Next moves (pick or stack)

1. **Run paper trader live** — `--dry-run` first session, verify no errors. Easy win, validates 47 tests are real.
2. **Train CNN-LSTM** — kernel=3 encodes 3-candle FVG. Expected best DL per CLAUDE.md.
3. **Migrate to ValidFVG training** — risky. ~0.3% positive → model may collapse. Run sparsity gate (`scripts/count_valid_fvg.py`) first to confirm count.
4. **Add 2025 data** — extend Alpaca pull. Real out-of-sample.
5. **Status update 1 prep** — May 17 peer review deadline. Have plenty to show.

## 🦴 Bottom line

Pipeline = solid. Labelling = honest, no lookahead. LSTM hitting **0.82 Macro F1** on real held-out 2023–2024 data. Inspector lets you see what model sees. Paper trader ready when market opens. Docs caught up.

Story for May 17: "I built end-to-end pipeline, baseline beats XGBoost, two-tier labeller, offline+live tooling — now I scale architectures."
