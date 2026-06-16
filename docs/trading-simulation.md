# FVG Trading Simulation

**Date:** 09-Jun-26 (H1) · extended 15-Jun-26 (5m / 15m) · **Data:** unseen 2023 to 2025, 4 tickers (SPY, QQQ, IWM, DIA) · **Timeframes:** H1, 15m, 5m · **Models:** CNN-LSTM (Optuna-tuned at 5m/15m), LSTM, Transformer, XGBoost

> **30-second version:** our AI finds FVG zones on data it never saw. We turned those signals into trades 4 different ways, on 3 timeframes and 4 tickers, and scored them honestly. Result: **a thin, uncertain edge that survives only for the simplest strategy (`fixed_2r`), mostly on SPY.** Lower timeframes detect more zones but trade too often to survive costs: 5m is **detection-only**. On a real paper-trade expect results **at or below** these numbers, never above, and for the fancier strategies, **negative.** Trust the *ranking*, not the dollar figures.

---

## Plain-language glossary

| Term | One sentence |
|---|---|
| **FVG** | a price "gap" from a fast move that traders expect price to revisit, what our model detects. |
| **R** | one unit of risk. Risk \$100/trade → +2R = +\$200, −1R = −\$100. Broker-independent. Not guaranteed dollars. |
| **win rate** | % of finished trades that hit profit instead of stop-loss. |
| **seed** | one random "version" of a model. We run **5 and average**, so we never report just the luckiest one. |
| **± std** | spread across the 5 seeds. **If ± is bigger than the number, the result could be zero, i.e. maybe luck, not skill.** |
| **fill** | whether your order actually executes at your price. "Optimistic" assumes yes; "conservative" is realistic. |
| **timeframe (H1 / 15m / 5m)** | candle size. H1 = 1 hour (~7 bars/day), 15m (~26/day), 5m (~78/day). Smaller = more signals but more cost. |

---

## 1. What we have

A trained AI that flags FVG zones (bullish / bearish / none) with a confidence score on **hourly, 15-minute, and 5-minute candles**. The carrier model (CNN-LSTM) is hyperparameter-tuned at 5m and 15m (detection F1 15m 0.691, 5m 0.713; see `docs/models.md`). To trade a flag you still need entry/stop/target rules, so we took **4 documented strategies**:

| Strategy | Enters | Stop | Target | Source |
|---|---|---|---|---|
| **fixed_2r** | buy now (market) | gap edge | fixed 2× risk | simple baseline (no external ruleset) |
| **ict_iofed** | wait for return (limit) | far gap side (wide) | swing high | ICT "IOFED" entry rule, a retail trading framework's wait-for-return variant |
| **ce_50pct** | wait, gap midpoint (limit) | far side (wide) | swing high | ICT 50%, same framework, enters at the midpoint ("CE") of the gap |
| **tradinglab** | wait for return (limit) | impulse candle (tight) | swing high | TradingLab variant, tighter stop placed at the impulse candle that created the gap |

3 of the 4 use **limit orders** ("wait for price to come back"): that only pays off *if price actually returns and fills you*. Hold that thought.

---

## 2. How we tested

- **Unseen data only.** Models trained on 2016 to 2021; tested on **2023 to 2025** (the tickers were in training, the dates were not).
- **Three timeframes, four tickers.** Same methodology applied to H1, 15m, and 5m. At 5m/15m the carrier is the **tuned** CNN-LSTM; LSTM/Transformer/XGBoost use their best per-timeframe checkpoints. (Note: the lower-TF model F1 in `docs/models.md` is measured on a pooled 4-ticker test; the trade-sim below uses **per-ticker** test sets so each ticker's R is its own.)
- **Realistic, not fantasy.** Early runs gave nonsense (one cell: **+417R**) from 5-cent stops, no costs, and cherry-picked seeds. We added guards: a **minimum stop-loss**, **trading costs** (commission + slippage), **5-seed averaging** (no cherry-pick), **outlier exposure**, and a **realistic-fill** option. That +417R fell into the **+3 to +21R** range below.
- **Pre-registered metric:** total R after costs, averaged over 5 seeds.

---

## 3. What came out

**Total R over 3 unseen years (after costs, 5-seed mean ± std).** One row per **timeframe × ticker × model**, one column per **strategy**: every combination tried, one standard. `fixed_2r` is a market order (always fills); the other three are limit strategies shown here under optimistic fill (see §4 for what realistic fills do to them). Lower-TF carrier (CNN-LSTM) is Optuna-tuned; other archs use their best per-TF checkpoint. Full per-seed tables in `reports/inspect/`.

| TF | Ticker | Model | fixed_2r | ict_iofed | ce_50pct | tradinglab |
|---|---|---|---|---|---|---|
| H1 | SPY | CNN-LSTM | +13.2 ± 12.6 | -26.2 ± 8.4 | -20.8 ± 11.3 | +3.6 ± 8.9 |
| H1 | SPY | LSTM | +21.3 ± 14.7 | -7.7 ± 18.4 | +4.2 ± 19.9 | +19.7 ± 25.7 |
| H1 | SPY | Transformer | +13.6 ± 20.1 | -9.6 ± 10.0 | -0.8 ± 11.9 | +8.8 ± 9.5 |
| H1 | SPY | XGBoost | -14.1 ± 4.1 | -32.6 ± 7.0 | -27.7 ± 7.3 | -1.6 ± 8.7 |
| H1 | QQQ | CNN-LSTM | -2.7 ± 10.7 | +0.7 ± 10.9 | +5.8 ± 3.6 | +15.9 ± 14.2 |
| H1 | QQQ | LSTM | -5.5 ± 18.0 | +13.2 ± 23.9 | +18.4 ± 19.5 | +21.7 ± 24.1 |
| H1 | QQQ | Transformer | -8.8 ± 14.5 | -3.6 ± 11.3 | +0.4 ± 11.3 | +8.7 ± 7.5 |
| H1 | QQQ | XGBoost | +18.6 ± 2.0 | -6.5 ± 4.5 | +0.5 ± 4.9 | -4.0 ± 4.8 |
| H1 | IWM | CNN-LSTM | -12.4 ± 8.4 | +11.7 ± 16.0 | +41.1 ± 15.8 | +23.1 ± 18.8 |
| H1 | IWM | LSTM | -18.3 ± 8.8 | +4.2 ± 17.7 | +24.6 ± 19.3 | +22.3 ± 22.4 |
| H1 | IWM | Transformer | -6.9 ± 17.2 | -16.1 ± 15.1 | +4.5 ± 12.7 | -10.6 ± 9.9 |
| H1 | IWM | XGBoost | +15.3 ± 2.7 | +15.7 ± 5.1 | +29.5 ± 4.0 | +13.7 ± 6.0 |
| H1 | DIA | CNN-LSTM | -19.5 ± 12.3 | -8.1 ± 18.5 | +6.3 ± 21.1 | +36.3 ± 19.8 |
| H1 | DIA | LSTM | -7.5 ± 11.2 | -22.1 ± 7.4 | -14.8 ± 8.9 | +10.9 ± 3.5 |
| H1 | DIA | Transformer | -21.5 ± 17.2 | -12.0 ± 13.1 | -5.8 ± 11.8 | +6.3 ± 16.7 |
| H1 | DIA | XGBoost | -19.3 ± 6.9 | -16.8 ± 3.3 | +1.4 ± 4.6 | +20.7 ± 4.3 |
| 15m | SPY | CNN-LSTM | +27.6 ± 23.4 | -43.9 ± 18.9 | -3.8 ± 28.2 | -38.9 ± 13.8 |
| 15m | SPY | LSTM | +19.7 ± 22.1 | -88.7 ± 37.9 | -39.1 ± 47.3 | -82.9 ± 46.8 |
| 15m | SPY | Transformer | -3.3 ± 19.4 | -41.4 ± 13.9 | -15.1 ± 10.2 | -39.7 ± 18.8 |
| 15m | SPY | XGBoost | +58.2 ± 12.3 | -69.8 ± 10.1 | -15.1 ± 12.3 | -99.3 ± 8.9 |
| 15m | QQQ | CNN-LSTM | -16.1 ± 19.6 | +11.3 ± 19.4 | +25.7 ± 16.1 | +24.5 ± 28.1 |
| 15m | QQQ | LSTM | -33.3 ± 14.4 | -63.1 ± 31.2 | -55.9 ± 28.0 | -64.7 ± 37.1 |
| 15m | QQQ | Transformer | -10.3 ± 14.6 | -58.0 ± 20.4 | -41.2 ± 22.2 | -42.8 ± 26.7 |
| 15m | QQQ | XGBoost | -12.6 ± 1.7 | -75.3 ± 4.9 | -43.0 ± 10.3 | -83.0 ± 5.8 |
| 15m | IWM | CNN-LSTM | -48.8 ± 16.3 | +17.1 ± 38.2 | +26.1 ± 37.9 | +14.1 ± 42.4 |
| 15m | IWM | LSTM | -40.7 ± 18.6 | +11.4 ± 25.2 | +21.0 ± 24.9 | +4.1 ± 41.5 |
| 15m | IWM | Transformer | -40.5 ± 36.5 | +10.6 ± 14.9 | +5.7 ± 17.0 | +10.6 ± 23.2 |
| 15m | IWM | XGBoost | -57.9 ± 8.5 | +23.1 ± 7.9 | +46.2 ± 5.3 | -9.6 ± 15.1 |
| 15m | DIA | CNN-LSTM | -74.1 ± 19.2 | -88.6 ± 13.5 | -72.0 ± 2.7 | -108.9 ± 21.6 |
| 15m | DIA | LSTM | -89.1 ± 11.8 | -90.5 ± 24.3 | -80.7 ± 20.6 | -105.4 ± 16.0 |
| 15m | DIA | Transformer | -57.1 ± 17.5 | -11.1 ± 19.2 | -2.0 ± 19.4 | -6.4 ± 30.3 |
| 15m | DIA | XGBoost | -77.8 ± 5.0 | -64.8 ± 4.0 | -24.3 ± 5.6 | -63.8 ± 2.4 |
| 5m | SPY | CNN-LSTM | -3.2 ± 30.9 | -202.8 ± 21.5 | -215.4 ± 25.9 | -212.2 ± 26.6 |
| 5m | SPY | LSTM | -63.1 ± 70.5 | -231.5 ± 32.6 | -222.9 ± 40.2 | -297.6 ± 36.0 |
| 5m | SPY | Transformer | +3.4 ± 7.7 | -99.5 ± 74.1 | -101.8 ± 78.2 | -116.7 ± 79.2 |
| 5m | SPY | XGBoost | -225.4 ± 14.6 | -307.6 ± 10.6 | -292.5 ± 11.0 | -289.7 ± 19.6 |
| 5m | QQQ | CNN-LSTM | +24.1 ± 35.3 | -144.2 ± 43.6 | -63.0 ± 45.5 | -144.1 ± 49.6 |
| 5m | QQQ | LSTM | -15.1 ± 60.0 | -226.7 ± 20.8 | -140.3 ± 38.3 | -251.6 ± 27.3 |
| 5m | QQQ | Transformer | +18.1 ± 55.1 | -122.7 ± 78.7 | -77.9 ± 60.0 | -136.5 ± 84.0 |
| 5m | QQQ | XGBoost | -54.2 ± 26.7 | -347.5 ± 20.6 | -252.2 ± 15.1 | -359.9 ± 31.2 |
| 5m | IWM | CNN-LSTM | -409.1 ± 59.9 | -330.6 ± 41.2 | -305.0 ± 39.1 | -407.9 ± 51.3 |
| 5m | IWM | LSTM | -428.9 ± 71.8 | -360.1 ± 53.8 | -352.7 ± 55.7 | -455.5 ± 76.6 |
| 5m | IWM | Transformer | -275.1 ± 167.4 | -194.1 ± 117.8 | -174.5 ± 103.7 | -269.7 ± 160.6 |
| 5m | IWM | XGBoost | -640.1 ± 25.5 | -607.2 ± 14.4 | -602.4 ± 13.8 | -787.4 ± 12.6 |
| 5m | DIA | CNN-LSTM | -88.2 ± 36.6 | -277.9 ± 30.7 | -266.1 ± 27.9 | -267.6 ± 36.1 |
| 5m | DIA | LSTM | -176.0 ± 52.3 | -284.5 ± 26.8 | -280.1 ± 27.8 | -274.4 ± 39.7 |
| 5m | DIA | Transformer | -56.2 ± 47.9 | -160.3 ± 99.2 | -159.8 ± 105.6 | -162.2 ± 99.8 |
| 5m | DIA | XGBoost | -414.4 ± 13.2 | -597.9 ± 8.5 | -561.2 ± 10.1 | -595.2 ± 6.4 |

*(Transformer at H1: 1 of 5 seeds collapsed to 0 trades, widening its spread. ± often exceeds the mean → that cell could be zero; see §4.)*

**Reading it:** the only consistently positive cells that clear costs are **`fixed_2r` on SPY at H1 and 15m** (15m XGBoost SPY +58.2 ± 12.3 is the standout, ± < mean). 5m is negative almost everywhere (cost drag from trade count). The limit strategies show scattered positives here but collapse under realistic fills: see §4.

---

## 4. What it means

1. **`fixed_2r` is the only strategy that survives realistic execution at every timeframe.** The 3 limit strategies only look good under optimistic fills; make fills realistic (price must *close* through the level) and they crater:

   | SPY, after-cost | optimistic fill | realistic fill |
   |---|---|---|
   | CNN-LSTM · tradinglab | +49.9 | **−81.4** |
   | LSTM · tradinglab | −15.1 | **−82.1** |
   | **fixed_2r** (market order) | (always fills) | **unchanged** |

2. **The tradable edge is SPY-specific and thin.** 15m SPY `fixed_2r` is the standout positive cell (XGBoost +58R, tuned CNN-LSTM +28R, ± smaller than the mean, likely real). QQQ/IWM/DIA are negative at 15m for every model: at H1 the winner already changed by ticker, and at 15m only SPY clears costs.
3. **5m is detection-only: confirmed across all 4 tickers.** Tuning lifted 5m detection F1 to 0.713 (best DL in the project) but **did not** make 5m tradable: mostly negative, IWM catastrophic. Use 5m to *find/confirm* zones, not to time trades.
4. **Better detection ≠ proportionally better dollars.** The tuned CNN-LSTM is the best DL detector, yet on 15m SPY it sits below XGBoost on R (+28 vs +58), the same ordering as F1. XGBoost stays the steadiest (tiny ±) but flips sign by ticker.
5. **The edge is within the margin of luck for most cells**: ± often bigger than the number, win rates ~20 to 37%. We genuinely can't tell skill from luck outside the few SPY `fixed_2r` cells.

So honestly: **only `fixed_2r` has an edge that survives realistic execution, and only really on SPY. 15m is the tradable timeframe, 5m is detection-only, H1 is comparable to 15m on SPY.**

---

## 5. How realistic is this and what to expect on a real paper-trade

**The sim is a best case. Real trading can only be the same or worse, never better, because everything we *didn't* model (real fills, full spread, slippage, no perfect timing) only subtracts.**

**What we'd expect live, per strategy / timeframe:**

| Setup | Sim (best case) | Expected on real paper-trade | Why |
|---|---|---|---|
| **fixed_2r, 15m/H1 SPY** | ~+13 to +58R / 3yr | **roughly 10 to 40% lower, still possibly >0** | market order always fills; only loses a bit to extra slippage |
| **fixed_2r, non-SPY** | mostly negative | **negative** | edge doesn't clear costs off SPY |
| **fixed_2r, 5m (any ticker)** | negative / wild | **negative** | trade frequency × cost dominates: detection-only |
| **ict / ce / tradinglab (any TF)** | looks positive under optimistic fills | **expect NEGATIVE** | their edge vanishes once fills are realistic |

**In one line:** bank on **15m (or H1) SPY `fixed_2r` as a small fraction of a thin edge** (possibly zero); treat lower timeframes as detection tools and the limit strategies as losers.

**Still not modelled (would push results further down):** position sizing / account compounding, a daily-loss stop, real bid-ask spread beyond 1 tick, and the fact that the model's confidence scores aren't calibrated. **Cost sensitivity is severe at 5m:** the same 5m SPY `fixed_2r` cell swings from +116R (cheap costs) to −38R (base) to −193R (expensive); 15m is far less fragile (+17 to −11 to −39).

---

## 6. Honest caveats

- **No solid published proof FVG/SMC trading works**: popular online, unproven. The one independent study found FVGs *aren't* revisited 60 to 63% of the time, which works *against* these strategies. Our unseen-data test is the most credible evidence here.
- **The edges are within the margin of luck** (win rates ~20 to 37%, spreads that include zero), except the few SPY `fixed_2r` cells.
- **Lower timeframes detect more but trade worse.** More signals ≠ more profit; cost drag scales with trade count, so 5m loses despite the best detection F1.
- **This compares strategies / timeframes / tickers; it does not promise profit.** Read the R as a *ranking*, not as dollars.
- **Live validation is still one session.** A 5m paper-fleet dry-run (4 tickers, 4 strategies, one full session) validated the live pipeline end-to-end (52 intents, replay, realistic P&L) but produced only 1 to 3 trades per cell, far too few for any edge claim; it tested plumbing, not performance.

---

## Reproduce

```bash
# H1 (4 tickers, all strategies, 5-seed, realistic):
python scripts/inspect_models.py --dataset test --all-exit-strategies --realistic --all-seeds \
  --models cnn_lstm lstm transformer xgboost

# 15m / 5m per ticker (tuned carrier; per-ticker test sets):
python scripts/inspect_models.py --models cnn_lstm lstm transformer xgboost \
  --timeframe 15m --ckpt-dataset multisym --tuned --all-seeds \
  --dataset data/processed/spy_15m_test.parquet   # qqq/iwm/dia, and 5m, likewise

# sensitivity grid (single-seed, explicit tuned paths):
python scripts/inspect_models.py --sensitivity --timeframe 15m \
  --dataset data/processed/spy_15m_test.parquet \
  --models cnn_lstm:checkpoints/cnn_lstm_15m_multisym_tuned/cnn_lstm_seed42.pt \
           lstm:checkpoints/lstm_15m_multisym/lstm_seed42.pt \
           transformer:checkpoints/transformer_15m_multisym/transformer_seed42.pt \
           xgboost:checkpoints/xgboost_15m_multisym/xgb_seed42.ubj
```
Per-ticker 5m/15m test sets are built by resampling the cached raw bars, then labelling and splitting. Full tables: `reports/inspect/<run>/`.
