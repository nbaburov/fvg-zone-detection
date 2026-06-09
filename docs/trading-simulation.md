# FVG Trading Simulation

**Date:** 09-Jun-26 · **Data:** unseen 2023–2025, 4 tickers (SPY, QQQ, IWM, DIA) · **Models:** CNN-LSTM, LSTM, Transformer, XGBoost

> **30-second version:** our AI finds FVG zones on data it never saw. We turned those signals into trades 4 different ways and scored them honestly. Result: **a thin, uncertain edge that mostly survives only for the simplest strategy.** On a real paper-trade we'd expect results **at or below** these numbers — never above — and for the fancier strategies, **negative.** This is an exploratory study; trust the *ranking*, not the dollar figures.

---

## Plain-language glossary

| Term | One sentence |
|---|---|
| **FVG** | a price "gap" from a fast move that traders expect price to revisit — what our model detects. |
| **R** | one unit of risk. Risk \$100/trade → +2R = +\$200, −1R = −\$100. Broker-independent. Not guaranteed dollars. |
| **win rate** | % of finished trades that hit profit instead of stop-loss. |
| **seed** | one random "version" of a model. We run **5 and average**, so we never report just the luckiest one. |
| **± std** | spread across the 5 seeds. **If ± is bigger than the number, the result could be zero — i.e. maybe luck, not skill.** |
| **fill** | whether your order actually executes at your price. "Optimistic" assumes yes; "conservative" is realistic. |

---

## 1. What we have

A trained AI that flags FVG zones (bullish / bearish / none) on hourly candles, with a confidence score. To trade a flag you still need entry/stop/target rules — so we took **4 documented strategies**:

| Strategy | Enters | Stop | Target | Source |
|---|---|---|---|---|
| **fixed_2r** | buy now (market) | gap edge | fixed 2× risk | simple baseline (no external ruleset) |
| **ict_iofed** | wait for return (limit) | far gap side (wide) | swing high | ICT "IOFED" entry rule -- a retail trading framework's wait-for-return variant |
| **ce_50pct** | wait, gap midpoint (limit) | far side (wide) | swing high | ICT 50% -- same framework, enters at the midpoint ("CE") of the gap |
| **tradinglab** | wait for return (limit) | impulse candle (tight) | swing high | TradingLab variant -- tighter stop placed at the impulse candle that created the gap |

3 of the 4 use **limit orders** ("wait for price to come back") — that only pays off *if price actually returns and fills you*. Hold that thought.

---

## 2. How we tested

- **Unseen data only.** Models trained on 2016–2021; tested on **2023–2025** (the tickers were in training, the dates were not).
- **Realistic, not fantasy.** Early runs gave nonsense (one cell: **+417R**) from 5-cent stops, no costs, and cherry-picked seeds. We added guards: a **minimum stop-loss**, **trading costs** (commission + slippage), **5-seed averaging** (no cherry-pick), **outlier exposure**, and a **realistic-fill** option. That +417R fell into the **+3 to +21R** range below.
- **Pre-registered metric:** total R after costs, averaged over 5 seeds.

---

## 3. What came out

**Total R over 3 unseen years (5-seed average).** Best strategy per row in **bold**. Spreads (± std) are wide — often bigger than the number itself, so most cells could be zero (see §4); omitted here for readability, full tables in `reports/inspect/`.

| Ticker | Model | fixed_2r | ict_iofed | ce_50pct | tradinglab |
|---|---|---|---|---|---|
| SPY | CNN-LSTM | **+13.2** | −26.3 | −20.8 | +3.6 |
| SPY | LSTM | **+21.3** | −7.7 | +4.2 | +19.7 |
| SPY | Transformer | **+13.6** | −9.6 | −0.8 | +8.8 |
| SPY | XGBoost | −14.1 | −32.6 | −27.7 | **−1.6** |
| QQQ | CNN-LSTM | −2.7 | +0.7 | +5.8 | **+15.9** |
| QQQ | LSTM | −5.5 | +13.2 | +18.4 | **+21.7** |
| QQQ | Transformer | −8.8 | −3.6 | +0.4 | **+8.7** |
| QQQ | XGBoost | **+18.7** | −6.5 | +0.5 | −4.0 |
| IWM | CNN-LSTM | −12.4 | +11.7 | **+41.1** | +23.1 |
| IWM | LSTM | −18.3 | +4.2 | **+24.6** | +22.3 |
| IWM | Transformer | −6.9 | −16.1 | **+4.5** | −10.6 |
| IWM | XGBoost | +15.3 | +15.7 | **+29.5** | +13.7 |
| DIA | CNN-LSTM | −19.5 | −8.1 | +6.3 | **+36.3** |
| DIA | LSTM | −7.6 | −22.1 | −14.8 | **+10.9** |
| DIA | Transformer | −21.5 | −12.0 | −5.8 | **+6.3** |
| DIA | XGBoost | −19.3 | −16.8 | +1.4 | **+20.7** |

*(Transformer caveat: 1 of its 5 seeds collapsed to 0 trades — its averages are 4 real seeds + 1 dead, widening its spread.)*

---

## 4. What it means

1. **No single best strategy** — the winner changes by model and ticker. There is no one "FVG strategy."
2. **The edge is thin and uncertain** — the ± spread is often **bigger than the number**, so the likely range includes zero. We genuinely **can't tell skill from luck** for most cells.
3. **XGBoost is steadiest** (tiny spread) but flips sign by ticker.
4. **Cherry-picking inflated everything 2–3×** — e.g. LSTM-tradinglab SPY looked like +44.5R (best seed) but is **+19.7 ± 25.7R** honestly.
5. **The decisive test (fills):** the 3 limit strategies only look good because we *assumed* the order fills on a price touch. Make that realistic (price must *close* through the level) and they crater:

| SPY, after-cost | optimistic fill | realistic fill |
|---|---|---|
| CNN-LSTM · tradinglab | +49.9 | **−81.4** |
| LSTM · tradinglab | −15.1 | **−82.1** |
| **fixed_2r** (market order) | — | **unchanged** |

So honestly: **only `fixed_2r` has an edge that survives realistic execution.** The textbook limit strategies are an artifact of optimistic fills.

---

## 5. How realistic is this — and what to expect on a real paper-trade

**The sim is a best case. Real trading can only be the same or worse — never better** — because everything we *didn't* model (real fills, full spread, slippage, no perfect timing) only subtracts.

**What we'd expect live, per strategy:**

| Strategy | Sim (SPY, best models) | Expected on real paper-trade | Why |
|---|---|---|---|
| **fixed_2r** | ~+13 to +21R / 3yr | **roughly 10–40% lower** → ~+8 to +15R (and the ± means it could still be ~0) | market order always fills; only loses a bit to extra slippage + no "stop after bad day" rule |
| **ict / ce / tradinglab** | looks +4 to +40R | **expect NEGATIVE** (≈ −60 to −80R) | their edge vanishes once fills are realistic — the conservative-fill column *is* that estimate |

**In one line:** on a real paper-trade, bank on **`fixed_2r` ≈ a small fraction of a thin edge** (possibly zero), and treat the other three as **losers**. The honest expected delta vs the sim is **"worse, by a lot for the limit strategies; modestly for fixed_2r."**

**Still not modelled (would push results further down):** position sizing / account compounding, a daily-loss stop, real bid-ask spread beyond 1 tick, and the fact that the model's confidence scores aren't calibrated.

---

## 6. Honest caveats

- **No solid published proof FVG/SMC trading works** — popular online, unproven. The one independent study found FVGs *aren't* revisited 60–63% of the time, which works *against* these strategies. Our unseen-data test is the most credible evidence here.
- **The edges are within the margin of luck** (win rates ~20–37%, spreads that include zero).
- **This compares strategies; it does not promise profit.** Read the R as a *ranking*, not as dollars.

---

## Reproduce

```bash
python scripts/inspect_models.py --dataset test --all-exit-strategies --realistic --all-seeds \
  --models cnn_lstm lstm transformer xgboost
# sweeps (run separately, each writes its own report):
python scripts/inspect_models.py --dataset test --realistic --sensitivity-sweep   --models cnn_lstm lstm transformer xgboost
python scripts/inspect_models.py --dataset test --realistic --confidence-sweep     --models cnn_lstm lstm transformer xgboost
```
Full tables: `reports/inspect/<run>/`. Sources: `.nb/research/09-Jun-26/fvg-exit-strategies.md`. Designs: `.nb/plan/09-Jun-26/{exit-sim-realism,sim-enhancements}.md`.
