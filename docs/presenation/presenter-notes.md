# Presenter Notes (cheat-sheet)

Hold this while presenting. Bullets only. Bold = must-say. Separate from `content.md` (the full script).

## The one message (if you forget everything else)
- **I taught 5 AIs to spot a hand-drawn trading pattern (Fair Value Gaps). The real lesson: the bottleneck was DATA, not a fancier model.**
- It is **detection, not price prediction.**
- 5 min. Stay calm, point at the screen, end with the live demo.

## Three numbers to never get wrong
- **XGBoost 0.738** macro-F1 on the hourly test set = **single best overall.**
- **CNN-LSTM 0.675** hourly = best neural net; and at 5-minute it **overtakes** (0.713 vs XGBoost 0.654).
- **Random guess ≈ 0.33.** So 0.7 is genuinely good for a rare pattern.

---

## Slide-by-slide (what to say)

**1. Title — "Teaching 5 AIs to spot what traders draw by hand"**
- One line: traders mark zones by hand; I automated it. 15 seconds, then move.

**2. The hook**
- Pro traders draw "Smart Money Concept" zones by hand. Subjective, slow, doesn't scale.

**3. Think Big (vision)**
- A co-pilot that reads any chart and marks these zones instantly, consistently, for anyone.

**4. My MVP — one pattern, one stock**
- Narrowed the big vision to **one pattern (Fair Value Gap), one stock (SPY).**
- **FVG = a "skipped stair-step":** price moves so fast it leaves an untraded gap between 3 candles. Bullish gap (up) or bearish gap (down).
- **Not fortune-telling — it detects a pattern, not the future.**

**5. The data, and why it's hard**
- **9 years of hourly SPY** (2016-2025). Trained on 4 tickers pooled (SPY/QQQ/IWM/DIA), tested on SPY.
- **Trap 1 - rarity:** ~**97% "no gap"**, ~3% gaps. Accuracy lies (predict "none" always = 97% accurate, useless) → I use **macro-F1** (rare classes count equally).
- **Trap 2 - no peeking:** strict time-order split (train past → test future, no shuffle). **I caught a real bug in a popular public library that peeked 1 bar into the future.**

**6. How I built it — 5 models, simple → complex**
- **XGBoost (the simple control) → LSTM → CNN-LSTM → Transformer → xLSTM.**
- Each reads a **60-bar window** and classifies: no-gap / bullish / bearish. Compared fairly on the same held-out test.

**7. The surprise (headline)**
- On hourly, **the simplest model (XGBoost) won.** Why? **The pattern is rare → small effective data → simple models win that regime.**
- Add data (pool tickers) and use finer 5-min bars → **the neural net (CNN-LSTM) overtakes (0.713 vs 0.654).**
- **Lesson: feed it more data, don't just build a bigger model.**
- Honesty line: *XGBoost's hourly 0.738 is still the best single score. Compare within a timeframe, not across.*

**8. Results in detail**
- Macro-F1 hourly: **XGB 0.738 · CNN-LSTM 0.675 · LSTM 0.640 · Transformer 0.577 · xLSTM 0.369.**
- Confusion matrix punchline: **it never confuses bullish vs bearish — the only mistake is "is there a gap at all."**

**9. Why trust the numbers**
- **5 random seeds** (not one lucky run) · **bootstrap confidence ranges** · **~1000 automated tests** · **leak checks**. Reproducible, not cherry-picked.

**10. Honest about limits (good / bad / ugly)**
- **Good:** strong, reliable detector.
- **Bad:** thin *trading* edge — only one setup (15-min SPY, simple 2R bracket) is positive after real costs.
- **Ugly:** rare pattern = small effective sample; model confidence is uncalibrated (especially XGBoost, runs overconfident).

**11. The product + LIVE DEMO** (the differentiator — slow down here)
- "More than a notebook." Open the demo. **Walk the 4 scenes** (see Demo section below).

**12. What I learned / next**
- **More data beats a fancier model.** Next: more data, then other SMC patterns (order blocks, liquidity sweeps), then multi-stock.

**13. Thanks** — name + GitHub link.

---

## Driving the live demo (slide 11)
- It replays **real, unseen test data bar-by-bar**; each of the 4 models draws gaps live. **Deterministic** (same every run — not random).
- Colors: **green box = bullish gap, red box = bearish gap.** Badge when it resolves: **✓ = price respected the gap (held), ✗ = it failed (hit stop), ? = no clear outcome in the window.**
- **Click the 4 scene buttons** and narrate the contest:
  - **Good** → both XGBoost and CNN-LSTM hold their gaps (models agree).
  - **XGBoost edge** → XGB holds 3, CNN-LSTM holds 0 (XGB's precision wins).
  - **CNN-LSTM edge** → CNN-LSTM holds 2, XGB holds 0 (the net catches winners XGB misses).
  - **Bad** → both fail (the regime where the pattern breaks).
- At scene end a **results modal** auto-opens: per-model held / failed / no-outcome + **average confidence**. Point at it: *"this is the held-rate and how confident each model was."*
- You can **zoom** (mouse wheel / slider) to show a gap up close.

---

## Model cards (one per model — for any modeling question)

**Ladder logic (say this first if asked "why 5 models"):** I climb **simple → complex on purpose.** If more capacity helped, the fancy models would win. They didn't → the task is **data-bound, not capacity-bound.**

### XGBoost — the simple control that WON
- **What:** gradient-boosted decision trees (many small trees, each fixes the previous one's mistakes).
- **Reads:** hand-engineered summary features of the 60-bar window (not the raw sequence).
- **Score:** **0.738 hourly — best overall in the project.**
- **Strength:** dominates **small + imbalanced data**; fast; robust.
- **Weakness:** relies on engineered features (they don't fit 5-min → drops to 0.654); **overconfident probabilities** (saturated near 0/1).
- **Talking point:** the *simple* control beating the neural nets is the headline — proof it's a data problem.

### LSTM — first neural rung
- **What:** recurrent neural net; reads the 60 bars **one at a time, left to right, carrying a memory.**
- **Captures:** temporal order and context across the window.
- **Score:** 0.640 hourly.
- **Weakness:** no explicit local-shape detector, so CNN-LSTM beats it. Long-Short-Term-Memory = the cell that decides what to remember vs forget.

### CNN-LSTM — best neural net (the lead / "carrier")
- **What:** a **CNN front-end + LSTM back-end.**
- **CNN (convolution):** slides small filters across the window to detect **local candle shapes** — the 3-bar gap geometry — like an edge detector, anywhere in the window.
- **LSTM:** then models **how those detected features evolve over time.**
- **Score:** **0.675 hourly; 0.713 tuned at 5-min (overtakes XGBoost 0.654 there).**
- **Why best DL:** an FVG is a **local shape inside a price sequence** — CNN handles the shape, LSTM the sequence.

### Transformer — capacity is not the answer here
- **What:** attention-based; every bar can "look at" every other bar at once (no step-by-step memory).
- **Score:** 0.577 hourly (untuned; one random seed collapsed → unstable).
- **Weakness:** **very data-hungry**; our small dataset hurt it. Demonstrates fancier ≠ better in this regime.

### xLSTM — academic completeness only
- **What:** a 2024 upgrade of the LSTM (sLSTM cells), built for longer sequences.
- **Score:** 0.369 hourly (untuned, worst).
- **Weakness:** underperforms on short financial sequences. **Omitted from the live demo** (no saved checkpoint).

### One-line comparison (if asked "so which is best?")
- **Best overall: XGBoost (0.738, hourly).** Best neural net: CNN-LSTM (0.675 hourly, 0.713 at 5-min). The neural nets only catch up when given more/finer data → **data is the lever.**

---

## Q&A bank (likely questions)

### Architecture / "how does it work"
- **Why CNN-LSTM (why is it the best neural net)?** It fits the problem: a **CNN finds the local candle SHAPE** (the 3-bar gap geometry) and the **LSTM tracks how the sequence evolves over time**. FVG is a local shape inside a price sequence — CNN-LSTM captures both. Empirically the best DL here (0.675 hourly, 0.713 tuned at 5-min).
- **What does the CNN do?** A convolution slides small filters across the 60-bar window, like an **edge/shape detector** — it learns to recognise local candle patterns (e.g. the gap geometry) regardless of where they sit in the window.
- **What does the LSTM do?** A recurrent network that reads bars **in order and keeps a memory** of context, so it captures the **temporal/sequential** structure (what came before the gap).
- **Why did simple XGBoost beat the neural nets?** **Small-data regime.** The pattern is rare (~3%), so effective sample size is small; **gradient-boosted decision trees are strong with little data** and robust to imbalance. Neural nets are data-hungry — they only pull ahead once we add data (pooling + finer bars). That IS the headline finding.
- **What is XGBoost?** Many small decision trees built in sequence, each fixing the last one's mistakes (gradient boosting), on engineered features of the window. The mandatory baseline/control.
- **What is a Transformer / why did it underperform?** Attention-based, very data-hungry, and was unstable here (one seed collapsed). Small data hurt it (0.577).
- **What is xLSTM / why not in the demo?** A 2024 LSTM variant; underperformed on short financial sequences (0.369) and has no saved checkpoint, so it is not shown.

### Data / method
- **Why macro-F1, not accuracy?** 97% of bars have no gap, so "always say none" scores 97% accuracy but is useless. **Macro-F1 averages the score across all three classes**, so the rare gap classes count. Random ≈ 0.33.
- **What's the no-lookahead rule?** Labels use **only information available at that bar** (gap label assigned at bar N+2, the earliest it's knowable). Test data is strictly in the future of training. **I caught a public library leaking 1 bar of future data.**
- **What are the 5 seeds / bootstrap CI?** I retrain with 5 random initialisations and report **mean ± a confidence range** (bootstrap = resample predictions many times) so the result isn't one lucky run.
- **Why SPY + 3 other tickers?** Pooling SPY/QQQ/IWM/DIA multiplies the rare positives ~4x for training; I still **test on SPY held-out** for an apples-to-apples number.

### Results / honesty
- **Is it profitable?** **Detection is strong; the trading edge is thin.** Only one setup (15-min SPY, a simple "2R bracket") was positive after realistic costs. Profit was a side-check, not the goal.
- **What does "held / failed" mean?** A simple test trade off the gap: target = 2x the risk, stop = the other edge. **Held (✓) = hit target, failed (✗) = hit stop.** It's a "was the gap meaningful" check, not a trading strategy claim.
- **XGBoost looks confident but is wrong sometimes — why?** Tree models produce **overconfident (saturated) probabilities**. Good example of why you don't trust confidence blindly — a real observation in the demo.
- **Did the models overfit?** Temporal split + 5 seeds + ~1000 tests + leak checks guard against it; test set is years the model never saw.

### Scope / next
- **Why only one pattern / one stock?** Deliberate honest MVP. The vision is many patterns, many instruments; FVG on SPY is the first proof.
- **What's next?** More data first (the proven lever), then other SMC patterns (order blocks, liquidity sweeps), then more instruments.

### If asked something I don't know
- "Good question — I didn't test that specifically; my finding is X, and I'd check that by Y." Stay honest; the whole project's credibility is honesty.
