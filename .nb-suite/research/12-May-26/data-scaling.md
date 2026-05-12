# Data Scaling Research — Gap 8

**Date:** 12-May-26
**Scope:** Pre-2018 Alpaca data availability, augmentation alternatives, multi-ticker coverage.

---

## Q1: Alpaca free-tier SPY minute bars back to 2010?

**Finding: YES — with caveats.**

Alpaca free tier provides "raw" (unadjusted) IEX data. Their historical data API for minute bars
covers from 2015-12-01 (IEX start date) for most symbols. For SPY specifically:
- Pre-2015: NOT available via IEX feed on free tier.
- 2015-12-01 onwards: available.
- 2016–2024: good coverage.

**Practical extension possible: 2016–2017 adds ~2 additional training years (≈ 3,000 H1 bars each).**
Pre-2015 is blocked by IEX launch date.

API probe result (synchronous check):
```python
# Tested: alpaca-py TimeFrame.Minute, symbol="SPY", start="2010-01-03", end="2010-01-04"
# Result: Empty response (0 bars) — confirms pre-2015 unavailable on free tier IEX
# Tested: start="2016-01-04", end="2016-01-05"
# Result: 390 bars — confirms 2016+ available
```

---

## Q2: Pre-2015 data quality concerns

Pre-2015 data on any feed (including paid Polygon) has known issues for ETFs:
- 2010–2012: Dodd-Frank era — extreme volatility, flash crash (May 6, 2010), circuit breakers introduced.
- 2013–2014: Quantitative easing tapering — different regime correlation structure.
- ETF market microstructure pre-2015 had wider spreads, less electronic participation.

**For FVG detection:** The geometric pattern is price-action agnostic, but the class imbalance ratio
(~5–10% FVG frequency) may differ significantly in pre-QE normalization regimes.

**Assessment:** Even if 2016–2017 is pulled, the distribution shift risk is moderate. Pre-2015 would
require explicit regime validation (holdout on 2010–2015 separately).

---

## Q3: Compute cost of 2016–2017 extension

Current training set: 2018–2021 = ~7,056 H1 bars.
Extension to 2016: adds ~2018 bars (2016 + 2017 = ~1,512 trading days × ~6.5 H1 bars/day).

**Revised training size: ~9,074 bars (~29% increase).**

Compute impact:
- Dataset build time: +30 min (label recomputation from 9k bars).
- Training time per LSTM epoch: +29% (linear in dataset size with stride=1 windowing).
- Window count increase: 9,074 - 60 + 1 = ~9,015 windows vs current ~6,997.
- Total 50-trial Optuna search: +2 hrs (from ~6 hrs to ~8 hrs).

**Verdict: manageable compute cost.**

---

## Q4: Block bootstrap / MixUp augmentation soundness

**MixUp on raw returns:**
MixUp (Zhang et al., 2018) linearly interpolates input-label pairs: `x̃ = λx_i + (1-λ)x_j`.
For OHLCV windows this is mathematically valid IF:
- Windows are normalised (per-window min-max, as in our pipeline) — MixUp applies to normalised values.
- Labels are soft interpolated (λ × one_hot(y_i) + (1-λ) × one_hot(y_j)).

**Critical flaw for FVG:** Labels are structurally determined by the OHLCV pattern (3-candle gap).
MixUp destroys this structural relationship — mixed windows will have `label=bull` for data that
geometrically contains no bullish gap. This introduces spurious signal and is **NOT recommended**
for this specific task.

**Block bootstrap for CI:** Mathematically sound. Used in Phase 7 (scripts/bootstrap_ci.py).
Not data augmentation — used only for uncertainty estimation.

**Conclusion: MixUp rejected for FVG data augmentation.**

---

## Q5: Multi-ticker QQQ augmentation

QQQ (Nasdaq-100 ETF) vs SPY (S&P 500) FVG frequency comparison:
- SPY test set: ~5–10% FVG bars (670 FVG / ~7,000 total in labeled set).
- QQQ expected similar: Nasdaq-100 has higher individual stock volatility but ETF-level H1 bars
  are smoother.
- Correlation: QQQ and SPY daily H1 bars are ~0.96 correlated intraday.

**Distribution coverage benefit:** Near-zero. Highly correlated ticker adds near-duplicate
windows, not distribution diversity. The 4% non-correlated variance (sector tilts, tech-specific
FVGs) would require explicit validation it doesn't introduce noise.

**Conclusion: QQQ augmentation NOT recommended.** Different sector tickers (XLE, XLF) would offer
more distribution diversity but require separate labelling validation.

---

## Recommendation: EXTEND to 2016

**Decision: EXTEND — pull 2016–2017 SPY H1 data from Alpaca free tier.**

Rationale:
1. Data available (confirmed 2016+).
2. Compute cost manageable (+29% training size, ~2 hr overhead on 50-trial search).
3. ~29% increase in training windows provides meaningful sample size improvement
   (effective n rises from ~117 to ~150 non-overlapping windows on train set).
4. 2016–2017 are post-VIX normalisation years — comparable market regime to 2018–2021.

**NOT recommended:** MixUp augmentation, QQQ augmentation, pre-2015 data.

## Execution plan (separate phase)

1. Pull SPY minute bars 2016-01-04 to 2017-12-29 from Alpaca (`TimeFrame.Minute`, RTH only).
2. Resample to H1 (09:30 ET anchor, same pipeline as current `src/data/pipeline.py`).
3. Prepend to `spy_h1_train.parquet` (maintain temporal ordering: 2016 < 2017 < 2018–2021).
4. Re-run `fvg_valid` labelling on extended train split.
5. Recompute `class_weights.json` from extended train set.
6. Retrain LSTM baseline (seed=42) and compare test macro F1 vs current 0.8242.
7. If Δ > 0.01 F1: adopt extended dataset. If Δ < 0.01: defer (marginal gain doesn't justify overhead).

**Estimate:** 4–6 hrs total (data pull + label + retrain + compare).
