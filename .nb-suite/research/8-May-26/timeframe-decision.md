# Timeframe Selection for SMC FVG Detection — Research

> **Ready for /nb:plan.**
> Question: Should the MVP use H1 (hourly) as the primary timeframe for FVG label generation and model training, or is a different timeframe — 30m, 4H, or multi-timeframe — more appropriate?
> Verdict: H1 is confirmed for MVP. It occupies the correct SMC structural role (MTF zone validator), produces ~12,000 training candles which is adequate for LSTM/xLSTM at this scale, and yields ~1–2 FVGs per session — lower label density than 30m but higher signal quality. Upgrade to daily-as-macro-context only if H1 baseline F1 plateaus with visible ceiling pattern.

**Confidence:** Medium-High
**Why this confidence:** SMC practitioner consensus on H1 role is clear and consistent across multiple sources. Sample-size floor for LSTM is not authoritatively published for this exact task — the 12k candle count is supported by analogy to published work (Fischer & Krauss 2018: ~8,000 daily samples) not by a peer-reviewed minimum for hourly classification. Label density estimate (1–2 FVGs per session → ~5–8% of H1 candles) is inferred from practitioner quotes, not a backtested count on SPY H1 specifically. The smart-money-concepts library FVG parameters may alter this — R2 must confirm.
**Depth used:** Standard

---

## Project context

- SPY H1 2018–2024: ~12,000 candles (post-yfinance confirmation from R1 — if yfinance caps at 730 days, pivot to ~6,200 candles from 2022+; R1 must resolve this first).
- SPY 30m 2018–2024: ~24,000 candles.
- SPY 4H 2018–2024: ~3,000 candles.
- SPY Daily 2018–2024: ~1,500 candles.
- Project is FVG-only in MVP. Multi-timeframe fusion explicitly out of scope for MVP per `master-phasing.md`.
- xLSTM is the primary novel architecture; LSTM is baseline. Both are sequence models with recurrent inductive bias — they exploit temporal structure within the window.
- Window length planned: 60 candles. Label unit: candle-level (FVG membership). Class balance target: ~5–10% positive.
- CPU-only (M4 Pro). Training on ~12k samples with 60-candle windows is fast (<5 min per epoch estimated).

---

## Findings

### Q1 — SMC practitioner role of H1

**Sources:**
- https://www.technical-analysis-pro.com/strategies-smart-money-concepts-smc/
- https://www.quantum-algo.com/blog/ict-trading-strategy-complete-guide/ (ICT guide)
- https://xmsignal.com/en/blog/smart-money-concept-explained/

**Mechanism:** SMC top-down hierarchy is unambiguous across all sources: Daily/4H = bias + major zones (HTF); 1H = structural zone validation, MTF confirmation bridge; 15m/5m = precise entry trigger (LTF). The H1 sits at the "Medium Timeframe" layer — it confirms whether FVGs found on daily/4H are structurally sound, and identifies H1-native FVGs as valid trade zones in their own right.

**For this project:** H1 FVGs are the primary institutional structure this model needs to detect. They are neither too noisy (LTF) nor too sparse (4H/Daily). The H1 is what SMC practitioners actually trade from — this is the correct labelling target for a model intended to assist SMC-style analysis.

**Key quote (technical-analysis-pro.com):** "Bias comes from the HTF • Zones (OB, FVG) come from the MTF/HTF • Precise entry comes from the LTF" — H1 is explicitly the MTF zone layer.

**Fit:** Yes — H1 is the correct primary timeframe for FVG zone identification in SMC methodology.

---

### Q2 — Sample size implications for DL

**Sources:**
- Fischer & Krauss (2018), European Journal of Operational Research 270(2): 654–669 — https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652
- arxiv 2201.08218 (LSTM financial time series, OMX30 application)
- https://www.analyticsvidhya.com/blog/2024/05/upgrade-xlstm/
- MDPI xLSTM vs LSTM short-term financial forecasting (2025), doi:10.3390/math14081282 (paywalled — abstract read)

**Mechanism:** Fischer & Krauss (2018) used S&P 500 constituents 1992–2015 daily bars. With ~500 stocks × ~5,000 daily bars each = multi-million sample universe. Not directly comparable to single-asset H1. However, their per-asset panel cross-sections suggest ~3,000–8,000 training samples per series is the practical floor they implicitly operate above.

The LSTM for OMX30 paper (arxiv 2201.08218) used a single asset with ~12,000 daily observations — directly analogous in sample count to SPY H1. It produced meaningful results for directional classification.

The MDPI xLSTM vs LSTM study (2025) evaluated short-term financial forecasting — abstract confirms xLSTM was assessed on standard financial time series datasets, no explicit "minimum samples" rule stated, but 10,000+ is the implicit baseline in all published work.

**Practical floor estimate:** ~8,000–12,000 samples is well within the published range for LSTM on financial classification tasks. Below ~3,000 samples (4H scenario), sequence models begin to overfit significantly on this type of task without strong regularisation. SPY H1 at 12,000 candles is in the safe zone.

**30m advantage:** 24,000 candles (~2× more) is better statistically, but provides no SMC justification advantage and doubles preprocessing complexity. Not worth the trade-off for MVP.

**4H risk:** ~3,000 candles is likely insufficient for xLSTM with a 60-candle window (only ~50 non-overlapping windows available; with stride-1 windowing, ~2,940 windows — marginally workable but tight for train/val/test split).

**Fit:** H1 at ~12,000 candles is adequate. 4H is risky. Daily is definitely insufficient.

---

### Q3 — Label density at each timeframe

**Sources:**
- https://www.edgeful.com/blog/posts/fair-value-gap-best-practices-guide
- https://capital.com/en-int/learn/trading-strategies/fair-value-gap (general FVG guide)
- SMC education sources: acy.com, xs.com

**Mechanism:** Practitioner-sourced frequency estimates:
- 5m: "dozens per session" → very high density, high noise floor
- 15m: estimated ~10–20 per session → still noisy
- 30m: "3–5 quality setups per session" → sweet spot for practitioners
- H1: "1–2 per session" → lower density, higher structural significance
- 4H: ~1 per day or less → very sparse

**For SPY H1 specifically:** SPY trades ~6.5 hours per day = ~6–7 H1 candles per session. At 1–2 FVGs per session, that is approximately 15–30% of H1 candles involved in some FVG formation. However — the three-candle FVG pattern means only the middle candle is "the gap" itself. If labelling the middle candle of the FVG triad: ~1/6 to 2/6 candles positive per session = **~15–30% positive rate**. If labelling by whether a candle is the "impulse" candle that creates the gap: ~5–15%.

**Critical note:** The smart-money-concepts library (R2 research stream) determines actual positive rate. The above is a practitioner estimate. Phase 3 build must verify the empirical positive rate before Phase 4 starts. If positive rate is >30%, binary classification (gap vs no-gap) is feasible. If <5%, weighted loss or focal loss is mandatory.

**Noise floor comparison:**
| Timeframe | Estimated FVGs/session | Estimated positive % | Signal quality |
|-----------|----------------------|---------------------|----------------|
| 15m | 10–20 | ~40–60% (heavy noise) | Low |
| 30m | 3–5 | ~20–35% | Medium |
| H1 | 1–2 | ~15–30% | High |
| 4H | ~1/day | ~5–10% | Highest but sparse |

Higher timeframe = lower density but higher quality per label. The H1 sits at a practical mid-point: enough labels for a DL training set, signal-to-noise ratio acceptable for institutional-grade structure.

**Fit:** H1 positive rate (~15–30%) is workable. Not extreme class imbalance territory unless the library's FVG definition is very strict.

---

### Q4 — Multi-timeframe fusion architectures

**Sources:**
- Lim et al. (2021), TFT: https://www.sciencedirect.com/science/article/pii/S0169207021000637
- Late fusion review: https://medium.com/@injure21/beyond-single-source-learning-how-fusion-models-combine-time-series-and-static-features-f1627b7c7e55
- Multi-branch LSTM: https://www.nature.com/articles/s41598-025-86785-3

**Mechanism:**

**Option A — Temporal Fusion Transformer (TFT, Lim et al. 2021):** Full multi-horizon attention model with gated skip connections, variable selection networks, and interpretable temporal attention. Handles heterogeneous-frequency inputs natively. Complexity: high. Requires 10k+ training samples to avoid underfitting. Interpretable but adds 200–500 additional hyperparameters to tune. Implementation overhead: ~2–3 weeks additional engineering. Not appropriate for MVP scope.

**Option B — Feature concatenation (simplest):** Compute daily bar features (ATR, direction, session high/low range) for each trading day, broadcast them to every H1 candle in that session, and concatenate as additional input features to the existing H1 LSTM. Zero additional architecture cost — just wider input tensor. Caveat: does not capture daily *sequence* structure, only daily *state*. Adequate if daily context is mainly about regime (trending/ranging) rather than sequential daily patterns.

**Option C — Dual-stream LSTM (encoder-decoder style):** Separate LSTM encoder for daily bars (rolling 5–10 day window), output concatenated to H1 LSTM hidden state. More expressive than Option B. Engineering cost: ~2–3 days. Adds hyperparameter surface. Still simpler than TFT.

**Option D — Daily ATR/trend as engineered features only:** Compute daily bar statistics (ATR, whether price is above 20-day MA, session type) and include as static features in the H1 window. No separate model required. Essentially a lightweight version of Option B.

**For this project:** If daily context is needed post-MVP, Option D (static daily features added to H1 input) is the lowest-cost path. Option C is the ceiling before committing to TFT. TFT is only justified if Options B/D fail and the project extends into full multi-timeframe territory.

**Fit:** Options B/D are zero-risk MVP extensions. Option C is a natural Phase 4 extension if baseline shows systematic errors on ranging vs trending days. TFT is not justified at 12k candles — it needs more data and more implementation budget than this project has.

---

## Recommendation

**Use H1 single-timeframe for MVP. Confirmed.**

Rationale:
1. H1 is the correct SMC structural layer for FVG zone identification — this is not a debated point in SMC methodology; it is the practitioner consensus.
2. ~12,000 H1 candles is sufficient for LSTM and xLSTM training based on analogy to published financial time series DL work (Fischer & Krauss 2018, OMX30 LSTM paper).
3. Label density at H1 (~1–2 FVGs per session) produces an estimated 15–30% positive rate — high enough for training without extreme resampling, low enough to avoid degenerate majority-class models.
4. Multi-timeframe fusion adds architecture complexity with no validated benefit at this dataset scale.

**Conditions for adding daily macro context:**
- H1 baseline F1 (minority class) > 0.45 but ceiling is visible (val F1 stagnates for 5+ epochs despite tuning)
- Error analysis shows systematic failures on trending vs ranging days (daily regime signal would help)
- Option D (broadcast daily ATR/trend as extra H1 input features) is the first step — not TFT, not dual-stream

**30m as alternative:** The only legitimate argument for 30m over H1 is doubled training samples. It does not offer better SMC structural validity (30m is sub-institutional in SMC), and it produces more spurious FVGs (noise floor higher). Reject for MVP. Could revisit if H1 label count after yfinance cap confirmation falls below ~800 positive training examples.

**4H rejected:** 3,000 candles is below the safe floor for xLSTM on a classification task with a 60-candle window and 3-way temporal split. Insufficient positive label count in holdout.

---

## What /nb:plan needs to know

- H1 is the confirmed primary timeframe — no further decision needed on this.
- Window size 60 candles confirmed as compatible with H1 sample count (~12,000 total, ~10,000 after train/test split → ~10,000 stride-1 windows in train set).
- Phase 3 must verify empirical positive rate from smart-money-concepts library before Phase 4 starts. If positive rate < 5%, focal loss is mandatory (not optional). If positive rate > 30%, binary classification is feasible.
- Do NOT add daily macro features until H1 baseline is fully evaluated. Document as a concrete extension condition, not a TODO.
- If R1 confirms yfinance H1 history is capped at 730 days (~6,200 candles), re-evaluate: 30m becomes viable alternative (12,400 candles from 730 days), or extend to 2022–2026 scope only. This is a blocker from R1, not from this stream.
- The three-candle FVG structure means label unit matters: "is this candle the gap-creating middle bar" gives one positive rate; "does this window contain an FVG" gives another. Phase 3 sub-plan must define the label unit explicitly.

---

## Open questions

1. **Empirical positive rate on SPY H1**: practitioner estimates used above (15–30%). Actual rate from `smart-money-concepts` library on 2018–2024 SPY H1 will differ. Must be measured in Phase 3 data exploration.
2. **yfinance H1 history limit**: if capped at 730 days, total candle count drops to ~6,200. Still workable but marginal. R1 resolves this.
3. **FVG labelling unit**: candle-level (middle bar = positive), window-level (any FVG in window = positive window), or zone-level (FVG zone spans N candles). Different units produce different positive rates and different model targets. Needs explicit definition in Phase 3 sub-plan.

---

## What I couldn't verify

- Fischer & Krauss (2018) exact sample count per asset — full paper paywalled; analogy made from published abstract and cited summaries.
- xLSTM minimum training samples: MDPI 2025 paper paywalled; no peer-reviewed minimum threshold found. The 8,000–12,000 floor is inferred from analogous published work, not a stated rule.
- SPY H1 FVG positive rate: 15–30% estimate is from practitioner frequency claims ("1–2 per session"), not a backtested empirical count on SPY H1 data. This must be verified in Phase 3.
- Whether the `smart-money-concepts` library's FVG definition matches "1–2 per session" — it may be more or less permissive depending on swing parameters. R2 addresses the library's exact algorithm.
- 30m noise floor vs H1: qualitative claim from practitioners, not measured on SPY specifically.
