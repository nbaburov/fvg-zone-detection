# Architecture Comparison — SMC FVG Sequence Labelling on SPY H1 — Research

> **Ready for /nb:plan (Phase 4 sub-plan).**
> Question: Which deep-learning architecture(s) should we use to classify 60-candle SPY H1 windows into {none, bullish FVG, bearish FVG} given 6,997 train windows, 25% positive rate, and Apple Silicon (no CUDA)?
> Verdict: Keep master plan order **LSTM → CNN-LSTM → xLSTM → Transformer** but (a) make CNN-LSTM the *expected best* DL model rather than xLSTM, (b) add an XGBoost boring-baseline as a control to be run before LSTM, (c) plan xLSTM as a documented negative-result candidate, not a frontrunner, (d) treat the Transformer workstream as time-permitting only and lightweight (≤1 encoder layer, ≤2 heads).

**Confidence:** Medium-High
**Why this confidence:** Multi-source corroboration on three load-bearing claims — (1) Transformers underperform LSTMs at <10k samples (multiple 2024–26 sources, OpenReview paper 2309.11400, MachineLearningMastery), (2) xLSTM's mLSTM/sLSTM advantages are observed at long-sequence/large-data regimes, with at least one 2025 study (MDPI Mathematics 14/8/1282) finding xLSTM *worse* than vanilla LSTM on short-term financial forecasting, (3) NX-AI xlstm package supports Apple Silicon via `native` kernels (confirmed against the official README). Medium not High because: (a) no published paper benchmarks CNN-LSTM vs xLSTM specifically on a 60-step OHLC FVG-style task, so the ranking between them on *this* exact setup is extrapolated, (b) the Springer 2025 FVG-DL paper exists but is paywalled and I could not read methodology details, (c) "minority-F1 ballpark" estimates below are extrapolated from analogous candlestick-pattern literature, not direct measurements.
**Depth used:** Deep / Multi-perspective (synthesised from five investigative angles — implementer, skeptic, auditor, domain expert, pragmatist — without spawning sub-agents because the question decomposed cleanly into source queries).

---

## Project context

- **Task:** Per-window 3-class classification (one label per 60-candle window — NOT per-candle sequence labelling). Confirmed in `phase3-data-preparation.md` Foundation 6: `__getitem__(idx) -> (Tensor[60,5], int)`. The model output is `(B, 3)`, not `(B, 60, 3)`. This dramatically changes architecture choices vs true sequence labelling.
- **Dataset:** 6,997 train / 29 val / 58 test windows. Train uses stride=1 so windows overlap heavily; effective independent samples are far fewer than 6,997 (R&D rule of thumb: independent windows ≈ N / window_size = ~117). Val/test stride=60 gives non-overlapping evaluation.
- **Class distribution:** 75/15/10 (none/bull/bear). Positive rate ~25% — **NOT extreme imbalance**. This is moderate. Class weights `[0.22, 1.10, 1.69]` are mild.
- **Hardware:** MacBook Pro M4 Pro, 48 GB, no CUDA. PyTorch MPS backend.
- **Pre-decided constraints:** Input `(B, 60, 5)`, output `(B, 3)`, weighted CE loss already specified, custom FVG detector already validated as causal, no `smartmoneyconcepts` dependency.
- **Time budget:** 17 May – 4 June (19 days) for *all four* architectures + write-up.
- **Prior research already done (read):** R3 labeling-strategy, R4 timeframe-decision, R5 data-quality, all in `8-May-26/`. These constrain — they don't decide architecture.

Critical realisation that changes the framing: with only 6,997 *highly overlapping* train windows and ~25% positive rate, this is closer to a **medium-tabular-classification problem with a temporal axis** than a "long-sequence learning" problem. That dramatically rebalances the architecture comparison toward simpler models.

---

## Findings

### Option A — LSTM baseline (2-layer stacked, 64u, dropout)

**Sources:**
- Fischer & Krauss (2018), *Eur. J. Oper. Res.* 270(2): 654–669 — established LSTM > logistic regression / RF / DNN on S&P daily directional. Used 2-layer LSTM.
- 2024–26 reviews (MachineLearningMastery, OpenReview 2309.11400, Ryz Labs 2026) consistent: "LSTM remains a strong choice for smaller datasets where consistency is crucial."

**Mechanism:** Sequential gate-controlled recurrence over 60 steps. Final hidden state → linear → 3 logits. Inductive bias is "smooth temporal dependence with limited long-range memory" — adequate for 60-step windows.

**Fit for this project:** **Yes.** Sample size matches the regime where LSTM is empirically reliable. 60 steps is well within practical LSTM range. MPS support is present though *slower than CPU on M-series for LSTMCell* (PyTorch issue #138898) — recommend CPU for LSTM or full sequence-batched `nn.LSTM` on MPS (which is faster than per-step `LSTMCell`).

**Hyperparameter starting points (justified, not invented):**
- Layers: 2 (Fischer & Krauss baseline; deeper rarely helps at this scale)
- Hidden: 64 (matches `idea.md`; total params ~50k — well below the ~200k "danger zone" for 7k overlapping windows)
- Dropout: 0.3 between LSTM layers + 0.5 on classifier head
- Bidirectional: **No** for MVP. Bidirectional sees future within window — for FVG specifically the *label is at position 59*, so a bidirectional net could "see ahead" to non-existent future bars. Keep unidirectional, take final hidden.
- Sequence length: 60 (locked by Phase 3)
- Optimiser: AdamW, lr=1e-3, weight_decay=1e-4
- Batch size: 64 (gradient noise helps regularise small datasets)

**Expected minority-F1 ballpark:** macro-F1 0.40–0.55, minority (bull or bear) F1 0.30–0.45. **Confidence: Medium.** Anchored on (a) Fischer & Krauss showing LSTM > naive on similar size, (b) related candlestick-CNN work hitting 0.5–0.7 F1, (c) our task is harder than candlestick recognition because FVG is rarer and labels are weakly supervised.

**Risk:** May plateau at "barely-better-than-majority" if 25% positives concentrate at window edges (label at position 59 means LSTM only "sees" the FVG-defining triple at the very end; without architectural support the LSTM may have to compress the 3-bar pattern into a single state update). This risk is **the empirical motivation for trying CNN-LSTM next**.

**Time budget:** 2 days build + train + eval (smallest model, debug-friendly, gold-standard baseline).

---

### Option B — CNN-LSTM (1D conv kernel 3 + 64u LSTM)

**Sources:**
- Chen & Tsai (2020) — 90.7% on candlestick CNN.
- Ramadhan et al. (2022) — 82.7% on CNN-LSTM candlestick.
- 2024 ICML conference paper "Investigating Market Strength Prediction with CNNs on Candlestick Chart Images" — F1/AUC ~0.7 across asset blends.
- Bai et al. (2018) "Empirical Evaluation of Generic Convolutional and Recurrent Networks" (TCN paper) — convolutional front-ends consistently match or beat RNN-only on short sequences with local structure.

**Mechanism:** 1D conv with kernel=3 directly encodes the 3-candle FVG geometry as an inductive bias. Output of conv is a sequence of "FVG-likelihood-ish" features per position; LSTM then aggregates context. This is the architecturally-correct match to FVG's definitional locality.

**Fit for this project:** **Yes — strongest match of the four.** FVG is by-construction a 3-candle pattern, so kernel=3 is not arbitrary, it is *exactly* the receptive field the rule encodes. The LSTM's job is reduced to "given a per-position FVG-signal, decide if the latest candle is part of a confirmed pattern" — a much easier learning problem than the LSTM-only case. PyTorch MPS supports `nn.Conv1d` natively (one known bug at >65k channels — irrelevant; we use ≤64).

**Hyperparameter starting points:**
- Conv front: `Conv1d(in=5, out=32, kernel=3, padding=1)` then `Conv1d(32, 32, kernel=3, padding=1)` (two-layer to also capture *contextual* 5-candle frame, e.g. trend leading into FVG). ReLU. Dropout 0.2.
- LSTM: 1 layer, hidden=64 (one layer is enough — conv already extracts features). Dropout 0.3 on output.
- Classifier head: linear(64 → 3), dropout 0.5.
- Total params ~30–40k.
- Same optimiser/lr/batch as LSTM.
- Optional ablation (Phase 5): kernel=5 (encode "fuller" pattern), kernel=3 only (no second conv).

**Expected minority-F1 ballpark:** macro-F1 0.50–0.65, minority F1 0.40–0.55. **Confidence: Medium-High.** Architectural match is strong; supporting literature (Chen & Tsai, Ramadhan, Bai et al.) consistently shows conv front-ends help at this exact pattern-detection sub-task.

**Risk:** Two-conv stack might over-smooth and *hide* the precise 3-candle geometry. Mitigation: include kernel=3-only ablation in Phase 5. Also: if conv extracts the FVG perfectly, the LSTM becomes a pass-through and we essentially have a CNN classifier — that would itself be a finding worth reporting.

**Time budget:** 2 days (very similar to LSTM, slightly more to ablate kernel sizes).

---

### Option C — xLSTM (sLSTM + mLSTM blocks via `xlstm` package)

**Sources:**
- Beck et al. (2024) — original xLSTM paper, arXiv:2405.04517.
- NX-AI/xlstm GitHub README — explicitly states native PyTorch kernels available for non-CUDA platforms. Configure `step_kernel="native"`, `sequence_kernel="native_sequence__native"`, `chunkwise_kernel="chunkwise--native_autograd"` (or `chunkwise--native_custbw`).
- **MDPI Mathematics 14(8):1282 (2025)** — *"Beyond xLSTM: A Comparative Study of sLSTM and mLSTM for Short-Term Financial Forecasting"* — finding: "both sLSTM and mLSTM substantially outperform xLSTM, which requires the longest training time and produces the highest errors on financial forecasting tasks." Suggests xLSTM block stacking adds harm at small/short-term scale.
- xLSTMTime (MDPI AI 5/3/71, 2024) — xLSTM superior on long-horizon time series forecasting (>500 steps).
- Apple's xLSTM-metal community port exists (MLX-native) but using it diverges us from the official package — not recommended for academic reproducibility.

**Mechanism:** sLSTM = scalar memory with exponential gates and stabiliser state (numerical-stability fix on classic LSTM). mLSTM = matrix memory with covariance update, fully parallelisable. Both designed to extend "memory capacity" and close gap with Transformers on long sequences. Stacked into xLSTM blocks.

**Fit for this project:** **Partial / questionable.** The architectural advantage shows at scale (long sequences ≥1000 steps, datasets >100k). At 60 steps × 7k windows we are nowhere near that regime. The 2025 MDPI study is the strongest contrary evidence — on a structurally similar task (short-term financial forecasting with limited data) xLSTM was *worst* of the three. Our likely outcome: xLSTM ≈ LSTM, possibly worse, with substantially longer training time. **This is academically valuable as a documented negative result** — fits master plan pre-mortem item 3.

**Apple Silicon compatibility:** Confirmed working on MPS *only* with native kernels. CUDA-optimised Triton kernels do not run on MPS. Use:
```python
from xlstm import xLSTMBlockStackConfig, xLSTMBlockStack, mLSTMBlockConfig, sLSTMBlockConfig
# In configs, set:
mLSTMBlockConfig(... mlstm=mLSTMLayerConfig(..., step_kernel="native", sequence_kernel="native_sequence__native", chunkwise_kernel="chunkwise--native_autograd"))
sLSTMBlockConfig(..., backend="vanilla")  # CPU/MPS fallback path
```
Smoke test required day 1 of xLSTM workstream (master plan already specifies this).

**Hyperparameter starting points:**
- Block stack: 2 blocks total (1 mLSTM + 1 sLSTM, alternating per Beck et al.'s recommended ratio). Going deeper is wasted on 7k samples.
- Embedding/hidden dim: 64 (match LSTM baseline so comparison is fair).
- Heads (mLSTM): 4 (default).
- Backend: `"vanilla"` for sLSTM (MPS-safe), native PyTorch for mLSTM.
- Context length: 60 (locked).
- Dropout: 0.2 inside blocks, 0.5 on classifier head.
- Same optimiser/lr/batch as LSTM. Train *longer* — xLSTM needs more epochs to converge per Beck et al.; budget 50 epochs vs LSTM's 30.

**Expected minority-F1 ballpark:** macro-F1 0.40–0.55, minority F1 0.30–0.45 — **roughly matching LSTM.** **Confidence: Medium.** I expect xLSTM to *not* significantly beat LSTM on this scale; the report's value is documenting *why*, not in beating the baseline.

**Risk:** (a) Training instability — sLSTM's exponential gates can NaN early; mitigation = lower lr (1e-4) and gradient clipping (max_norm=1.0). (b) Native kernel slowness on M4 Pro — could blow time budget. Time-box hard to 5 days per master plan risk register. (c) Package install breaks on Python 3.12 due to triton dependency — confirmed by GitHub issues; install with `triton` excluded or use a 3.11 venv as fallback.

**Time budget:** 4–5 days (1 day install/smoke-test + 2 days build + 1 day train (slow on MPS) + 1 day eval/write-up). **Hard cap.**

---

### Option D — Transformer encoder (lightweight, ≤1 layer, ≤2 heads)

**Sources:**
- Vaswani et al. (2017).
- Lim et al. (2021) — Temporal Fusion Transformer (TFT) — explicitly designed for forecasting; overkill for classification.
- OpenReview 2309.11400 (2023) "Transformers versus LSTMs for Electronic Trading" — LSTMs outperform Transformers on most financial tasks at <100k samples.
- 2026 Medium article (Ken @ Medium April 2026) — same conclusion for time-series forecasting.

**Mechanism:** Multi-head self-attention computes pairwise interactions across the 60 positions. Position encoding adds order information. Attention learns which candle pairs matter. No locality bias — must learn the 3-candle FVG pattern from data.

**Fit for this project:** **Marginal / time-permitting only.** With only 6,997 *overlapping* train windows (~117 effectively independent), Transformers will overfit unless drastically regularised. The lack of locality inductive bias is the wrong prior for FVG. Multiple 2024–26 sources confirm the data-hungriness. PyTorch MPS supports `nn.MultiheadAttention` but with known NaN issues under boolean masks + dropout (workaround: `x = x + 0` after attention, or avoid boolean masks).

**Hyperparameter starting points (deliberately tiny):**
- Layers: 1 encoder block.
- Heads: 2.
- d_model: 32 (project 5 OHLCV → 32).
- d_ff: 64.
- Sinusoidal positional encoding.
- Dropout: 0.3.
- Pool: `[CLS]` token prepended, take its final embedding → linear → 3.
- Total params ~10–20k. Larger than this and overfit is near-certain.
- Same optimiser/lr; consider warmup over 5 epochs.

**Expected minority-F1 ballpark:** macro-F1 0.35–0.50, minority F1 0.25–0.40 — **probably worse than LSTM.** **Confidence: Medium.**

**Risk:** Will likely be the slowest to train per epoch on MPS due to scaled_dot_product_attention overhead, *and* most likely to overfit. Honest finding: "Transformer underperforms on this dataset size" is itself a publishable observation aligned with consensus literature.

**Time budget:** 2 days. Skip if behind schedule — master plan already permits this.

---

### Option E — Hybrid: CNN front + xLSTM body

**Sources:** No published 2024–26 paper on exact CNN+xLSTM stack for OHLC; the closest is Beck et al.'s own architectural section discussing combinations.

**Mechanism:** Conv1d kernel=3 encodes FVG-locality, xLSTM block aggregates. In principle combines best of B and C.

**Fit:** **Skip in MVP.** Reasoning: (1) we already have CNN-LSTM as the locality-extractor option; if xLSTM doesn't beat LSTM, CNN+xLSTM almost certainly won't beat CNN-LSTM either, (2) extra complexity costs time we don't have, (3) no literature precedent on this exact combination at this exact scale. Document as "future work" in final report.

---

### Option F — Hybrid: CNN front + Transformer body

**Sources:** Chen et al. 2024 ICML conference work on candlestick + CNN+attention.

**Mechanism:** Patch the 60-step sequence via CNN (kernel=3, stride=1 or 3), feed patches as Transformer tokens.

**Fit:** **Skip in MVP.** Same argument — Transformer is already at risk; adding it to a hybrid before establishing it works alone is putting cart before horse. Listed as future work.

---

### Option G — Stacked CNN-LSTM (2-3 conv layers + LSTM)

**Mechanism:** Multi-scale conv (k=3, k=5, dilated k=3) → LSTM.

**Fit:** **Marginal.** A 2-conv variant is already the recommended Option B configuration. A 3-conv variant becomes a TCN; covered indirectly by Bai et al. (2018) finding TCN ≥ LSTM on local-structure tasks. Worth ablating in Phase 5 but not as a Phase 4 workstream.

---

### Option H (CONTROL) — XGBoost on flattened windows + hand-crafted features

**Sources:**
- D&T Systems blog (2025) "XGBoost Beats LSTM and Transformers on Most Financial Time Series" — corroborated by netanel.io (Shoshan), DEV Community articles 2024–25.
- Multiple 2024 papers showing XGBoost dominant on tabular financial data with <10k samples.
- Consensus rule-of-thumb in field: <10k samples + engineered features → gradient boosting beats DL.

**Mechanism:** Flatten 60×5 = 300 features, plus hand-crafted features: (a) high[i-1] vs low[i+1] gap-size feature for last 3 bars, (b) ATR over window, (c) volume z-score last 3 bars, (d) candle-body ratios. XGBoost classifier with class weights.

**Fit:** **Run as a control before LSTM.** This is the *most likely* candidate to beat all four DL models on this dataset size. Including it as a baseline:
1. Provides a hard floor for "DL adds value" — if no DL model beats XGBoost, that *is* the finding.
2. Lets us report "XGBoost achieves X minority-F1; LSTM achieves Y; CNN-LSTM achieves Z" — a complete academic story.
3. Trains in seconds, costs ~half a day total.
4. Aligns with literature consensus.

**Hyperparameter starting points:**
- `xgboost.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, objective="multi:softprob", num_class=3, scale_pos_weight=...)` (use `sample_weight` from class_weights instead since multi-class).
- 5-fold time-series CV on train for tuning; never touch val/test until final eval.

**Expected minority-F1 ballpark:** macro-F1 0.45–0.60, minority F1 0.35–0.50. **Confidence: Medium-High.** Strong literature consensus on this sample-size regime.

**Time budget:** 0.5–1 day total (build + tune + eval + write-up).

**Recommendation:** **Include.** Add to Phase 4 plan as "Stage 0: XGBoost baseline" before LSTM. Master plan currently has no boring baseline; this fills a real academic gap and de-risks the whole phase.

---

## Recommendation

**Final ranked order for Phase 4 build (revised from master plan):**

1. **Stage 0: XGBoost on flattened windows + 5–10 hand-crafted features** (NEW — 0.5 day). Boring baseline + likely strong performer. Sets the floor for "did DL add value?"
2. **Stage 1: LSTM 2-layer 64u** (2 days). DL baseline. Lit-grounded, reproducible, debug-friendly.
3. **Stage 2: CNN-LSTM (Conv1d k=3 + LSTM 64u)** (2 days). **Expected best DL model.** Architectural match to FVG locality is the strongest of the four.
4. **Stage 3: xLSTM (2 blocks, mLSTM+sLSTM, native MPS kernels)** (5 days hard cap). Documented as either "beats CNN-LSTM" or "negative result with academic explanation." Master plan pre-mortem item #3 already anticipates this.
5. **Stage 4: Transformer (1 layer, 2 heads, d_model=32)** (2 days, time-permitting). Most likely to underperform. Skip without guilt if behind schedule — master plan permits.

**Total budget:** 11.5–12.5 days for all 5 stages. Master plan allots 17–4 June ≈ 18 days for Phase 4 modelling — comfortable fit with 5–6 days buffer for the iteration loops `nb-build` will spawn.

**Hybrids (CNN+xLSTM, CNN+Transformer):** **Skip in MVP.** Document as future work in final report.

**Key reframe vs master plan:** master plan implicitly treated xLSTM as a frontrunner ("after LSTM baseline validated, swap in xLSTM and compare"). Evidence does not support that. CNN-LSTM is the better-motivated frontrunner; xLSTM is a "must-explore-because-teacher-said-so" academic exercise that is most valuable as a documented finding regardless of outcome.

---

## What /nb:plan needs to know

- **Add Stage 0 (XGBoost baseline) to Phase 4 sub-plan.** ~0.5 day. Reuses Phase 3 dataset directly via `build_pipeline()`. New file: `src/models/baseline_xgboost.py`. Already-engineered features (gap size last 3 bars, ATR, volume z-score) fit on top of `build_windows` output.
- **CNN-LSTM is the expected best DL architecture, not xLSTM.** Adjust Phase 4 sub-plan's section ordering and "headline result" framing accordingly. xLSTM is documented exploration, not the climactic comparison.
- **xLSTM Apple Silicon installation:** must use `step_kernel="native"`, `sequence_kernel="native_sequence__native"`, `chunkwise_kernel="chunkwise--native_autograd"`, and `sLSTMBlockConfig(backend="vanilla")`. Smoke-test on day 1 of the xLSTM workstream — if install fails or smoke-test NaNs, fall back to mLSTM-only configuration (master plan already permits this).
- **PyTorch MPS gotcha for Transformer:** known NaN bug on `nn.MultiheadAttention` + boolean mask + dropout on MPS. Workaround: insert `x = x + 0` after attention, or use `nn.functional.scaled_dot_product_attention` with float mask. Document in Transformer workstream day-1.
- **PyTorch MPS gotcha for LSTM:** `nn.LSTMCell` is *slower* on MPS than CPU on M-series (issue #138898). Use `nn.LSTM` (sequence-batched, faster on MPS) — never per-step `LSTMCell` loops.
- **Bidirectional LSTM is unsafe for this label scheme.** Label is at position 59; bi-LSTM at position 59 sees positions 60+ (which don't exist within the window) — but more subtly, bi-LSTM can leak information from the conv-extracted "FVG signal" into earlier hidden states of the backward pass. Keep all RNN variants unidirectional.
- **Effective sample size is ~117, not 6,997.** Stride=1 means ~60× overlap. Regularisation must be aggressive: dropout 0.3+ inside, dropout 0.5 on classifier head, weight_decay=1e-4 minimum, early stopping on minority-F1 (NOT val loss — small dataset → val loss noisy).
- **Training protocol R7 should specify minority-F1 as early-stopping signal, not val loss.** With val set = 29 windows, val loss has high variance; per-class F1 is more stable. R7 will own this.
- **Class weights are mild (`[0.22, 1.10, 1.69]`).** Focal loss is unnecessary — weighted CE is sufficient. R7 should test both anyway as ablation.

---

## Open questions

- **Is XGBoost addition acceptable to the user?** Master plan's "out of scope" section doesn't forbid classical baselines; `idea.md` explicitly compares to "classical methods." User should confirm — adds 0.5 day, removes risk of "no baseline floor" criticism in peer review.
- **Should we add a per-candle (full-sequence) labelling head as an alternative output mode?** Phase 3 plan explicitly excludes Option D from R3 (per-candle labels). Sticking with per-window classification is correct — flag only because per-candle would unlock TCN/Transformer architectures more naturally. Not a blocker.
- **Does the Springer 2025 FVG-DL paper change anything?** Paywalled — could not read. If user has institutional access via Fontys, surface methodology before locking Phase 4 sub-plan. Worst case, we miss a known benchmark to compare against; not a blocker.

---

## What I couldn't verify

- **Springer 2025 "Deep Learning Approach to Identify Fair Value Gaps in Forex Markets"** (10.1007/978-3-032-10670-4_36) — paywalled, redirect to authentication. Could not extract architecture, dataset size, or F1 results. Likely the most directly comparable published work; worth one Fontys library lookup before /nb:plan locks the architecture order.
- **Exact NX-AI xlstm package install on M4 Pro Python 3.12** — verified package supports MPS via native kernels, but did not run install end-to-end. Master plan already mandates day-1 smoke test in the xLSTM workstream.
- **Minority-F1 ballparks** are extrapolated from analogous candlestick-pattern literature, not direct measurements on FVG. Confidence intervals will be wide. Phase 5 evaluation will set the actual numbers.
- **TFT (Temporal Fusion Transformer)** evaluated only via the original 2021 paper and 2024 reviews — did not test on our exact tensor shape. Consensus is "overkill for classification at this scale," which we accepted at face value.
- **MDPI Mathematics 14/8/1282 (xLSTM negative result)** — abstract and conclusion read via search snippets; full methodology behind paywall (403). The "xLSTM worse than sLSTM and mLSTM" claim is the conclusion — could not verify their dataset size matches ours.
- **Stride=1 → effective sample size ~117** is a heuristic argument, not a measured quantity. The true effective sample size depends on the autocorrelation of the FVG signal; could be higher (autocorrelation drops fast in returns space).

---

## Falsification attempts

**Tried to break "CNN-LSTM is the best DL model" with three angles:**

1. *"Pure 2-conv CNN (no LSTM) might tie or beat CNN-LSTM."* — possible. Bai et al. (2018) shows TCN ≥ LSTM on short sequences. Mitigation: include kernel=3-only conv classifier as a Phase 5 ablation. If conv-only beats CNN-LSTM, that's a finding — not a contradiction.

2. *"What if xLSTM actually wins because mLSTM's matrix memory captures FVG context the LSTM misses?"* — possible but contradicted by 2025 MDPI study showing xLSTM worse on short-term financial forecasting at our regime. If it does win, master plan's "negative result" framing flips to a positive finding — also fine. Either way the workstream is worthwhile; ranking only affects which model gets headlined.

3. *"What if XGBoost beats every DL model and the entire DL phase is wasted?"* — survivable. The academic story becomes "DL did not add value over gradient boosting on this scale; here are the hypotheses why" — that's a legitimate research finding (consistent with 2024–26 consensus literature). Mitigation: Stage 0 XGBoost baseline establishes this *before* committing all 18 days to DL, allowing pivot if XGBoost dominates by week 1.

**Survived falsification:** the recommended order. Adjustments made: (a) explicitly added Stage 0 XGBoost (was implicit "consider baseline" originally), (b) downgraded xLSTM from frontrunner to "documented exploration", (c) explicitly capped Transformer scope and made it skippable.

---

## Disagreements surfaced (multi-perspective synthesis)

| Angle | Claim |
|-------|-------|
| Implementer | LSTM is hardest to debug *on MPS* due to known LSTMCell slowness; use `nn.LSTM` not `LSTMCell` loops. CNN-LSTM is easiest to build (well-established components). xLSTM is hardest (novel package, kernel config, install fragility). |
| Skeptic | Strongest case against CNN-LSTM frontrunner: a pure 2-conv CNN classifier (no LSTM) might match it — and would be even simpler. Add as Phase 5 ablation, not a Phase 4 stage. |
| Auditor | Effective sample size is ~117, not 6,997. Every architecture is at risk of overfitting. Mandatory: aggressive dropout, weight_decay≥1e-4, early stopping on minority-F1, fixed seeds (3+ runs per architecture for variance estimate). |
| Domain expert | FVG = 3-candle local pattern. Convolutional kernel=3 is the architecturally-correct prior. Attention without locality bias must learn this from 6997 windows — wrong prior for this task. xLSTM's strengths show at long-context scale we don't have. Literature consensus 2024–26 is XGBoost > DL at this size. |
| Pragmatist | Smallest viable answer: XGBoost + LSTM + CNN-LSTM done well, xLSTM and Transformer documented as exploration. Five stages in 18 days is realistic only with strict time-boxing. |

**Resolution:** All five angles converge on the same final ordering. The single material disagreement was Implementer's (and Skeptic's) push for "skip xLSTM entirely" vs Domain expert's (and master plan's) "must explore xLSTM per teacher feedback." Resolution: **keep xLSTM but cap at 5 days and frame as documented exploration, not headline result.** This satisfies academic obligation (teacher feedback in `idea.md`) while protecting time budget.

---

## Sources

- [NX-AI xlstm GitHub](https://github.com/NX-AI/xlstm)
- [xlstm PyPI](https://pypi.org/project/xlstm/)
- [Beyond xLSTM — sLSTM and mLSTM Short-Term Financial Forecasting (MDPI Mathematics 14/8/1282, 2025)](https://www.mdpi.com/2227-7390/14/8/1282)
- [xLSTMTime: Long-Term Time Series Forecasting with xLSTM (MDPI AI 5/3/71, 2024)](https://www.mdpi.com/2673-2688/5/3/71)
- [A Deep Learning Approach to Identify Fair Value Gaps in Forex Markets (Springer 2025) — paywalled](https://link.springer.com/chapter/10.1007/978-3-032-10670-4_36)
- [Transformer vs LSTM for Time Series — MachineLearningMastery](https://machinelearningmastery.com/transformer-vs-lstm-for-time-series-which-works-better/)
- [Transformers versus LSTMs for Electronic Trading — OpenReview / arXiv 2309.11400](https://openreview.net/forum?id=2L1OxhQCwS)
- [LSTM vs Transformer vs CNN — Time Series Forecasting (Apr 2026 Medium)](https://ligaoke.medium.com/lstm-vs-transformer-vs-cnn-which-deep-learning-model-actually-wins-for-time-series-forecasting-6035589a3448)
- [XGBoost Beats LSTM and Transformers on Most Financial Time Series — D&T Systems](https://dtsystems.dev/blog/xgboost-vs-lstm-financial-time-series)
- [Fischer & Krauss 2018 — Deep Learning with LSTM for Financial Market Predictions (Eur. J. Oper. Res.)](https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652)
- [Bai, Kolter, Koltun (2018) — Empirical Evaluation of Generic Convolutional and Recurrent Networks](https://arxiv.org/pdf/1803.01271)
- [Investigating Market Strength Prediction with CNNs on Candlestick Chart Images (ACM 2024)](https://dl.acm.org/doi/full/10.1145/3690771.3690776)
- [PyTorch MPS — Conv1d issue #134416](https://github.com/pytorch/pytorch/issues/134416)
- [PyTorch MPS — LSTMCell slowness #138898](https://github.com/pytorch/pytorch/issues/138898)
- [Apple Developer — PyTorch MPS](https://developer.apple.com/metal/pytorch/)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
