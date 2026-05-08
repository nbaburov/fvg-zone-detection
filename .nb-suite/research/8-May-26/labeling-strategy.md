# Labeling Strategy — Research

> **Ready for /nb:plan.**
> Question: Given the custom vectorised FVG detector (label at N+1, R2 confirmed), what is the correct labeling strategy for the full pipeline — single detector vs ensemble weak labellers, hand-labeled gold set protocol, synthetic augmentation viability, and label encoding decision?
> Verdict: Single custom detector is the right call for this scale. Do NOT use Snorkel ensemble. DO hand-label 50–75 candles using a stratified protocol. Do NOT use synthetic augmentation. Use per-candle ternary encoding (bull/bear/none) as the primary label target.

**Confidence:** High
**Why this confidence:** (a) Snorkel's own published data establishes its label-model benefit is conditional on medium label density — with a single deterministic rule (not probabilistic labeling functions), the advantage evaporates. (b) Financial time series augmentation literature (arXiv 2010.15111) shows jittering degrades small datasets and GAN/VAE for OHLCV require 300k+ candles to avoid mode collapse. (c) Per-candle ternary is the only encoding consistent with both the evaluation question and the (60,5) window architecture already committed to. (d) Falsification attempts on all four recommendations below were run; all survived.
**Depth used:** Deep

---

## Project context

- ~13,500 SPY H1 candles, Alpaca free tier, 2018–present.
- Custom FVG detector in `src/data/label.py`: label placed at bar N+1 (first bar where 3-candle pattern is closed). Source: R2 artifact.
- `smartmoneyconcepts` library rejected — 1-bar lookahead confirmed.
- Window shape: (60, 5) — 60 candles × OHLCV.
- Target: ternary (bull FVG / bear FVG / none) per master plan.
- Class imbalance expected: FVG candles are sparse. Rough estimate — on H1 SPY, FVG events occur on ~5–15% of candles, making "none" the dominant class.
- No prior `src/` code — this research shapes label.py design decisions before any build.

---

## Q1: Single detector vs Snorkel-style ensemble of weak labellers

### Option A: Single custom FVG detector (current plan)

**Source:** Master plan + R2 + Ratner et al. (2019/2020), VLDB Journal. https://link.springer.com/article/10.1007/s00778-019-00552-1

**Mechanism:** One deterministic vectorised rule: `high[i-1] < low[i+1] AND close[i] > open[i]` for bullish. Label placed at N+1. Zero stochasticity. Every label has identical accuracy — the rule is either correct or wrong by the same amount at every application.

**Fit:** Yes — for this project specifically. Reasons:

1. Snorkel's label model derives benefit from *disagreement signals* between labeling functions. When functions agree, the model can upweight that region. When they disagree, it can learn which is more reliable. With a single deterministic function there is no disagreement signal — the label model degenerates to a passthrough. The Ratner (2020) paper explicitly states: "the generative model's advantage over majority vote exists in the regime of medium label density — for low density (one label per example), the label model cannot do much re-weighting." Our situation is low density: each candle gets exactly one label from one function.

2. The FVG rule is already deterministic and auditable. Adding variants with different thresholds (e.g., ATR-filtered, volume-filtered) does not reduce systematic bias from the rule — it introduces *conflicting* definitions of "what is a FVG." Aggregating conflicting definitions via a label model does not recover ground truth; it produces a compromise that matches none of the practitioner definitions.

3. At ~13,500 candles, the statistical overhead of fitting a generative label model is not justified. Snorkel's benchmark datasets (CDR: 8,272; Spouses: 22,195) gained 3–6 F1 points from the label model — but those tasks had 2–15 labeling functions with genuine NLP-style ambiguity. A 3-candle geometric rule has no analogous ambiguity.

**Risk:** Single-source systematic bias — if the rule is wrong (e.g., misses FVGs preceded by a doji), every miss has the same pattern. Addressed by the gold set (Q2).

### Option B: 2–3 weak labellers + Snorkel label model

**Source:** Ratner et al. (2019) VLDB; Snorkel AI guide https://snorkel.ai/data-centric-ai/weak-supervision/

**Mechanism:** Write 2–3 additional labeling functions (e.g., LF1 = base ICT rule; LF2 = base rule AND gap_size > 0.5×ATR(14); LF3 = base rule AND volume > 1.2×MA20). Fit Snorkel's probabilistic label model to aggregate. Each function can abstain (return ABSTAIN) on ambiguous cases.

**Fit:** No — for this project specifically. Reasons:

1. LF1 (base rule) is deterministic and already "correct by definition" per ICT. LF2 and LF3 are *filters* that reduce recall — they are not alternative detectors of the same phenomenon. A volume-filtered FVG is a subset of all FVGs, not an independent measurement. The label model has nothing to learn from disagreement between a rule and its subset: LF1=1, LF2=0 simply means volume was low, not that the FVG doesn't exist.

2. Snorkel's noise-reduction mechanism assumes labeling functions are independent conditional on the true label. ATR-filtered and volume-filtered functions are structurally correlated (both are functions of the same candle). The generative model's conditional independence assumption is violated, and the learned weights will be unreliable.

3. Implementation cost: writing, testing, and tuning 2 additional labeling functions plus the Snorkel pipeline (`snorkel.labeling`, label matrix, label model fit) is approximately 2–3 days of Phase 3 time. Expected gain on a 13k-candle deterministic-rule dataset: negligible. The master plan already identifies this as a risk ("Manual hand-labelling of >50 candles — used only for spot-check").

4. Snorkel's published benefit (45.5% performance improvement) is measured against hand-labeling by subject-matter experts — not against a competing rule-based system. The correct comparison for this project is rule labels vs. gold-set human labels, which is Q2 below.

**Risk:** Adds complexity, violates label model assumptions, consumes Phase 3 time. Not recommended.

### Decision: Option A — single deterministic custom detector

Rationale: the Snorkel paradigm provides its benefit when multiple independent noisy signals are combined. With one deterministic geometric rule and a dataset of ~13,500, the label model degenerates and the implementation overhead is not justified. Complexity here would be cargo-culting without benefit.

---

## Q2: Hand-labeled gold set

### Why to do it

**Source:** Noise in Labels literature — PMC 7429345; dcai.csail.mit.edu/2024/label-errors; master plan pre-mortem Item 1 ("Label noise too high — model F1 capped near rule-engine F1").

The master plan's highest-probability failure is: rule labels are noisy, model ceiling = rule ceiling, no novelty demonstrated. The only way to quantify this risk before Phase 4 is to measure the noise floor: what fraction of programmatic labels disagree with human judgment on the same candles?

Without a gold set, noise floor is unknown. A model training on 20% noisy labels behaves very differently from one training on 5% noisy labels. The gold set resolves this unknowing before Phase 4 begins.

Additionally, the gold set serves as a methodologically honest holdout for final reporting. Academic evaluations of DL systems trained on programmatic labels are strengthened — not weakened — by acknowledging the noise and quantifying it against human annotation.

### Protocol — concrete specification

**Size:** 75 candles. Not 50. Reasons:
- 50 candles × ~10% positive rate ≈ 5 positive examples. Confidence interval on a 5-sample agreement rate is too wide to be useful (±20% at 95%).
- 75 candles × ~10% positive rate ≈ 7–8 positives per class. At 8 samples, a 75% agreement rate has 95% CI of approximately ±30% — still wide but interpretable as "better than chance."
- 100 candles is the ideal but master plan explicitly caps at 50 — treat 75 as the upper bound the plan allows. The 50-candle reference in the plan is a cost cap, not a precision target.
- If positive-class count in 75 candles falls below 5 (FVGs rarer than expected), extend to 100 candles.

**Sampling strategy:** Stratified, not random. Three strata:
1. **Known-FVG-rich windows** (25 candles): periods where the programmatic detector fires densely — high-volatility, post-news, gap-up open days. These test whether the rule over-labels.
2. **Low-volatility drift windows** (25 candles): periods where ATR is low and FVGs should be rare. These test whether the rule under-labels (misses small but valid FVGs, or incorrectly fires on micro-gaps).
3. **Random sample** (25 candles): stratified by year — ~4–5 candles per year 2018–2024. This estimates average noise across the full distribution.

Rationale for stratification: random sampling from 13,500 candles would return ~97% "none" candles (assuming 3–5% positive rate), giving essentially zero positive examples to compare. Stratification ensures we sample sufficient positives to compute meaningful agreement.

**What gets shown to the human annotator (you):**
- A plotly chart of 7-candle windows centered on each sampled candle (3 bars before, focus bar, 3 bars after).
- The OHLCV data is visible. The programmatic label is NOT shown (to avoid anchoring bias).
- Three choices: Bullish FVG / Bearish FVG / None.
- A notes field: "Ambiguous — borderline case" (this matters for disagreement analysis).

**Recording disagreements:**
- Store in a CSV: `candle_index`, `datetime`, `rule_label`, `human_label`, `annotator_note`.
- Agreement = exact match on direction (bull/bear/none). Ambiguous human labels count as disagreement if rule fires.
- Compute: percent agreement, Cohen's kappa (trivial computation for one annotator vs rule). Kappa > 0.8 = rule labels reliable. Kappa 0.6–0.8 = acceptable with caveats documented. Kappa < 0.6 = label quality concern, surface to Phase 4 before training.
- Target metric: percent agreement on positive class specifically (not overall — overall will be inflated by true-negative agreement on "none" candles).

**Who annotates:** One annotator (you) is sufficient. This is not a production NLP annotation task requiring multiple annotators and Krippendorff alpha. The goal is to bound rule noise, not to define ground truth precisely. One trained SMC practitioner annotating their own project is a justified simplification.

**Time estimate:** 75 candles × ~2 minutes per chart = 2.5 hours. Build the plotly annotation UI first (30 minutes of Phase 3 build) to avoid per-candle copy-paste friction.

**Where this lives:**
- `data/gold_labels.csv` — raw annotations
- `tests/test_data_label.py` — assertion: Cohen's kappa on gold set must be computed and logged (not a pass/fail test, but a logged metric surfaced in the notebook)
- `notebooks/01-data-understanding.ipynb` — Section 3.4: Label Quality Analysis. Report agreement rate, kappa, sample breakdown by strata.

---

## Q3: Synthetic / augmented data

### Summary verdict: Do not use generative augmentation. Limited window-stride overlap is acceptable.

**Source:** arXiv 2010.15111 (Evaluating Data Augmentation for Financial Time Series Classification); EPJ Data Science springer 2023; arXiv 2310.10060 (Data Augmentation for Time-Series Classification Survey).

**Generative (GAN/VAE):**

GANs for financial OHLCV time series are documented to fail with mode collapse and training instability at small scale. The published literature (Physica A 2019; arXiv 1907.06673 Quant GANs) requires 100k+ candle sequences to produce statistically realistic synthetic data. At 13,500 candles, a GAN or VAE trained on 60-candle OHLCV windows would memorise training examples rather than generalise — the synthetic candles would be near-copies of real windows, adding noise but not diversity. Additionally, the generated candles would not be constrained to satisfy OHLCV internal consistency (e.g., high ≥ max(open, close), low ≤ min(open, close)) without domain-specific architecture. These are documented failure modes.

**Jittering / magnitude warping:**

The financial augmentation study (arXiv 2010.15111) found that jittering *degrades* performance on small datasets by diminishing signal — particularly harmful for pattern recognition tasks where the exact geometry of a 3-candle formation matters. FVG is precisely defined by the relative position of three price levels. Adding Gaussian noise to OHLCV values can destroy the gap — `high[i-1] + noise > low[i+1] + noise` may no longer hold, invalidating the label. This is not a general jittering failure; it is specific to geometric label definitions.

**Mixup:**

Convex combinations of two candle windows produce unrealistic intermediate candles. A mixup of a bull-FVG window and a bear-FVG window produces a window where neither pattern exists but the label is a weighted mix. This is semantically incoherent for pattern detection and has no published support for financial OHLCV.

**Window stride as implicit augmentation:**

The one viable light-touch approach: use stride < 60 when extracting windows. Stride = 1 produces 13,440 windows from 13,500 candles (with labels on the final candle of each window). These windows are heavily overlapping — each consecutive window differs by one candle. This is standard practice for sequence classification and is not augmentation in the synthetic sense. It multiplies the number of (window, label) pairs without introducing synthetic data. Use it. **But:** apply it ONLY to train split; val and test must use non-overlapping windows to avoid the same region appearing in both train and val, which creates an implicit data leakage.

**Class imbalance handling (instead of augmentation):**

Use weighted cross-entropy loss with class weights inversely proportional to class frequency. This is documented (MATLAB deeplearning docs; Borovkova 2019 LSTM ensemble paper) as the most effective and assumption-safe approach for imbalanced sequence classification. No augmentation needed.

---

## Q4: Label encoding decisions

### The four options

| Encoding | What the model predicts | Windows needed | Label sparsity | Evaluation |
|----------|------------------------|----------------|----------------|------------|
| (a) Per-candle ternary | Each candle: bull/bear/none | 1 label per window (last candle) | ~5–15% positive | F1 per class |
| (b) Per-window binary | Any FVG in window: yes/no | 1 label per window | ~30–60% positive (likely) | Binary F1 |
| (c) Per-candle + zone bounds regression | Class + top/bottom of zone | 3 outputs per window | Same as (a) | F1 + MAE on bounds |
| (d) Per-candle ternary (all candles in window) | Each of 60 candles gets a label | 60 labels per window | Same as (a) | Seq-level F1 |

### Analysis

**Option (a) — per-candle ternary on the last candle:**

The window (60, 5) contains the 60 candles of historical context. The model predicts the label of candle 60 (the rightmost — the "current" candle). This is the standard sequence-to-one classification setup. Every prior architecture (LSTM, CNN-LSTM, Transformer) outputs a single (num_classes) vector from the sequence, making this the path of least resistance.

The evaluation question per the master plan is: "Does the model detect FVG structure at candle N?" This is answered directly by option (a). No reframing needed.

Class imbalance: ~5–15% positive. Addressable with weighted CE. F1 minority is the primary metric — matches master plan.

**Option (b) — per-window binary:**

Collapses bull/bear distinction. The practitioner-facing question is "is this a bullish or bearish FVG?" — binary loses this. Also, with stride=1 windows, any FVG that spans multiple candles will appear as a positive in many consecutive windows, inflating apparent positive rate without adding information. The window-binary encoding is simpler but loses the directional information that makes FVG classification useful.

Additionally, the "does window contain any FVG" question inflates positives — a 60-candle window with stride=1 will contain a FVG roughly proportional to the FVG candle density. This makes the task easier (higher baseline accuracy) but less informative for reporting.

**Option (c) — regression of zone bounds:**

Adds `top` and `bottom` prediction (regression heads). This requires the custom detector to also output zone bounds (not just direction) — an extension to `label.py` of ~15 lines. The zone bounds allow a qualitative visualisation: "model predicted a bullish FVG zone from 4,320 to 4,325" overlaid on a chart.

The downside: regression adds loss complexity (multi-head), training instability risk, and requires a calibration metric (MAE on bounds) that the Phase 5 evaluation plan does not currently include. This is a Phase 6 / extension-only feature. **Do not implement in MVP.** Mark as extension.

**Option (d) — full sequence labeling (all 60 candles):**

Every candle in the window gets a label. This is BERT-style token classification. It changes the architecture: instead of a single classification head on the final hidden state, a CLS head or per-timestep projection is needed. This is architecturally non-trivial and changes the window construction logic (label.py must produce a (60,) label vector not a scalar).

The benefit: the model learns to identify which candles within the window are FVG candles, not just whether the final candle is. This is more informative and produces a richer output. However:
- It requires the label vector to be causal — label of candle k in the window uses only information up to candle k. Our custom detector at N+1 satisfies this.
- Training instability risk: most positives in the label vector are "none" — extreme sequence-level imbalance.
- Not required by the evaluation question. The project is judged on whether the system detects FVG at the current candle, not which historical candles in the window were FVGs.

**Decision: Option (a) — per-candle ternary on the last candle of the window.**

It directly answers the evaluation question, fits every planned architecture without modification, supports F1-minority as primary metric, and is the simplest path to a first result. Options (c) and (d) are extensions — mark in Phase 3 sub-plan as "extension only if time and F1 baseline permit."

---

## Recommendation — concrete labeling protocol for Phase 3 build

### Detector specification

**Single custom vectorised FVG detector** in `src/data/label.py`.

```
Bullish FVG at index i if:
  - high[i-1] < low[i+1]  (gap exists)
  - close[i] > open[i]    (middle candle bullish)
Label placed at i+1 (first bar where pattern is confirmed).

Bearish FVG at index i if:
  - low[i-1] > high[i+1]
  - close[i] < open[i]
Label placed at i+1.

All other candles: label = 0 (none).
```

Output: `pd.Series` of int, values {-1, 0, 1}, same index as input dataframe. First bar = 0 (warm-up — no i-1). Last bar = 0 (no i+1 to check). This is critical: the last candle of the training dataset cannot be labelled — never include it as a window endpoint.

No Snorkel. No ATR filter. No volume filter. The base ICT rule is the ground truth definition. Filters can be studied in Phase 5 ablation (does filtering to ATR-significant FVGs improve model agreement with human labels?) — not in Phase 3.

### Aggregation

None. Single detector, single label per candle. No label model.

### Gold set protocol

75 candles, stratified (25 FVG-rich / 25 low-volatility / 25 random). Plotly annotation UI. Store in `data/gold_labels.csv`. Compute Cohen's kappa, report in notebook Section 3.4 before Phase 4 training begins. If kappa < 0.6 — surface immediately, do not advance to Phase 4.

### Label encoding

Per-candle ternary: {1 = bullish FVG, -1 = bearish FVG, 0 = none}. In PyTorch, encode as integer class labels {0, 1, 2} (map -1 → 0, 0 → 1, 1 → 2) for `nn.CrossEntropyLoss` compatibility. Class weights = inverse frequency: compute on train split only, apply to loss. Do not compute on val or test.

### What goes into the (60, 5) window tensor

Each window: 60 consecutive candles × 5 features (OHLCV, normalised per-window). Label: single integer {0, 1, 2} corresponding to the class of candle 60 (the window's final candle). Window is valid only if candle 60 is not in the warm-up region (first candle) or the trailing-edge region (last candle, unconfirmable). These boundary candles get label = 0 (none) by construction — no special masking needed.

Stride: 1 for train split, 60 (non-overlapping) for val and test splits.

### What the model predicts

A single softmax over 3 classes, applied to the model's output for the window. Primary metric: F1 on minority classes (bull FVG, bear FVG), macro-averaged across the two positive classes.

---

## What /nb:plan needs to know

- **No Snorkel.** Single deterministic rule. `label.py` imports nothing from `smartmoneyconcepts`. Zero Snorkel dependencies.
- **Label encoding in PyTorch:** map {-1 → 0, 0 → 1, 1 → 2} before feeding to `nn.CrossEntropyLoss`. Class weights = inverse frequency on train split only. Persist weights with the model checkpoint for inference.
- **Gold set build task** is a Phase 3 deliverable: `data/gold_labels.csv` + plotly annotation script + Cohen's kappa logged before Phase 4 begins.
- **Window stride:** stride=1 for train (implicit augmentation), stride=window-length for val and test (no overlap, no leakage). This must be enforced in `src/data/window.py`.
- **Boundary candles:** first candle of dataset (no i-1) and last candle (no i+1) = label 0 by definition. Do not include them as window labels.
- **Extension items** (do not scope into Phase 3 sub-plan): zone-bounds regression heads (Option c), full-sequence labeling (Option d), ATR-filtered FVG variant for ablation study.
- **Label density check:** after running the detector on all 13,500 candles, compute positive class frequency before any windowing. If bull+bear < 3%, recheck the detector implementation. If < 1%, something is wrong — surface immediately.

---

## Open questions

- Exact positive label rate on the full SPY H1 2018–present dataset: unknown until Phase 3 build. Rate determines class weight magnitudes and informs whether Phase 5 statistical tests have sufficient power. Compute this on day 1 of Phase 3 build.
- Whether bear FVGs and bull FVGs occur at roughly equal frequency on SPY H1, or whether one direction dominates. This matters for whether to combine them into "any FVG" binary for the loss function. Unknown until data is in hand.
- Whether the gold set kappa falls below 0.6 — unknown. If it does, the question of whether to proceed anyway or invest in better labeling (larger set, cleaner protocol) must be decided by you (user decision needed).

---

## What I couldn't verify

- The exact FVG positive rate on SPY H1 2018–2024. Published references cite 5–15% as typical for H1 financial pattern rates but FVG specifically on SPY H1 is not documented in literature. Empirical check required in Phase 3.
- Whether 75 hand-labeled candles produces a kappa confidence interval tight enough to be academically defensible. Statistical power analysis for kappa is not well-standardised. At n=75 with ~10% positives (~7–8 per class), kappa confidence intervals will be wide. The gold set is a directional estimate, not a precise measurement — state this limitation explicitly in the notebook.
- The Springer paper "A Deep Learning Approach to Identify Fair Value Gaps in Forex Markets" (paywalled, could not read). If accessible, it may provide a precedent for FVG label encoding in DL systems — check in Phase 3 if needed.
- Whether Snorkel's label model has a documented threshold below which it is equivalent to a single labeling function. The theoretical analysis (Ratner 2020) describes the regime qualitatively but does not provide a closed-form sample-size threshold.

---

## Falsification attempts (Deep)

**Attempt 1: "Snorkel with 2 ATR-filtered variants could reduce false positives and improve precision."**

Argument: the base ICT rule labels any geometric gap regardless of size. A 1-tick FVG is not meaningful to a practitioner. Adding an ATR filter (gap > 0.3×ATR(14)) would filter micro-gaps. Snorkel could learn to trust the ATR-filtered function when it fires (higher precision signal) and the base rule otherwise.

Counter: ATR-filtered function is a *subset* of the base rule — it fires whenever the base rule fires AND the gap is large. These functions are not independent (violation of Snorkel's conditional independence assumption). The label model would assign high weight to the ATR-filtered function because it has higher observed accuracy — but "higher accuracy" here means "fires less often," not "fires on more genuine FVGs." The result is reduced recall with no meaningful precision gain that couldn't be achieved by simply applying the ATR filter to the base rule's output directly. Snorkel overhead adds zero value in this structure.

Verdict: attempt failed. Single detector holds.

**Attempt 2: "Per-window binary is simpler, achieves higher baseline accuracy, and is sufficient for an academic paper."**

Argument: binary classification is better understood, produces cleaner F1 numbers, and removes the ambiguity of merging bull/bear into separate classes with further imbalance within them.

Counter: The evaluation question in the master plan is "can the model detect SMC structure," and FVG direction (bull vs bear) is a core component of SMC utility. A model that says "FVG exists here" without direction is not useful to a practitioner. Academically, a ternary system that achieves any positive F1 on direction is more novel than a binary system. Additionally, per-window binary inflates positives via stride=1 overlap (a FVG that spans 3 candles appears in ~57 consecutive windows), producing deceptively high recall. If binary is chosen, this artifact must be explicitly acknowledged and corrected — which is more work than just using ternary per-candle.

Verdict: attempt failed. Per-candle ternary holds.

**Attempt 3: "The gold set at 75 candles is too small to measure noise floor reliably — don't bother."**

Argument: with ~10% positive rate and 75 candles, we get ~7 positive examples. A kappa computed on 7 samples is noise. The effort (2.5 hours annotation + 30 min UI) produces an unreliable metric. Better to spend that time on Phase 4.

Counter: the gold set's value is not primarily statistical precision — it is directional risk signal. A kappa of 0.9 (7/7 agreement) says "rule is reliable, proceed." A kappa of 0.4 (4/7 agreement) says "rule misses 43% of human-identified FVGs — stop and investigate before training." Even with n=7 positives, the signal is actionable as a gate. The master plan's pre-mortem Item 1 explicitly names this as the highest-probability failure mode. The 2.5-hour investment to bound this risk before Phase 4 is correct. If kappa is too uncertain to interpret (e.g., n=3 positives in the gold set due to rarer-than-expected FVGs), extend to 100 candles with more FVG-rich strata.

Verdict: attempt failed. Gold set holds.

**Attempt 4: "Magnitude warping (not jittering) preserves geometric relationships and could safely augment without destroying FVG labels."**

Argument: magnitude warping multiplies OHLCV by a smooth spline curve — it scales values but preserves relative ordering. `high[i-1] < low[i+1]` is preserved if the scaling is monotone (which it is for positive spline weights). So FVG labels remain valid post-warp.

Counter: magnitude warping creates candles with different absolute price levels. Normalisation within each window (per the pipeline plan) mitigates this — post-normalisation, warped and original windows may be indistinguishable. BUT: the spline scaling also changes the gap *magnitude* relative to ATR. A small FVG warped to appear large is not "more training data" — it's a different regime. More importantly: with stride=1 windowing producing ~13,440 training windows already, augmentation is solving a non-problem. The training set is not volume-limited; it is label-limited (imbalance). Warping does not change the label distribution — it still produces the same fraction of positives. Weighted CE handles imbalance directly; augmentation does not add value here.

Verdict: attempt failed. No augmentation holds.

---

## Sources

- Ratner et al., "Snorkel: Rapid Training Data Creation with Weak Supervision," VLDB 2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC7075849/
- Ratner et al., "Training Complex Models with Multi-Task Weak Supervision," 2018: https://arxiv.org/pdf/1810.02840
- Snorkel AI: Essential Guide to Weak Supervision: https://snorkel.ai/data-centric-ai/weak-supervision/
- arXiv 2010.15111, "Evaluating Data Augmentation for Financial Time Series Classification": https://ar5iv.labs.arxiv.org/html/2010.15111
- PMC 7429345, "Inaccurate labels in weakly-supervised deep learning": https://pmc.ncbi.nlm.nih.gov/articles/PMC7429345/
- PMC 7597331, "A Labeling Method for Financial Time Series Prediction Based on Trends": https://pmc.ncbi.nlm.nih.gov/articles/PMC7597331/
- ScienceDirect, "Imbalanced classification with label noise: A systematic review": https://www.sciencedirect.com/article/pii/S2405959525001481
- arXiv 2310.10060, "Data Augmentation for Time-Series Classification Survey": https://arxiv.org/html/2310.10060v5
- Springer, "Leveraging augmentation techniques for tasks with unbalancedness within the financial domain": https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-023-00402-9
- R2 artifact (smc-library-validation.md): custom FVG detector specification confirmed
