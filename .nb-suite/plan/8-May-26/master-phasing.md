# Master Phasing — SMC SPY Deep Learning Project

> **Plan type:** Master phasing / research orchestration plan (NOT a build plan).
> **Depth:** Deep (multi-month, multi-stream, irreversible architectural choices).
> **Work type:** New project — greenfield.
> **Methodology:** CRISP-DM (per `~/.claude/nb-suite/CRISP-DM.pdf` + STANDARDS.md §AI/ML methodology).
> **Test strategy:** TDD on code; for ML, "test" = held-out F1 + falsification on label correctness.
> **Date:** 8 May 2026
> **Final delivery:** 20 June 2026 (43 days from now).

**Goal:** Phase the entire project from data retrieval through final delivery so each CRISP-DM phase has answered research questions and concrete sub-plans before any building starts. Avoid building on unverified assumptions.

**Architecture (meta):** Six CRISP-DM phases. Each phase = (1) research stream(s) producing `.nb-suite/research/D-Mon-YY/<topic>.md`, (2) sub-plan in `.nb-suite/plan/D-Mon-YY/<phase>.md`, (3) build cycle producing code + log. Research → plan → build → review → loop. Phases overlap where dependencies allow (parallelisation explicit per phase).

**Tech stack / constraints:** Python 3.12+, PyTorch 2.x, **Alpaca Markets `alpaca-py` SDK (free tier, paper account)** for SPY H1 2018+ with `adjustment="raw"`, **custom FVG detector** (`src/data/label.py` — `smartmoneyconcepts` library rejected due to lookahead), pandas, numpy, plotly, scikit-learn, pytest. CPU-first (M4 Pro 48 GB RAM); GPU not required for ~16k-candle dataset. Notebooks for status updates and final demo. Source code in `src/` per CLAUDE.md.

> **R1 update (8 May 2026):** yfinance dropped — empirical test confirmed Yahoo backend caps H1 history at 730 days (hard server-side limit). Alpaca free tier confirmed as primary source. See `.nb-suite/research/8-May-26/data-source-validation.md`.

> **R2 update (8 May 2026):** `smartmoneyconcepts` v0.0.27 confirmed to use 1-bar future lookahead on FVG labels (source: `smc.py` lines 74–100, `shift(-1)`). PR #95 (`causal=True` flag) unmerged. Falsification test on synthetic candles confirmed leakage. **Custom vectorised FVG detector required** — place label at `N+1` (first bar where pattern is knowable). See `.nb-suite/research/8-May-26/smc-library-validation.md`. `MitigatedIndex` column is unbounded-lookahead — never use as DL feature.

> **R5 update (8 May 2026):** Alpaca `TimeFrame.Hour` is clock-hour aligned (first daily bar 09:00–10:00 mixes pre-market with first RTH half-hour). Must pull `TimeFrame.Minute` and pandas-resample to 09:30-anchored H1. Adds `exchange_calendars` dep for half-day tagging. Cleaning rules ordered checklist in research doc. See `.nb-suite/research/8-May-26/data-quality-baseline.md`.

**Breaking changes:** None — greenfield.

**Out of scope (mandatory):**
- Real-money trading or live broker integration
- Multi-asset (only SPY)
- Multi-timeframe fusion in MVP (H1 only; daily fusion = optional extension after Stage 2)
- Reinforcement learning approaches
- Order Block / BOS / CHoCH / Liquidity / S&D detection in MVP (FVG only — extensions only if time permits)
- Production deployment beyond a self-contained demo (Plotly + script)
- F1 threshold commitments before baseline run
- Manual hand-labelling of >50 candles (used only for label-validation spot-check)
- Custom xLSTM implementation from scratch (use official `xlstm` package; if unstable, fall back to community PyTorch port — see Risk Register)

**Resolved ambiguities:**
- Master plan filename location: STANDARDS mandates dated subfolder, so file lives at `.nb-suite/plan/8-May-26/master-phasing.md` not `.nb-suite/plan/master-phasing.md`.
- "Research" in this plan = `/nb:research` skill artifacts, not academic literature reviews — though every research artifact will cite literature.
- Each research stream gets its own artifact file. Compound topics split into sub-streams when dependencies diverge.
- Status update 1 (May 17) target = end of CRISP-DM Phase 3 (Data Preparation) + LSTM baseline trained. Status update 2 (June 7) target = end of Phase 5 (Evaluation) for at least 2 of 4 architectures. Final (June 17/20) = full Phase 6 with deployable demo.

---

## CRISP-DM phase mapping

| # | CRISP-DM phase | Window | Status update gate |
|---|----------------|--------|--------------------|
| 1 | Business Understanding | 8 May – 9 May (1 day) | (already done in `idea.md`; consolidate only) |
| 2 | Data Understanding | 9 May – 11 May (3 days) | — |
| 3 | Data Preparation | 11 May – 16 May (6 days) | **Status update 1 — 17 May** |
| 4 | Modelling | 17 May – 4 June (19 days) | — |
| 5 | Evaluation | 4 June – 7 June (4 days) | **Status update 2 — 7 June** |
| 6 | Deployment (demo + report) | 7 June – 20 June (13 days) | **Final demo 17 June; Final delivery 20 June** |

Phases iterate per CRISP-DM. Evaluation findings may force return to Data Preparation (e.g., bad labels) or Modelling (e.g., wrong architecture).

---

## Sequential phase — foundations first

### Foundation 1: Phase 1 — Business Understanding (consolidation)

**Model:** haiku (mostly already done).
**Files:**
- Create: `.nb-suite/research/8-May-26/business-understanding.md`

**What it does:** Consolidates `docs/idea.md`, `docs/assignment.md`, ethics, success criteria, grading criteria, and the explicit "what would make this fail" into a single CRISP-DM Phase 1 artifact. Confirms problem framing is structure detection, not price prediction. States acceptance gates per status update.

**Research first:** Not needed — material already gathered.

**Spawn:** None. Direct write.

**Exit gate:** Single document covers: (a) business question, (b) ML question reframing, (c) success criteria for each status update, (d) ethics + risk, (e) explicit non-goals.

---

### Foundation 2: Phase 2 — Data Understanding

**Model:** sonnet (research orchestration).
**Files (research artifacts to produce):**
- Create: `.nb-suite/research/9-May-26/data-source-validation.md`
- Create: `.nb-suite/research/9-May-26/smc-library-validation.md`
- Create: `.nb-suite/research/9-May-26/labeling-strategy.md`
- Create: `.nb-suite/research/10-May-26/timeframe-decision.md`
- Create: `.nb-suite/research/10-May-26/data-quality-baseline.md`

**What it does:** Answers all open data questions before any preparation code is written. Each research artifact is independently scoped, individually citable, and signed off before moving on. Quality gate: every recommendation has High/Medium/Low confidence and a "What I couldn't verify" section per STANDARDS.md §Shared skill design principles.

**Spawn (parallel where independent):**

| Spawn | Agent | Model | Brief (what to investigate) | Confirms / falsifies |
|-------|-------|-------|-----------------------------|----------------------|
| R1 | `@nb-research` | sonnet | yfinance reliability, rate limits, dividend/split adjustment behaviour for SPY H1 2018+. Compare to alternatives (Polygon, Alpaca free tier, IBKR). Is hourly data actually returned for full history or windowed? | Decision: yfinance vs alternative source. Confidence + fallback. |
| R2 | `@nb-research` | sonnet | `smart-money-concepts` library (Attridge): which structures it computes, parameters, known issues, lookahead behaviour. Validate against ICT textbook FVG definition. | Confirms library is fit for FVG labelling, OR identifies need for custom implementation. |
| R3 | `@nb-research` | **opus** | Labeling strategy: weak supervision (Snorkel paradigm) vs synthetic vs manual. For FVG specifically, what's the noise floor of rule-based labels? Should we use multiple weak labellers + Snorkel-style aggregation? Should we hand-label a small validation set for ground-truth comparison? | Decision: programmatic-only (current plan) vs programmatic + small hand-labelled gold set. Confidence rating. |
| R4 | `@nb-research` | sonnet | Timeframe decision: H1 vs 30m vs 4H vs multi-timeframe. SMC practitioner consensus check, sample-size implications for DL, label density, noise floor. | Confirms H1 single-frame for MVP. Documents conditions for adding daily fusion later. |
| R5 | `@nb-research` | haiku | Data quality baseline checks: zero-volume handling, holiday/weekend gaps for SPY, market open/close hour boundary effects, DST transitions. | Concrete cleaning rules per gap type. |

**Dependencies:** R1 must complete before R5 (need source confirmed). R2 must complete before R3 (R3 builds on what R2 finds the library can do). R4 independent. Run R1 + R2 + R4 in parallel; then R3 + R5.

**Escalation:** If R2 reveals smart-money-concepts has lookahead leakage, R3 must include "custom rule re-implementation" as a sub-stream. If R1 reveals yfinance H1 history is windowed (e.g., only last 730 days), pivot to Polygon free tier or pre-downloaded dataset.

**Exit gate:** All five artifacts complete. Data source decided. Labeling strategy decided. Timeframe decided. Quality cleaning rules drafted. No "TBD" anywhere.

---

### Foundation 3: Phase 3 — Data Preparation (sub-plan + build)

**Model:** sonnet.
**Files (sub-plan + build artifacts to produce):**
- Create: `.nb-suite/plan/11-May-26/phase3-data-preparation.md` (sub-plan via `/nb:plan`)
- Create: `src/data/download.py`, `src/data/clean.py`, `src/data/label.py`, `src/data/normalize.py`, `src/data/window.py`, `src/data/split.py`
- Create: `tests/test_data_*.py` per module
- Create: `notebooks/01-data-understanding.ipynb` (status update 1 artefact)
- Create: `.nb-suite/build/12-May-26/phase3-data-prep.md`

**What it does:** Builds full data pipeline from yfinance download to (60×5)-tensor windows + ternary FVG labels with temporal train/val/test split. Hand-labelled validation set (~50 candles) compared to rule labels for noise estimate. Notebook documents Phase 2 + 3 in CRISP-DM format with figures and observations — this becomes the Status Update 1 deliverable on 17 May.

**Research first:** Already done in Foundation 2 — read those artifacts before planning.

**Sub-plan triggers:**
- `/nb:plan` standalone for Phase 3 once Foundation 2 research artifacts are signed off. Sub-plan defines exact module contracts, test coverage, integration point with Phase 4.
- Sub-plan must answer: (a) what does the windowed dataset emit (tensor shape, label encoding, mask for warm-up candles), (b) split boundaries (exact dates), (c) normalisation scope (per-window vs per-split, what statistics persisted for inference).

**Spawn:** `/nb:plan` → `/nb:build` → auto-spawned `@nb-review` + `@nb-test` per STANDARDS.

**Escalation:** If holdout set has fewer than ~500 positive FVG examples after splitting, pivot to (a) longer history (try 2015+), or (b) augmentation via window stride <1, or (c) merge bullish + bearish FVG into binary target.

**Exit gate:** Notebook runs end-to-end. Train/val/test tensors saved. Status Update 1 ready for peer review. Hand-label noise estimate documented.

---

### Foundation 4: Phase 4 — Modelling (sub-plan + build, 4 architectures)

**Model:** sonnet (orchestrator); haiku/sonnet per workstream.
**Files (research + sub-plan + build):**
- Create: `.nb-suite/research/17-May-26/architecture-comparison.md`
- Create: `.nb-suite/research/17-May-26/training-protocol.md`
- Create: `.nb-suite/plan/19-May-26/phase4-modelling.md`
- Create: `src/models/lstm.py`, `src/models/cnn_lstm.py`, `src/models/xlstm.py`, `src/models/transformer.py`
- Create: `src/training/train.py`, `src/training/loss.py` (weighted CE, focal as alt), `src/training/scheduler.py`, `src/training/early_stop.py`
- Create: `tests/test_models_*.py`
- Create: `.nb-suite/build/D-Mon-YY/phase4-*.md` (one log per architecture)

**Research first:** Two streams before sub-plan:

| Spawn | Agent | Model | Brief | Confirms / falsifies |
|-------|-------|-------|-------|----------------------|
| R6 | `@nb-research` | **opus** | Architecture selection for sequence labelling on financial OHLCV with extreme class imbalance. LSTM vs CNN-LSTM vs xLSTM vs Transformer vs hybrid (CNN + xLSTM, CNN + Transformer). What does literature show on similar tasks (candlestick recognition, anomaly detection, chart pattern detection)? Multi-perspective fan-out per nb-research Deep tier. | Ranked architecture list. Recommended start = LSTM baseline → CNN-LSTM → xLSTM. Justification for skipping or including hybrids. |
| R7 | `@nb-research` | sonnet | Training protocol for class-imbalanced sequence labelling: weighted CE vs focal loss vs balanced sampling vs SMOTE on time series (warning); learning rate schedule (cosine, OneCycle); early stopping signal (F1 minority not val loss); seed handling for reproducibility. | Concrete training recipe. Loss function choice per architecture. Reproducibility checklist. |

**Sub-plan triggers:**
- `/nb:plan` for Phase 4 reads R6 + R7. Plans LSTM baseline first, then CNN-LSTM, xLSTM, Transformer in series (each runs as its own `/nb:build` cycle so we don't carry mistakes forward).
- For xLSTM specifically: sub-plan must decide official `xlstm` package vs community port. Validate package install on Apple Silicon (M4 Pro) before committing — known to require careful CUDA / MPS handling.

**Build sequence (each = own `/nb:build`):**
1. LSTM baseline (target: any positive minority-F1 above naive).
2. CNN-LSTM (target: beat LSTM minority-F1).
3. xLSTM (target: beat CNN-LSTM OR document why it doesn't on this scale).
4. Transformer / TFT (time-permitting; document if skipped).

Each cycle: `/nb:plan` (architecture-specific if R6 reveals divergence) → `/nb:build` → `@nb-review` → `@nb-test` → log results. Reuse Phase 3 dataset unchanged.

**Escalation:** If LSTM baseline produces zero positive F1, problem is upstream (labels or windowing) — return to Phase 3, do not advance. If xLSTM training is unstable (loss NaN, gradient explosion — known issue with sLSTM exponential gates), fall back to mLSTM-only configuration and document.

**Exit gate:** At least 2 architectures fully evaluated on holdout by end of Phase 4. Both have reproducible training scripts. Both have logged hyperparameters and seeds.

---

### Foundation 5: Phase 5 — Evaluation

**Model:** sonnet.
**Files:**
- Create: `.nb-suite/plan/4-Jun-26/phase5-evaluation.md` (sub-plan)
- Create: `src/eval/metrics.py`, `src/eval/ablation.py`, `src/eval/qualitative.py`
- Create: `notebooks/02-evaluation.ipynb` (Status Update 2 deliverable)
- Create: `.nb-suite/build/D-Mon-YY/phase5-eval.md`

**What it does:** Compares all built architectures on identical 2023–2024 holdout. Three evaluation layers:

1. **Detection accuracy** — does model find FVG zones? Binary collapse of ternary: `{bull, bear} → positive`, `none → negative`. Metrics: precision, recall, F1, ROC-AUC. Answers "how accurately are SMC concepts plotted".
2. **Bias accuracy** — given a detected FVG, is direction right? Conditional metric on positive predictions only: bull-vs-bear F1, confusion matrix on `{bull, bear}` subset. Answers "how accurately is bias predicted".
3. **Per-class F1** (full ternary) — primary headline metric. Class-weighted and macro-averaged. Sanity: per-class F1 should not collapse to majority.

Secondary: calibration (Brier score), agreement-with-rule-engine analysis, zone-overlap qualitative case study (≥3 examples where model differs from custom detector with explanation). Compares against naive baseline (always-majority) and custom-detector baseline (deterministic rule on test set, gives label-noise ceiling). Honest limitations section per STANDARDS.

Bias presentation in final demo (Phase 6): each detected FVG rendered with colour + tooltip ("Bullish FVG — buy bias zone" / "Bearish FVG — sell bias zone") + confidence score. NOT a trade recommendation. See `.nb-suite/explanations/scope.md`.

**Research first:** 

| Spawn | Agent | Model | Brief |
|-------|-------|-------|-------|
| R8 | `@nb-research` | sonnet | Evaluation rigor for weak-supervised models: how to evaluate when labels are noisy? Confidence intervals via temporal bootstrap. Statistical significance of F1 differences when test set has ~500 positives. |

**Sub-plan triggers:** `/nb:plan` for Phase 5. Defines exact metrics, plots, table layout for Status Update 2.

**Iteration loop:** If evaluation reveals systematic failure mode, return to Phase 4 (model fix) or Phase 3 (data fix). Budget for one iteration only — Phase 4/5 budget is 23 days, return-trip eats ~5 days.

**Exit gate:** Status Update 2 notebook complete. All architectures benchmarked. Honest limitations stated. Ready for tutor review 7 June.

---

### Foundation 6: Phase 6 — Deployment (demo + report)

**Model:** sonnet.
**Files:**
- Create: `.nb-suite/plan/8-Jun-26/phase6-deployment.md` (sub-plan)
- Create: `src/viz/chart.py`, `src/inference/predict.py`
- Create: `notebooks/03-final.ipynb` (final delivery notebook)
- Create: `app/demo.py` (Streamlit or Plotly Dash standalone demo for live presentation)
- Create: `README.md` (top-level project README)
- Create: `docs/architecture.md` via `@nb-docs`
- Create: `.nb-suite/build/D-Mon-YY/phase6-deploy.md`

**What it does:** Produces the final demo and report. Plotly chart with rule zones vs model zones overlay. Standalone runnable demo (single command). Final notebook follows full CRISP-DM with figures, captions, observations per `nb-notebook` standard. Final report exported to HTML. `@nb-humanize` run on all prose per CLAUDE.md.

**Spawn:** `@nb-notebook` for final notebook polish, `@nb-humanize` after, `@nb-docs` for architecture doc.

**Escalation:** If demo unstable on presentation laptop, fall back to pre-rendered static charts + screen recording of interactive run.

**Exit gate:** 17 June live demo runs without manual intervention. 20 June final delivery: notebook + code + README + architecture doc + saved trained weights + sample data.

---

## Parallel-friendly research streams (start NOW, 8 May)

These can run today in parallel — none depend on later artifacts:

| Stream | Spawn target | When |
|--------|--------------|------|
| R1 yfinance validation | `@nb-research` | Today |
| R2 smart-money-concepts library audit | `@nb-research` | Today |
| R4 timeframe decision | `@nb-research` | Today |

R3 (labeling strategy) and R5 (data quality) wait for R1+R2.
R6 + R7 (architecture + training) wait until end of Phase 3 — only useful once data shape is finalised, but R6 can start in parallel with Phase 3 build since architecture comparison is data-shape-agnostic for the first pass.

---

## Test coverage

**Code-side (TDD per STANDARDS):**

| Component | What to test | Key failure modes | Edge cases |
|-----------|--------------|-------------------|-----------|
| `data/download.py` | Returns dataframe with expected schema; handles missing data; respects date range | Network failure, empty response, schema drift | Holidays, half-days, DST transitions |
| `data/label.py` | Labels match smart-money-concepts output bit-for-bit; no lookahead in custom code | Off-by-one, label leakage past candle boundary | First N candles (warm-up), last N candles (unconfirmed FVGs) |
| `data/window.py` | Windows are exactly 60×5; final candle of window has correct label index; no shuffling | Misaligned label, off-by-one window | Window straddling holiday gap |
| `data/split.py` | Train end < val start < test start strictly; no row appears in multiple splits | Time leakage | Boundary candles |
| `models/*.py` | Forward pass produces (batch, num_classes) with finite logits on dummy input | NaN, shape mismatch | Min-length input, max-length input |
| `training/train.py` | Loss decreases on tiny overfit dataset; weighted CE applies correct weights | Wrong class weights, gradient explosion | Single-class batch |
| `eval/metrics.py` | F1 calculation matches sklearn; bootstrap CI converges; binary-detection F1 != per-class F1 unless single positive class; bias-conditional F1 returns NaN when no positives predicted | Wrong averaging, label encoding mismatch, bias metric leaks into detection metric | Empty positive class, all-bull predictions, all-bear predictions |

**ML-side (falsification per STANDARDS §AI/ML):**
- Held-out F1 on 2023–2024 (never touched during Phase 3/4)
- Hand-labelled gold set comparison: rule labels vs human labels
- Sanity check: shuffle training labels — model F1 should drop to ~chance
- Sanity check: train on first half, test on second half — confirm no time leakage

**What is NOT tested:** end-to-end financial profitability (out of scope), model behaviour on assets other than SPY, model behaviour on intra-day timeframes outside H1.

---

## Edge cases and gotchas

- ~~**smart-money-concepts library may use future candles to confirm FVGs.**~~ **CONFIRMED via R2.** Library uses `shift(-1)`. Phase 3 build = custom detector. Reference impl in R2 artifact. Label index convention: `i+1` (first bar where 3-candle pattern is closed).
- **Rolling normalisation per STANDARDS:** must be computed *causally* — using only candles up to and including current. A single `df.rolling().mean()` call without `min_periods` and proper offset will leak future info subtly.
- **Class imbalance with windowing:** if FVG label is "candle is part of an FVG zone", positive class is sparser than 5–10% per *candle* but denser per *window* (any candle in window = positive window). Decide labeling unit explicitly in Phase 3 sub-plan.
- **xLSTM on Apple Silicon:** official package may require CUDA. Confirm MPS support or have CPU fallback before Phase 4 begins.
- **Bootstrap CI with temporal data:** standard bootstrap breaks i.i.d. assumption. Use block-bootstrap with block size ≥ window length.
- **yfinance H1 history limit:** common gotcha — yfinance silently caps intra-day history at 730 days. Foundation 2 R1 must verify full 2018+ retrieval; if capped, source pivot needed.
- **`smart-money-concepts` package install:** may pin old pandas/numpy. Test compatibility with PyTorch 2.x stack early — Phase 3 R5 territory.

---

## Risk register (Deep)

| Risk | Likelihood | Blast radius | Reversibility | Mitigation |
|------|------------|--------------|---------------|------------|
| ~~yfinance H1 history capped at 730 days~~ **CONFIRMED 8-May-26 — pivoted to Alpaca** | (resolved) | (resolved) | (resolved) | Source = Alpaca free tier, paper account, `adjustment="raw"` |
| Alpaca free tier rate-limit / outage during bulk download | Low | Phase 3 (data layer) | Easy — chunk requests, retry | 200 calls/min plenty; one-time bulk; cache CSV |
| Alpaca free tier policy change (paid tier required for historical) | Low | Phase 3 | Medium — pivot to alternative or paid | Pull data day 1 of Phase 3; cache to CSV; once cached, source change is moot |
| ~~smart-money-concepts library has lookahead leakage~~ **CONFIRMED 8-May-26 — custom detector required** | (resolved) | (resolved) | (resolved) | Custom `src/data/label.py` — vectorised FVG detector, label placed at N+1 |
| Custom FVG detector has subtle off-by-one bug | Medium | Phase 3 labels + all downstream | Medium — caught by tests | Mandatory pytest fixture: 9-candle synthetic with known FVG, assert label at N+1 not N |
| FVG positive examples too few in 2023-2024 holdout | Medium | Phase 4-5 (statistical power) | Medium — extend history or rebalance | Compute holdout positive count in Phase 3 *before* Phase 4 begins |
| xLSTM library incompatible with Apple Silicon | Medium | Phase 4 Workstream 3 | Easy (skip xLSTM, document) | Phase 4 sub-plan installs xlstm package on day 1, smoke-tests |
| Rule-based labels too noisy for DL ceiling | Medium | Phase 4-5 results | Hard — requires hand-labelling at scale | Hand-label 50-candle gold set in Phase 3; quantify noise floor before training |
| Status Update 1 misses 17 May | Medium | Peer review feedback lost | Hard — fixed deadline | Phase 3 hard cap = 16 May. If R6/R7 blocking, defer them to post-Update-1. |
| Final demo crashes during 17 June presentation | Low | Demo grade | Easy — pre-record fallback | Pre-render charts and record demo video by 16 June end-of-day |
| Scope creep into OB/BOS/CHoCH detection | High (self-inflicted) | Time budget | Easy if disciplined | OUT OF SCOPE in this plan; revisit only after FVG MVP shipped |
| Apple Silicon MPS support inconsistent across PyTorch ops | Medium | Training time / correctness | Medium — fall back to CPU | Smoke-test all model forward + backward on MPS at Phase 4 start; CPU acceptable for ~16k candles |
| Programmatic labels collapse to "always none" majority on full pipeline | Low | Phase 3-4 | Easy — re-tune library params | Phase 3 build verifies positive rate before training |

---

## Pre-mortem (Deep)

It's 20 June 2026 and the project failed or scored poorly. Most likely causes ranked by probability:

1. **Label noise too high — model F1 capped near rule-engine F1, no novelty demonstrated.** We trusted `smart-money-concepts` without validating against hand-labels. Mitigation: R2 + R3 + Phase 3 hand-label gold set. This is the most likely failure and gets the most upfront research.
2. **Time leakage in normalisation or labelling.** Subtle bug — model "looks great" until tutor asks how `rolling().mean()` is computed, and we realise it averaged past + future. Mitigation: Test in `tests/test_data_window.py` that asserts no future indices touched. CRITICAL test.
3. **xLSTM gave nothing extra.** Spent days on it; baseline LSTM was actually best. Result: report becomes "xLSTM didn't help on this scale, here's why" — that's still a valid academic result. Mitigation: hard time-box xLSTM workstream to 5 days; document negative result honestly.
4. **Demo crashes live.** Plotly bug, library version drift between dev and presentation laptop. Mitigation: lock versions, pre-render fallback.
5. **Scope creep.** Tried to add OB/BOS partway through Phase 4. Burned 10 days, FVG path suffered. Mitigation: this plan's "out of scope" section is *binding*.

---

## What to flag if found

- Any spawned `@nb-research` returns confidence "Low" — bring back to user before proceeding to next phase.
- Any phase exit gate fails — stop, write partial build log per STANDARDS §Build failure protocol, surface to user.
- Any deadline slip estimate >2 days — surface immediately, don't silently absorb.
- Mid-build factual unknown that R1–R8 didn't cover — spawn ad-hoc `@nb-research` per STANDARDS §Cross-skill delegation, log new artifact, link from this master plan.
- Any time the assumption "FVG is well-defined and labellable" cracks — flag immediately, this is a research project blocker.

---

## Extensibility — what's reusable vs what changes per concept

Architecture is built so adding a new SMC concept = swap one module + retrain. NOT rebuild from scratch.

**Reusable (built once, used for every SMC concept):**

| Component | Path | Why concept-agnostic |
|-----------|------|----------------------|
| Data download + clean | `src/data/download.py`, `src/data/clean.py` | Same OHLCV bars regardless of what we label |
| Normalisation | `src/data/normalize.py` | Causal rolling stats, no label dependency |
| Sliding window dataset | `src/data/window.py` | (60, 5) tensor shape independent of label semantics |
| Train/val/test split | `src/data/split.py` | Temporal cuts, label-blind |
| Model classes | `src/models/lstm.py`, `cnn_lstm.py`, `xlstm.py`, `transformer.py` | Input = (B, 60, 5), output = (B, num_classes). Just changes `num_classes` arg |
| Training loop | `src/training/train.py` | Generic weighted CE classifier |
| Evaluation harness | `src/eval/metrics.py` | Detection-F1 + bias-F1 + per-class-F1 generic over class count |
| Plotly viz | `src/viz/chart.py` | Renders any (start_idx, end_idx, top, bottom, type) tuple as a coloured zone |

**Concept-specific (one file per SMC concept — the only thing that changes):**

```
src/data/labels/
  __init__.py          # registry: name -> labeller class
  base.py              # ABC: label(df) -> Series of int class indices, label_index_offset, num_classes
  fvg.py               # FVG labeller (Phase 3 — built first)
  ob.py                # Order Block labeller (Phase 4.5 stretch, or post-final)
  bos.py               # BOS labeller (post-final)
  choch.py             # CHoCH labeller (post-final)
  liquidity.py         # Liquidity Pool labeller (post-final)
  sd.py                # Supply & Demand labeller (post-final)
```

Each labeller subclass implements:
- `label(df: DataFrame) -> Series[int]` — per-candle class indices, no lookahead
- `label_index_offset: int` — how many bars after pattern close before label is knowable (FVG = 1)
- `num_classes: int` — for model output head sizing (FVG = 3)
- `class_names: list[str]` — for viz tooltips (FVG = `["none", "bullish", "bearish"]`)

**Per-concept extension cost (after FVG MVP shipped):**
1. Write new `src/data/labels/<concept>.py` (~1 day, includes hand-label gold set)
2. Validate on 9-candle synthetic fixture (~30 min)
3. Hand-label kappa gate per R3 protocol (~2 hours)
4. Retrain best architecture (~few hours, reuse training script)
5. Add to evaluation table (~30 min, harness is generic)
6. Add tooltip strings to viz (~10 min)

Total: **~3 days per concept** assuming pipeline is solid. First concept (FVG) is slow because pipeline + labellers built simultaneously; subsequent concepts only build the labeller.

**Architectural decision required for Phase 3 sub-plan:** label registry pattern (above) MUST be implemented in Phase 3 even though only FVG ships. Builds the seam day 1. Refactoring later = expensive.

---

## SMC concept coverage roadmap (honest scoping)

User asked if all SMC concepts are planned through 20 June. **They are not — and they shouldn't be.** This table makes scope explicit.

| SMC Concept | MVP status | Extension cost | Realistic before 20 June? |
|-------------|------------|----------------|---------------------------|
| **FVG (Fair Value Gap)** | ✅ Core MVP — full pipeline | (built) | YES — locked |
| **Order Block (OB)** | ⚠️ Stretch goal | ~3–4 days (4+ competing definitions, must pick one) | Only if FVG ships by 28 May AND Phase 4/5 ahead of schedule |
| **BOS (Break of Structure)** | ❌ Post-final extension | ~3 days | NO — listed in final report as future work |
| **CHoCH (Change of Character)** | ❌ Post-final extension | ~3 days | NO — listed in final report as future work |
| **Liquidity Pools** | ❌ Post-final extension | ~3 days (definition is ambiguous in retail SMC) | NO — listed in final report as future work |
| **Supply & Demand zones** | ❌ Post-final extension | ~3 days (overlaps with OB heavily) | NO — listed in final report as future work |

**Why this scoping is correct:**
1. Each concept = own labelling rule + own noise floor + own hand-label gold set + own retraining + own evaluation. Pipeline reuse is real but not free.
2. Doing all 6 poorly < doing 1 well + 1 stretch.
3. Final report frames as "FVG zone detection — pattern-recognition primitive of SMC, generalisable to other structures via the same pipeline (demonstrated for OB if time)". Defensible academic position.
4. Five-concept claim with shallow execution = grading risk (peer review on 17 May will spot it).

**Decision gate for OB extension:** Re-evaluate **28 May** at end of LSTM + CNN-LSTM workstream. If both architectures finished + at least one has positive holdout F1, spawn `@nb-plan` for OB extension as a Phase 4.5 mini-cycle. If behind schedule, lock to FVG-only and document OB/BOS/CHoCH/Liquidity/S&D as concrete future work in final report.

**If you want all 6 covered:** project needs ≈30 extra days. Realistic scope expansion = post-graduation continuation, not within the Innovation Challenge deadline.

---

## Execution order — what to do next

The user reviewed this master plan. Once approved, execution order is:

1. **Today (8 May):** Spawn R1, R2, R4 in parallel (`@nb-research` × 3).
2. **9 May:** Read R1+R2 results. Spawn R3 + R5.
3. **10 May:** Read all Phase 2 artifacts. Run `/nb:plan` for Phase 3 sub-plan.
4. **11–16 May:** `/nb:build` Phase 3. Notebook + tests + data pipeline.
5. **17 May:** Status Update 1 submitted. Peer review.
6. **17 May parallel:** Spawn R6 + R7 (architecture + training research).
7. **19–24 May:** `/nb:plan` Phase 4. `/nb:build` LSTM baseline.
8. **25 May – 4 June:** Build CNN-LSTM, xLSTM, (Transformer if time).
9. **4–7 June:** R8, Phase 5 sub-plan, evaluation build, Status Update 2.
10. **8–17 June:** Phase 6. Demo, final notebook, README, architecture doc.
11. **17 June:** Live demo.
12. **17–20 June:** Polish, humanize prose, submit.

Buffer days: ~3 across the schedule. Use them, don't pretend they don't exist.

---

## Standalone artifacts produced by this plan

| Artifact | Path |
|----------|------|
| Master plan (this) | `.nb-suite/plan/8-May-26/master-phasing.md` |
| Phase 1 doc | `.nb-suite/research/8-May-26/business-understanding.md` |
| Phase 2 research × 5 | `.nb-suite/research/9-May-26/*.md`, `.nb-suite/research/10-May-26/*.md` |
| Phase 3 sub-plan | `.nb-suite/plan/11-May-26/phase3-data-preparation.md` |
| Phase 3 build log | `.nb-suite/build/12-May-26/phase3-data-prep.md` |
| Phase 4 research × 2 | `.nb-suite/research/17-May-26/*.md` |
| Phase 4 sub-plan | `.nb-suite/plan/19-May-26/phase4-modelling.md` |
| Phase 4 build logs × 4 | `.nb-suite/build/D-Mon-YY/phase4-*.md` |
| Phase 5 research × 1 | `.nb-suite/research/4-Jun-26/eval-rigor.md` |
| Phase 5 sub-plan + build | `.nb-suite/plan/4-Jun-26/phase5-evaluation.md`, `.nb-suite/build/D-Mon-YY/phase5-*.md` |
| Phase 6 sub-plan + build | `.nb-suite/plan/8-Jun-26/phase6-deployment.md`, `.nb-suite/build/D-Mon-YY/phase6-*.md` |
| Final notebook + report | `notebooks/03-final.ipynb`, `docs/architecture.md`, `README.md` |

---

## Confidence

**High** that CRISP-DM phasing, deadline mapping, and research-stream decomposition are correct.
**Medium** on time estimates — Phase 4 is the riskiest because xLSTM time and architecture-comparison depth are unknown until R6 returns.
**Medium** on label-strategy outcome — depends entirely on R2 + R3 findings; may force us to add a hand-labelling subproject.

## What I couldn't verify

- Exact yfinance H1 history limit for SPY 2018-present (R1 confirms).
- Whether smart-money-concepts library uses lookahead in its FVG implementation (R2 confirms).
- Whether xLSTM official package supports Apple Silicon natively (Phase 4 day-1 smoke test confirms).
- Whether ~16k candles is sufficient training volume for xLSTM (R6 + Phase 4 baseline confirm).
- Whether peer review on 17 May is mandatory deliverable or optional checkpoint — assume mandatory.
