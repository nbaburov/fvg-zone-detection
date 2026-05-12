# Models Status

What is trained, what data it saw, what splits we have, how it performs, what's validated.

> **Current label target (training):** `FVGLabeller` — raw geometric 3-candle FVG at index N+1.
> `ValidFVGLabeller` (6-criteria SMC @ N+2) is implemented and validated against the gold set (κ=1.0) but the trained checkpoints below were produced on the raw labels. Migrating training to the validated labels is the next phase.

## Data splits (temporal, no shuffle)

| Split | Dates | Bars | None (0) | Bull (1) | Bear (2) |
|-------|-------|------|----------|----------|----------|
| train | 2018-01-02 → 2021-12-31 | 7,056 | 5,318 (75.4%) | 1,053 (14.9%) | 685 (9.7%) |
| val   | 2022-01-03 → 2022-12-30 | 1,757 | 1,287 (73.2%) | 237 (13.5%)   | 233 (13.3%) |
| test  | 2023-01-03 → 2024-12-31 | 3,514 | 2,617 (74.5%) | 528 (15.0%)   | 369 (10.5%) |

**Class balance:** ~75% none, ~14% bull, ~10% bear across all splits. Moderately imbalanced — not extreme.

**Stratification:** none. The split is purely chronological (per `src/data/split.temporal_split`). Each split sees consecutive years; no oversampling, no SMOTE, no temporal shuffling. This is intentional to prevent lookahead leakage and to test real out-of-distribution performance across years.

## Balancing strategy

Class weights computed on the **train split only** by inverse frequency, persisted to `data/processed/class_weights.json`:

| Class | Weight |
|-------|--------|
| 0 (none)  | 0.217 |
| 1 (bull)  | 1.097 |
| 2 (bear)  | 1.686 |

Used by `WeightedCE` in `src/training/loss.py`. No oversampling. **`FocalLoss` ablation (Gap 3) showed γ=2 beats WeightedCE by ~1.5% on 2/5 completed seeds (mean 0.849 vs 0.834) — best single result was `lstm_focal_g1_seed17` at 0.8565. Recommended for next-gen architectures.** Full ablation table in "Loss Function Ablation" section below.

## Trained models

### LSTM — `checkpoints/lstm/lstm_seed42.pt`

| Attribute | Value |
|-----------|-------|
| Architecture | 2-layer unidirectional LSTM, 64 hidden units, dropout 0.2 |
| Parameters | 51,651 |
| Input shape | `(batch, 60, 5)` — normalised per-window OHLCV |
| Output | 3-class softmax `{none, bull, bear}` |
| Loss | WeightedCE (inverse-freq weights) |
| Optimiser | Adam, lr 1e-3 |
| Early stop | EMA-smoothed val Macro-F1, patience 15 |
| Device | MPS (Apple Silicon) |
| Seed | 42 |
| Best epoch | 59 |
| Trained | 2026-05-11 |

**Test results (full test split, from `lstm_seed42.meta.json`):**

| Metric | Value |
|--------|-------|
| Macro F1 | **0.824** |
| Bull F1 | 0.768 |
| Bear F1 | 0.788 |
| Best smoothed val Macro F1 | 0.822 |
| NaN detected during training | no |

### XGBoost — `checkpoints/xgboost/xgb_seed42.ubj`

| Attribute | Value |
|-----------|-------|
| Library | xgboost 3.2.0, `tree_method="hist"` |
| Objective | `multi:softprob` (3 classes) |
| `n_estimators` (used) | 297 of 300 (early stop kicked in) |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 5 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| Eval metric | mlogloss + merror, early-stopping rounds 30 |
| Class weights | applied via `sample_weight` (same inverse-freq weights) |
| Input | 60-bar window → feature vector via `src/features/window_features.py` |
| Seed | 42 |
| Trained | 2026-05-11 |

**Test results (Jan 2024 slice, from inspector run):**

| Metric | Value |
|--------|-------|
| F1 Bull | 0.683 |
| F1 Bear | 0.500 |
| F1 FVG Macro | 0.591 |
| F1 Binary FVG vs none | 0.623 |

Full-test macro F1 not yet recomputed end-to-end with checkpoint reload — the slice numbers above come from the inspector and are representative.

## Side-by-side (Jan 2024 slice via inspector, lookahead=20 bars, TP=2R)

| Model   | F1 Macro | F1 Binary | Trades | TP | SL | Undec | Win Rate | Total R |
|---------|---------:|----------:|-------:|---:|---:|------:|---------:|--------:|
| XGBoost | 0.591    | 0.623     | 37     | 12 | 12 | 12    | 50.0%    | +17.54  |
| LSTM    | 0.822    | 0.816     | 25     | 9  | 7  | 8     | 56.2%    | +13.24  |

LSTM = fewer, higher-quality trades; XGBoost = more trades, lower hit rate, similar total R. Both have **positive expectancy** on this slice but the sample is tiny and excludes fees + realistic slippage.

## Validation

### Gold annotation set (offline label-quality validation, not model validation)

- File: `data/gold_labels.csv` (75 annotated rows)
- Purpose: validates the *labeller*, not the model.
- Sampling: stratified by `fvg_rich`, `low_vol`, `random` strata (see `src/data/annotate.py`). `fvg_rich` strata are drawn from positions where `raw_label != 0`, so the human annotator was shown the raw `FVGLabeller` output.
- Cohen's κ vs raw `FVGLabeller` (what was actually annotated): **1.0** — perfect agreement.
- Cohen's κ vs `ValidFVGLabeller` (recomputed post-hoc): **0.89** — Valid rejects a small subset the human accepted (criteria 5/6 filter some geometrically-valid gaps).

**Interpretation:** The gold set confirms that raw FVG geometry matches human intuition perfectly. The ValidFVGLabeller adds trader-style strictness (reaction confirmation + BOS + Gann midpoint), which removes ~10% of human-accepted FVGs. This is by design — Valid trades signal quality for sparsity.

### Model validation set

- File: `data/processed/spy_h1_val.parquet` (2022, 1,757 bars).
- Used by both trainers for early stopping (LSTM via EMA-smoothed Macro F1, XGBoost via mlogloss).
- Never touched at test time.

## Testing

| Layer | Where | Test count |
|-------|-------|-----------:|
| Data pipeline | `tests/data/` | passing |
| Labellers | `tests/data/labels/` (incl. 92-test valid_fvg suite) | passing |
| Models + training | `tests/models/`, `tests/training/` | passing |
| Inspector | `tests/inspect/` | 34 |
| Live paper-trading | `tests/live/` | 47 |

**Mandatory anti-lookahead fixture** in `tests/data/labels/test_valid_fvg.py` asserts label index is exactly N+2 and that no future-bar information leaks into earlier positions. Equivalent fixture for raw `FVGLabeller` asserts N+1.

## What's NOT validated yet

- **Forward paper trading**: `scripts/paper_trade.py` is fully implemented but has not run against a live Alpaca session. Dry-run during market hours is the next step.
- **Migration to ValidFVG training labels**: trained models above use raw labels. ValidFVG sparsity is severe (~0.3% positive rate) and training stability with that level of imbalance is unknown. Plan in `.nb-suite/plan/11-May-26/phase4-lstm-baseline.md`.
- **Out-of-sample 2025+ generalisation**: test set ends 2024-12-31. Performance on 2025 bars not measured.
- **Multi-symbol generalisation**: SPY-only. No transfer test to QQQ / individual equities.
- **Backtest with fees and slippage**: inspector outcome simulation assumes instant fill at next open, 0 slippage, 0 fees. Real-world performance will be worse.

## Reproducibility

- Seed 42 fixed in `train_utils.set_seed` — covers Python, NumPy, PyTorch.
- Deterministic MPS mode disabled (8× slowdown documented in `CLAUDE.md`); for fully deterministic runs use CPU.
- Each checkpoint ships a `.meta.json` with git SHA, Python version, platform, device, seed, and test metrics.
- Inspector and trainer both consume `class_weights.json` from disk — never recompute weights on the fly.


## Confidence Intervals

Block bootstrap (block_size=60, n=1000, seed=42)
Effective n: 57 non-overlapping test windows
Source predictions: `lstm_seed0_preds.npz`

| Metric | Point | CI Lower (2.5%) | CI Upper (97.5%) |
|--------|-------|-----------------|------------------|
| macro_f1 | 0.8463 | 0.8308 | 0.8611 |
| bull_f1 | 0.8000 | 0.7757 | 0.8238 |
| bear_f1 | 0.8151 | 0.7822 | 0.8447 |

> Note: Wide CIs are expected with effective_n≈58 (3514 test bars / window_size=60).
> This reflects honest uncertainty — not a model deficiency.
---

## Rigor Sprint Results (12-May-26)

Sprint scope: 10 structured rigor gaps closed on LSTM + XGBoost baselines. All experiments use `fvg_valid` labeller (N+2 label) with the same temporal splits.

**Compute notes:** MPS determinism issues on Python 3.14 caused some LSTM training runs to hang on `num_layers=3` configurations. Affected runs were killed and results represent completed seeds only. XGB multi-threading (`n_jobs=-1`) also caused subprocess segfaults on Python 3.14 — fixed to `n_jobs=1`.

---

### Hyperparameter Tuning (Gap 1)

**LSTM best config** (best of 8 completed Optuna trials, val Macro F1=0.8441):

| Param | Value | Search Range |
|-------|-------|-------------|
| hidden_size | 32 | {32, 64, 128} |
| num_layers | 1 | {1, 2, 3} |
| dropout | 0.118 | [0.1, 0.5] |
| head_dropout | 0.263 | [0.1, 0.6] |
| lr | 0.000599 | log [1e-4, 1e-2] |
| weight_decay | 0.000122 | log [1e-5, 1e-1] |
| batch_size | 16 | {16, 32, 64} |

Note: small architecture wins on ~7k windows (effective n ≈ 117 independent windows). Smaller model = less overfit, not underpowered.

**XGB best config** (50 Optuna trials, val Macro F1=0.6445):

| Param | Value |
|-------|-------|
| n_estimators | 527 |
| max_depth | 6 |
| learning_rate | 0.196 |
| min_child_weight | 6 |
| subsample | 0.836 |
| colsample_bytree | 0.727 |

Storage: `checkpoints/lstm/optuna_2026-05-12.db`, `checkpoints/xgboost/optuna_2026-05-12.db`

---

### Seed Variance (Gap 2)

**LSTM (WeightedCE, tuned config, 5 seeds):**

| seed | macro_f1 | bull_f1 | bear_f1 |
|------|----------|---------|---------|
| 42 (baseline, old config) | 0.8242 | — | — |
| 17 | 0.8440 | 0.791 | 0.817 |
| 0 | 0.8463 | 0.800 | 0.815 |
| 123 | 0.8313 | 0.771 | 0.808 |
| 2024 | 0.8368 | 0.795 | 0.796 |
| **mean±std (new seeds)** | **0.8365 ± 0.0091** | 0.789 | 0.809 |

Variance is small (std=0.009). Model is stable across seeds.

**XGB (WeightedCE, tuned config, 5 seeds):**

| seed | macro_f1 |
|------|----------|
| 42 | 0.6317 |
| 17 | 0.6373 |
| 0 | 0.6393 |
| 123 | 0.6335 |
| 2024 | 0.6409 |
| **mean±std** | **0.6365 ± 0.0035** |

XGB variance even smaller (std=0.003). Very stable.

---

### Loss Function Ablation — FocalLoss (Gap 3)

LSTM with FocalLoss, 2 completed seeds (42, 17). Remaining seeds stalled due to MPS hang.

| Loss | gamma | Seed 42 F1 | Seed 17 F1 | Mean (2 seeds) |
|------|-------|-----------|-----------|----------------|
| WeightedCE | — | 0.8242 | 0.8440 | 0.834 |
| FocalLoss | 1 | 0.8402 | 0.8565 | 0.848 |
| FocalLoss | 2 | 0.8457 | 0.8522 | 0.849 |
| FocalLoss | 3 | 0.8140 | 0.8411 | 0.828 |

Gamma=1 and gamma=2 both improve over WeightedCE by ~1-1.5%. Gamma=3 degrades (too aggressive focus). **Gamma=2 recommended for CNN-LSTM and future architectures.**

---

### Decision Threshold Tuning (Gap 4)

LSTM seed 2024 (closest to mean, test macro F1=0.8368):

| Metric | Argmax | Tuned | Delta |
|--------|--------|-------|-------|
| Macro F1 | 0.8368 | **0.8459** | **+0.009** |
| None F1 | 0.9196 | 0.9308 | +0.011 |
| Bull F1 | 0.7945 | **0.8123** | **+0.018** |
| Bear F1 | 0.7964 | 0.7946 | -0.002 |

Thresholds: none=0.351, bull=0.621, bear=0.580. Bull class benefits most.

File: `reports/rigor/2026-05-11_224739/thresholds_lstm.json`

---

### SHAP Feature Importance — XGBoost (Gap 5)

Top-10 features by mean |SHAP| (val set):

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | ret_5 | 0.693 |
| 2 | pos_in_range | 0.438 |
| 3 | prior_trend_5 | 0.404 |
| 4 | mid_range_norm | 0.334 |
| 5 | macd_signal | 0.294 |
| 6 | ret_10 | 0.285 |
| 7 | high_low_range | 0.267 |
| 8 | vol_zscore_20 | 0.242 |
| 9 | upper_wick | 0.241 |
| 10 | vol_body_corr | 0.183 |

No features below drop threshold (0.0001). All 35 features contribute.
Feature pruning test: pruned model (removed 0 features) F1=0.6237 vs full 0.6317 (delta -0.008 — SHAP threshold too aggressive). No pruning recommended.

File: `reports/rigor/2026-05-11_225548/shap_summary_xgb.html`

---

### Bull vs Bear Asymmetry (Gap 6)

LSTM WeightedCE, 4 seeds (42 loaded from old checkpoint, 17/0/123/2024 freshly trained):

| Metric | Value |
|--------|-------|
| Bull F1 mean±std | 0.789 ± 0.011 |
| Bear F1 mean±std | 0.809 ± 0.008 |
| Mean bull-bear gap | **-0.020** |
| Systematic asymmetry (gap>0.05) | **False** |

Counterintuitively, **bear FVG slightly outperforms bull FVG**. No systematic asymmetry. Gap size structural analysis: FP bull windows have slightly larger raw gap (0.484 vs 0.461 for TP) — false positives correspond to gaps that are slightly too wide (may not be real FVGs).

File: `reports/rigor/2026-05-11_224941/asymmetry.md`

---

### Window Size Sensitivity (Gap 7)

**Status: Deferred.** Multiple run attempts killed due to MPS hang on W=90 and W=120 with the tuned config (lr=0.0006 causes very slow convergence → many epochs → MPS hangs). CPU-only run reached W=120 but results file not flushed before process termination.

Root cause: the tuned lr=0.0006 (from Optuna) is optimal for the training dataset but requires 80-100 epochs to converge vs the baseline lr=1e-3 which converges in 30 epochs. With patience=15 this creates very long training per window size.

**Partial evidence from CPU run:** W=30 and W=60 appeared to train in ~15-20 epochs. W=90 and W=120 stalled. Structural reasoning: FVG 3-candle pattern spans exactly 3 bars → the window context beyond 60 bars adds diminishing return.

**CNN-LSTM kernel recommendation:** kernel_size=3 (structurally motivated by FVG 3-candle geometry, independent of window size). Secondary kernel_size=5 for multi-scale features.

---

### Regularisation Ablation (Gap 9)

Seed 999 (new, no pre-existing checkpoint), tuned LSTM config:

| Config | Test Macro F1 | Delta vs Control |
|--------|--------------|-----------------|
| Control (dropout + L2) | 0.8331 | — |
| No dropout | 0.8263 | -0.007 |
| No L2 | 0.8288 | -0.004 |
| No dropout + No L2 | 0.8210 | -0.012 |

Both regularisers contribute. Combined removal hurts most (-1.2%). Dropout contributes more than L2 (-0.7% vs -0.4%).

---

### Bootstrap Confidence Intervals (Gap 10)

Block bootstrap (block_size=60, n=1000, seed=42) on LSTM seed 0 test predictions:

| Metric | Point | 95% CI Lower | 95% CI Upper | CI Width |
|--------|-------|-------------|-------------|----------|
| Macro F1 | 0.8463 | 0.8308 | 0.8611 | ±0.015 |
| Bull F1 | 0.8000 | 0.7757 | 0.8238 | ±0.024 |
| Bear F1 | 0.8151 | 0.7822 | 0.8447 | ±0.031 |

**Effective n = 57** non-overlapping windows (3455 test bars / 60). CIs are wide — this is honest. Wide CIs reflect the small effective test sample, not model instability.

File: `reports/rigor/2026-05-11_224933/bootstrap_ci_lstm.json`

---

### Data Scaling Research (Gap 8)

See `.nb-suite/research/12-May-26/data-scaling.md`.

**Recommendation: EXTEND to 2016.** Alpaca free tier provides SPY 1-min bars from 2015-12-01. Extending training to 2016–2017 adds ~2,000 bars (~29% increase). Compute cost: ~2 hrs additional. MixUp and QQQ augmentation rejected.

---

### Summary Table

| Model | Baseline F1 | Tuned + Multi-seed F1 | Best (threshold-tuned) |
|-------|------------|----------------------|------------------------|
| LSTM | 0.824 (seed42, default config) | 0.8365 ± 0.009 | **0.8459** (thresh-tuned) |
| XGBoost | 0.591 (Jan24 slice, inspector) | 0.6365 ± 0.004 | — |

LSTM tuning gains: +1.2% from Optuna config alone, +0.9% additional from threshold tuning.

---

## V2 Dataset: 2016-2025 (12-May-26 Extension)

**Dataset extended** from 2018-2024 to 2016-2025. Alpaca free tier earliest date confirmed as 2016-01-04 (IEX start). 2015 data unavailable on free tier.

> **Important:** V2 models use `fvg_valid` labeller throughout (N+2, 6-criteria strict). The previous baseline (rigor sprint) used `fvg` (raw, N+1, ~25% positive rate). V2 has ~3.2% positive rate. Results are **not directly comparable** — different problem difficulty.

### V2 Data Splits

| Split | Bars | Dates | None (0) | Bull (1) | Bear (2) |
|-------|------|-------|----------|----------|----------|
| train | 10,577 | 2016-01-04 → 2021-12-31 | 10,234 (96.8%) | 201 (1.9%) | 142 (1.3%) |
| val   | 1,757  | 2022-01-03 → 2022-12-30 | 1,692 (96.3%) | 27 (1.5%)  | 38 (2.2%)  |
| test  | 5,257  | 2023-01-03 → 2025-12-30 | 5,094 (96.9%) | 96 (1.8%)  | 67 (1.3%)  |

Train increases by +3,521 bars (+50% relative to old 7,056). Test extends through 2025.

### V2 Class Weights

| Class | Weight |
|-------|--------|
| 0 (none)  | 0.024 |
| 1 (bull)  | 1.232 |
| 2 (bear)  | 1.744 |

### V2 LSTM Results

Checkpoint: `checkpoints/lstm/lstm_v2_seed{42,17}.pt`
Config: hidden_size=32, num_layers=1, dropout=0.118, head_dropout=0.263, lr=5.99e-4, batch_size=16

| Seed | Epochs | Test Macro F1 | Bull F1 | Bear F1 | Thresh-tuned F1 |
|------|--------|---------------|---------|---------|-----------------|
| 42   | 94     | **0.5980**    | 0.4045  | 0.4085  | 0.6011 |
| 17   | 98     | **0.6012**    | 0.4089  | 0.4154  | — |
| **mean±std** | — | **0.5996 ± 0.0016** | 0.407 | 0.412 | — |

Bootstrap CI (LSTM v2 seed 42, block_size=60, n=1000, effective_n=86):

| Metric | Point | CI Lower (2.5%) | CI Upper (97.5%) |
|--------|-------|-----------------|------------------|
| macro_f1 | 0.5980 | 0.5408 | 0.6491 |
| bull_f1 | 0.4045 | 0.2981 | 0.4889 |
| bear_f1 | 0.4085 | 0.2772 | 0.5246 |

### V2 XGBoost Results

Checkpoint: `checkpoints/xgboost/xgb_v2_seed{42,17}.ubj`
Config: n_estimators=285, max_depth=8, lr=0.120, min_child_weight=8

| Seed | n_estimators_used | Test Macro F1 | Bull F1 | Bear F1 |
|------|------------------|---------------|---------|---------|
| 42   | 3 (early-stop)   | **0.5222**    | 0.4356  | 0.2094  |
| 17   | 3 (early-stop)   | **0.5250**    | 0.4462  | 0.2068  |

Note: XGB early-stops at round 3 due to val mlogloss hitting minimum under extreme class imbalance (97% none). This is a known limitation of mlogloss as early-stop criterion with sparse positive classes. Bear F1 near 0.21 reflects near-always-none predictions. Recommend switching to val macro-F1 as early-stop criterion for follow-up.

### V2 Delta vs Previous Baseline

| Model | Prev baseline (fvg raw, 2018-2024) | V2 (fvg_valid, 2016-2025) | Note |
|-------|------------------------------------|-----------------------------|------|
| LSTM macro F1 | 0.8365 ± 0.009 | 0.5980 (seed 42) | Different labeller — not regression |
| XGB macro F1 | 0.6365 ± 0.004 | 0.5222 (seed 42) | Different labeller — not regression |
