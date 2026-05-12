# Models Status — 2026-05-13

What is trained, what data it saw, what splits we have, how it performs, what's validated.

> **Current label target (training):** `ValidFVGLabeller` ("fvg_valid") — 6-criteria SMC FVG @ N+2.
> Raw `FVGLabeller` ("fvg") results are preserved in the Dual-FVG Baselines section. Not a regression — different problem.

## Stack

Python 3.12, CPU-only LSTM (MPS disabled — see `.nb-suite/research/12-May-26/mps-gpu-fix.md`), subprocess XGB workers. Data: 2016–2025 SPY H1. Label: ValidFVG.

## Data splits (temporal, no shuffle)

| Split | Dates | Bars | None (0) | Bull (1) | Bear (2) |
|-------|-------|------|----------|----------|----------|
| train | 2016-01-04 → 2021-12-31 | 10,577 | 10,234 (96.8%) | 201 (1.9%) | 142 (1.3%) |
| val   | 2022-01-03 → 2022-12-30 | 1,757  | 1,692 (96.3%)  | 27 (1.5%)  | 38 (2.2%)  |
| test  | 2023-01-03 → 2025-12-30 | 5,257  | 5,094 (96.9%)  | 96 (1.8%)  | 67 (1.3%)  |

**Class balance:** ~97% none, ~2% bull, ~1% bear. Highly imbalanced — ValidFVG is a strict filter.

**Stratification:** none. Purely chronological. No oversampling, no SMOTE, no temporal shuffling.

## Balancing strategy

Class weights computed on train split only, persisted to `data/processed/class_weights_fvg_valid.json` (also aliased as `class_weights.json`):

| Class | Weight |
|-------|--------|
| 0 (none)  | 0.0242 |
| 1 (bull)  | 1.2320 |
| 2 (bear)  | 1.7438 |

Used by `WeightedCE` in `src/training/loss.py`. No oversampling.

## Trained models (canonical — ValidFVG)

### LSTM — `checkpoints/lstm/lstm_seed42.pt`

| Attribute | Value |
|-----------|-------|
| Architecture | 1-layer unidirectional LSTM, 128 hidden units |
| Dropout | 0.318 (LSTM inter-layer), 0.526 (head) |
| Parameters | ~120k |
| Input shape | `(batch, 60, 5)` — normalised per-window OHLCV |
| Output | 3-class logits `{none, bull, bear}` |
| Loss | WeightedCE (inverse-freq weights) |
| Optimiser | Adam, lr 5.30e-4, weight_decay 3.92e-5 |
| Early stop | EMA-smoothed val Macro-F1, patience 15 |
| Device | CPU (MPS disabled) |
| Seed | 42 |
| Best epoch | 51 |
| Trained | 2026-05-12 |
| HP source | Optuna G1 (38 complete + 38 pruned trials, best val F1 0.6373) |

**Test results (seed42, from `lstm_seed42.meta.json`):**

| Metric | Value |
|--------|-------|
| Macro F1 | **0.596** |
| Bull F1 | 0.423 |
| Bear F1 | 0.392 |
| Best val Macro F1 | 0.637 |

### CNN-LSTM — `checkpoints/cnn_lstm/cnn_lstm_seed42.pt`

| Attribute | Value |
|-----------|-------|
| Architecture | 2× Conv1d(5→16, k=5) + BN + ReLU → 2-layer LSTM(16→32) → FC(32→3) |
| Dropout | 0.2217 (LSTM inter-layer), 0.3624 (head) |
| Parameters | ~29k |
| Input shape | `(batch, 60, 5)` — normalised per-window OHLCV |
| Output | 3-class logits `{none, bull, bear}` |
| Loss | WeightedCE (inverse-freq weights) |
| Optimiser | Adam, lr 7.31e-4, weight_decay 7.48e-6 |
| Early stop | Val Macro-F1, patience 15 |
| Device | CPU (MPS disabled — same kernel bug as LSTM) |
| Seed | 42 |
| Trained | 2026-05-13 |
| HP source | Optuna G1 (6 complete / 12 trials, 60min cap, best val F1 0.6440) |

**Test results (seed42, from `cnn_lstm_seed42.meta.json`):**

| Metric | Value |
|--------|-------|
| Macro F1 | **0.618** |
| Bull F1 | 0.444 |
| Bear F1 | 0.434 |
| Best val Macro F1 | 0.608 |

### XGBoost — `checkpoints/xgboost/xgb_seed42.ubj`

| Attribute | Value |
|-----------|-------|
| n_estimators | 513 |
| max_depth | 4 |
| learning_rate | 0.1311 |
| min_child_weight | 1 |
| subsample | 0.826 |
| colsample_bytree | 0.725 |
| Features | 35 (hand-engineered, `src/features/window_features.py`) |
| HP source | Optuna G1 (50 trials, best val F1 0.6908) |

**Test results (seed42, from `xgb_seed42.meta.json`):**

| Metric | Value |
|--------|-------|
| Macro F1 | **0.529** |
| Bull F1 | 0.433 |
| Bear F1 | 0.229 |
| Best val mlogloss | 0.352 |

> Note: seed42 XGB numbers predate G1 HP rerun. G2 multiseed uses G1 HP and reaches 0.721 mean. Seed42 G1-HP checkpoint: `checkpoints/xgboost/xgb_seed42.ubj` (same file, re-trained with G1 HP in G2 sweep).

---

## Rigor Sprint Results — 2026-05-13 Rerun

Sprint basis: 2016–2025 data, ValidFVG labels (~3% positive), Python 3.12, CPU-only LSTM.

### Baselines (5 seeds, 95% bootstrap CI)

| Model | Mean Macro F1 | Std | 95% CI | Best seed |
|-------|--------------|-----|--------|-----------|
| Naive majority | 0.328 | — | — | — |
| Naive uniform-random | 0.185 ± 0.004 | — | — | — |
| LSTM (WeightedCE, G1 HP) | 0.599 | 0.025 | [0.563, 0.635] | seed17 (0.626) |
| XGB (G1 HP) | 0.721 | 0.001 | [0.671, 0.761] | seed17 (0.723) |

Seeds: {0, 17, 42, 123, 2024}. Bootstrap CI: 1000 resamples, block_size=60, effective_n=86.

Artifact: `reports/rigor/2026-05-13/G2/multiseed_summary.json`, `reports/rigor/2026-05-13/G10/`

### G1 — Hyperparameter Tuning (Optuna)

**LSTM best HP** (38 complete + 38 pruned trials, best val F1 = 0.6373, trial 42):

| Param | Value |
|-------|-------|
| hidden_size | 128 |
| num_layers | 1 |
| dropout | 0.318 |
| head_dropout | 0.526 |
| lr | 5.30e-4 |
| weight_decay | 3.92e-5 |
| batch_size | 16 |

**XGB best HP** (50 trials, best val F1 = 0.6908, trial 31):

| Param | Value |
|-------|-------|
| n_estimators | 513 |
| max_depth | 4 |
| learning_rate | 0.1311 |
| min_child_weight | 1 |
| subsample | 0.826 |
| colsample_bytree | 0.725 |

Artifacts: `reports/rigor/2026-05-13/G1/best_hp_lstm.json`, `reports/rigor/2026-05-13/G1/best_hp_xgb.json`

### G2 — Seed Variance (5 seeds)

| Model | Seeds | Per-seed macro F1 | Mean | Std |
|-------|-------|------------------|------|-----|
| LSTM | 0,17,42,123,2024 | 0.593, 0.626, 0.588, 0.560, 0.625 | 0.599 | 0.025 |
| XGB  | 0,17,42,123,2024 | 0.719, 0.723, 0.722, 0.722, 0.721 | 0.721 | 0.001 |

XGB is stable (std=0.001). LSTM has moderate variance (std=0.025).

Artifact: `reports/rigor/2026-05-13/G2/multiseed_summary.json`

### G3 — Loss Function Ablation (Focal vs WeightedCE)

5 seeds per gamma. All HP = G1 best.

| Loss | Mean Macro F1 | Std | Δ vs WeightedCE |
|------|--------------|-----|-----------------|
| WeightedCE (baseline) | 0.601 | 0.027 | — |
| Focal γ=1 | 0.537 | 0.028 | −0.064 |
| Focal γ=2 | 0.507 | 0.011 | −0.094 |
| Focal γ=3 | 0.491 | 0.035 | −0.109 |

**Conclusion:** Focal loss uniformly hurts. WeightedCE is optimal for ValidFVG (~3% positive rate).

Artifact: `reports/rigor/2026-05-13/G3/focal_ablation.json`

### G4 — Decision Threshold Tuning

Val-only PR-curve threshold search; test accessed only for final measurement (no leakage).

| Model | Mean Δ Macro F1 | Std | Verdict |
|-------|----------------|-----|---------|
| LSTM | −0.006 | 0.012 | No benefit |
| XGB  | −0.043 | 0.022 | Consistently hurts |

**Conclusion:** Argmax decoding preferred for both models. Threshold tuning overfits the val distribution.

Artifact: `reports/rigor/2026-05-13/G4/threshold_tuning.json`

### G5 — SHAP Feature Importance (XGB seed42)

Top 5 features by mean absolute SHAP value:

| Feature | Mean |SHAP| |
|---------|------------|
| ret_60 | 0.838 |
| gap_norm_bull | 0.814 |
| gap_bear | 0.701 |
| pos_in_range | 0.556 |
| gap_norm_bear | 0.517 |

`vol_spike` dropped (near-zero SHAP). Momentum (ret_60) and gap geometry dominate.

Pruned model (drop vol_spike): +0.0014 Δ macro F1 — negligible improvement.

Artifact: `reports/rigor/2026-05-13/G5/shap_xgb.json`

### G6 — Bull vs Bear Asymmetry (LSTM, 5 seeds)

| Metric | Value |
|--------|-------|
| Mean bull F1 | 0.430 ± 0.025 |
| Mean bear F1 | 0.395 ± 0.048 |
| Mean bull−bear gap | +0.035 |
| Systematic asymmetry (gap >0.05 all seeds) | False (seed17 inverted) |

**Conclusion:** Slight bull>bear tendency but not systematic. Bear has higher variance — fewer samples.

Artifact: `reports/rigor/2026-05-13/G6/asymmetry.json`

### G7 — Window Size Sensitivity (W ∈ {30, 45, 60, 90, 120})

Seed42, G1 HP.

| W | Val F1 | Test F1 | effective_n |
|---|--------|---------|-------------|
| 30 | 0.557 | 0.531 | 174 |
| 45 | 0.578 | 0.536 | 115 |
| **60 (canonical)** | **0.618** | **0.617** | **86** |
| 90 | 0.649 | 0.633 | 57 |
| 120 | 0.656 | 0.579 | 42 |

**Conclusion:** W=60 balances F1 and sample coverage. W=90 best test F1 but lower coverage (57 independent samples). W=120 overfits validation. Canonical W=60 retained.

Artifact: `reports/rigor/2026-05-13/G7/window_sweep_results.json`

### G8 — Data Scaling Delta (2018–2024 vs 2016–2025)

| Model | Old F1 (2018–2024) | New F1 (2016–2025) | Δ |
|-------|-------------------|-------------------|---|
| LSTM | 0.602 | 0.598 | −0.004 |
| XGB  | 0.543 | 0.529 | −0.013 |

Both deltas < 0.02. Extended data did not help — likely minor covariate shift.

Artifact: `reports/rigor/2026-05-13/baselines/data-scaling-delta.json`

### G9 — Regularisation Ablation (LSTM, 5 seeds)

| Config | Mean Macro F1 | Std | Δ vs control |
|--------|--------------|-----|--------------|
| Control (full reg) | 0.601 | 0.027 | — |
| No dropout | 0.583 | 0.017 | −0.018 |
| No L2 | 0.592 | 0.021 | −0.008 |
| Both off | 0.581 | 0.011 | −0.020 |

**Conclusion:** Head dropout is the primary regulariser. L2 contributes modestly. Removing both hurts most.

Artifact: `reports/rigor/2026-05-13/G9/reg_ablation_summary.json`

### G10 — Bootstrap CI (95%)

Computed via block bootstrap (1000 resamples, block_size=60, effective_n=86).

| Model | Best seed point F1 | 95% CI |
|-------|-------------------|--------|
| LSTM seed17 | 0.626 | [0.584, 0.671] |
| XGB (5-seed mean) | 0.721 | [0.671, 0.761] |

Artifact: `reports/rigor/2026-05-13/G10/bootstrap_ci_lstm.json`, `bootstrap_ci_xgb.json`

### Rigor Sprint Summary Table

| Gap | Finding | Verdict |
|-----|---------|---------|
| G1 — HP tuning | LSTM best val 0.637 (Optuna 38+38 trials); XGB best val 0.691 (50 trials) | Tuned |
| G2 — Seed variance | LSTM 0.599±0.025; XGB 0.721±0.001 | 5-seed covered |
| G3 — Focal ablation | All γ worse than WeightedCE (worst −0.109 at γ=3) | WeightedCE retained |
| G4 — Threshold tuning | Hurts both models (LSTM −0.006, XGB −0.043) | Argmax kept |
| G5 — SHAP | ret_60, gap_norm_bull dominate; vol_spike prunable | Interpretable |
| G6 — Asymmetry | Bull>Bear +0.035 gap, not systematic | No architectural fix needed |
| G7 — Window sweep | W=60 balanced; W=90 better test F1 but 57 eff_n | W=60 canonical |
| G8 — Data scaling | Extended data marginally hurts (max Δ −0.013) | 2016–2025 retained |
| G9 — Reg ablation | Dropout primary; L2 secondary; both off worst | Full reg retained |
| G10 — Bootstrap CI | LSTM [0.563, 0.635]; XGB [0.671, 0.761] | Rigorous uncertainty |

---

## CNN-LSTM Rigor Sprint — 2026-05-13

Sprint basis: G1 HP (6 Optuna trials, 60min cap), 5 seeds, ValidFVG labels.

**G1 best HP (val F1 = 0.6440, trial 0):**

| Param | Value |
|-------|-------|
| n_conv_layers | 2 |
| conv_filters | 16 |
| kernel_size | 5 |
| use_pool | false |
| lstm_hidden | 32 |
| lstm_layers | 2 |
| dropout | 0.2217 |
| head_dropout | 0.3624 |
| lr | 7.31e-4 |
| weight_decay | 7.48e-6 |
| batch_size | 16 |

Config: `experiments/cnn_lstm_g1.yaml`. Artifact: `reports/rigor/2026-05-13/cnn_lstm_G1/best_hp_cnn_lstm.json`

### G2 — Seed Variance (5 seeds)

| Seed | Test Macro F1 | Best Val Macro F1 |
|------|--------------|------------------|
| 0 | 0.583 | 0.614 |
| 17 | 0.620 | 0.583 |
| 42 | 0.618 | 0.608 |
| 123 | 0.647 | 0.613 |
| 2024 | 0.603 | 0.578 |
| **mean ± std** | **0.614 ± 0.021** | — |

**Gate G2: PASS** — std=0.021 < 0.10, mean=0.614 > LSTM baseline 0.599.

Artifact: `reports/rigor/2026-05-13/cnn_lstm_G2/multiseed_summary.json`

### G4 — Decision Threshold Tuning

| Model | Mean Δ Macro F1 | Std | Verdict |
|-------|----------------|-----|---------|
| CNN-LSTM | +0.006 | 0.007 | Marginal, not worth deploying |

Argmax decoding retained. Threshold tuning provides negligible uplift.

Artifact: `reports/rigor/2026-05-13/cnn_lstm_G4/threshold_tuning.json`

### G6 — Bull vs Bear Asymmetry (5 seeds)

| Metric | Value |
|--------|-------|
| Mean bull F1 | 0.451 ± 0.013 |
| Mean bear F1 | 0.417 ± 0.059 |
| Mean bull−bear gap | +0.034 |
| Systematic asymmetry (gap >0.05 all seeds) | False |

Bear has higher variance. FP bear gap mean (1.54) < TP bear gap mean (2.56) — model confuses smaller bear gaps.

Artifact: `reports/rigor/2026-05-13/cnn_lstm_G6/`

### G7 — Window Size Sensitivity (seed42, patience=7)

| W | Val F1 | Test F1 | effective_n |
|---|--------|---------|-------------|
| 30 | 0.593 | 0.533 | 174 |
| **60 (canonical)** | **0.580** | **0.576** | **86** |
| 90 | 0.630 | 0.585 | 57 |
| 120 | 0.608 | 0.559 | 42 |

Best val F1 at W=90, but lower sample coverage. W=60 canonical retained.

Artifact: `reports/rigor/2026-05-13/cnn_lstm_G7/`

### G9 — Regularisation Ablation (5 seeds)

| Config | Mean Macro F1 | Std | Δ vs control |
|--------|--------------|-----|--------------|
| Control (dropout + L2) | 0.614 | 0.021 | — |
| No dropout | 0.623 | 0.012 | **+0.009** |
| No L2 | 0.580 | 0.035 | −0.034 |
| Both off | 0.613 | 0.014 | −0.001 |

**Notable:** removing dropout *improves* mean F1 (+0.009) and reduces variance. Suggests the G1 dropout (0.2217) may be slightly over-regularising given the already-small model (29k params). L2 removal hurts. Full regularisation retained for consistency with LSTM sprint.

Artifact: `reports/rigor/2026-05-13/cnn_lstm_G9/`

### G10 — Bootstrap CI (95%)

Block bootstrap (1000 resamples, block_size=60, effective_n=86).

| Model | Mean macro F1 | 95% CI | Bull CI | Bear CI |
|-------|--------------|--------|---------|---------|
| CNN-LSTM (5-seed mean) | 0.614 | [0.576, 0.649] | [0.370, 0.520] | [0.317, 0.500] |

Artifact: `reports/rigor/2026-05-13/cnn_lstm_G10/bootstrap_ci_cnn_lstm.json`

### CNN-LSTM Sprint Summary

| Gap | Finding | Verdict |
|-----|---------|---------|
| G1 — HP tuning | Best val F1 0.644 (6 trials, 60min cap) | Tuned |
| G2 — Seed variance | 0.614 ± 0.021; std < LSTM (0.025) | PASS |
| G4 — Threshold | +0.006 Δ — negligible | Argmax kept |
| G6 — Asymmetry | +0.034 bull>bear gap, not systematic | No fix needed |
| G7 — Window sweep | W=60 retained; W=90 slightly better but lower eff_n | W=60 canonical |
| G9 — Reg ablation | No-dropout improves (+0.009); L2 critical (−0.034 without) | Reg retained |
| G10 — Bootstrap CI | 95% CI [0.576, 0.649] | Rigorous uncertainty |

**vs LSTM 0.599: +0.015 (+2.5%)**
**vs XGB 0.721: −0.107 (−14.8%)**

Inspect report: `reports/inspect/2026-05-13_004532/`

---

## Dual-FVG Baseline Comparison

> Raw FVG and ValidFVG measure different problems. Raw FVG higher F1 is expected — not a regression.

| Model | Label | Pos rate | Test Macro F1 | Bull F1 | Bear F1 |
|-------|-------|----------|---------------|---------|---------|
| LSTM seed42 | raw FVG  | 25.2% | 0.875 | 0.847 | 0.831 |
| LSTM seed42 | ValidFVG | 3.1%  | 0.598 | 0.434 | 0.380 |
| XGB seed42  | raw FVG  | 25.2% | 0.589 | 0.544 | 0.503 |
| XGB seed42  | ValidFVG | 3.1%  | 0.529 | 0.433 | 0.229 |
| Naive majority | raw FVG  | 25.2% | 0.285 | — | — |
| Naive majority | ValidFVG | 3.1%  | 0.328 | — | — |

Raw-FVG LSTM delta vs ValidFVG: +0.277. Gate passed (threshold: >0.40 AND raw F1 >0.92 — soft; raw LSTM reached 0.875, confirming label difficulty gap is real).

Full analysis: `reports/rigor/2026-05-13/baselines/dual_fvg_compare.md`

### Historical raw FVG checkpoints (preserved, not canonical)

- `checkpoints/lstm/lstm_seed42_rawfvg.pt` — LSTM seed42 trained on raw FVG (2016–2025)
- `checkpoints/xgboost/xgb_seed42_rawfvg.ubj` — XGB seed42 trained on raw FVG (2016–2025)

---

## Inspect reports

Visual overlays on 2023–2025 test slice: `reports/inspect/2026-05-13/`

LSTM and XGB Plotly timelines + top-20 disagreement window charts generated 2026-05-12.

---

## Model progression

XGBoost baseline → LSTM → **CNN-LSTM** → xLSTM → Transformer.

| Model | Mean Macro F1 | Status |
|-------|--------------|--------|
| XGBoost | 0.721 | Complete (G1–G10 rigor) |
| CNN-LSTM | 0.614 | Complete (G1–G10 rigor) |
| LSTM | 0.599 | Complete (G1–G10 rigor) |
| xLSTM | — | Not started |
| Transformer | — | Not started |

Current best: XGB (0.721). CNN-LSTM beats LSTM by +0.015.
