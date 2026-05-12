# Plan: Model Rigor Sprint

**Date:** 12-May-26
**Depth:** Deep
**Type:** Improvement — 10 structured gaps in experimental rigour
**Executor:** nb-build, phase-by-phase with nb-test + nb-review per phase

---

## Architecture Overview

The sprint adds a `src/rigor/` package containing reusable utilities (Optuna wrapper,
seed sweeper, bootstrap CI, threshold optimiser) and a suite of scripts under
`scripts/` for each gap. All experiment outputs land in `reports/rigor/<timestamp>/`
(gitignored). Build logs per gap in `.nb-suite/build/12-May-26/`. Tests for new
utilities in `tests/rigor/`.

No existing code is modified except:
- `docs/models-status.md` — updated in final consolidation phase
- `checkpoints/lstm/` and `checkpoints/xgboost/` — new checkpoint files added

The existing 81+ test count is preserved. New tests live entirely in `tests/rigor/`.

---

## Component Breakdown

### `src/rigor/` package — Foundation utilities

**`src/rigor/__init__.py`** — empty, marks package.

**`src/rigor/optuna_utils.py`**
- `LSTMObjective(train_loader, val_loader, weights, device, n_epochs, patience)` — callable class, samples hyperparams from Optuna trial, trains LSTM, returns val Macro F1 as objective value. Calls `set_seed(42)` internally (search uses fixed seed). Uses `MedianPruner` via intermediate value reporting after each epoch.
- `XGBObjective(X_train, y_train, X_val, y_val, sample_weights)` — callable class, builds XGB booster from trial params, returns val Macro F1.
- `run_study(objective, n_trials, storage_url, study_name) -> optuna.Study` — creates or loads study from SQLite, runs trials, returns study object.
- Returns best params dict and logs best trial to stdout.

**`src/rigor/seed_sweep.py`**
- `SeedSweepConfig(dataclass)` — holds model type (`"lstm"` or `"xgb"`), best hyperparams dict, seeds list, loss type (`"weighted_ce"` or `"focal"`), focal_gamma (float), output_dir path.
- `run_seed_sweep(config: SeedSweepConfig) -> pd.DataFrame` — trains one model per seed, evaluates on test set (loads from disk, NEVER called during search), returns DataFrame with columns `[seed, macro_f1, none_f1, bull_f1, bear_f1]`.
- Does NOT touch test set during the sweep loop — evaluates once per seed after training converges.
- Saves each checkpoint as `checkpoints/<model>/<name>_seed<N>.{pt,ubj}`.

**`src/rigor/threshold.py`**
- `compute_pr_curves(y_true, y_proba) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]` — one-vs-rest PR curves per class, returns `{class_id: (precision, recall, thresholds)}`.
- `find_f1_optimal_threshold(precision, recall, thresholds) -> tuple[float, float]` — returns `(threshold, f1_at_threshold)`.
- `apply_thresholds(y_proba, thresholds: dict[int, float]) -> np.ndarray` — argmax over adjusted probabilities using per-class thresholds; where no class exceeds its threshold the fallback is class 0 (none). Returns integer label array.

**`src/rigor/bootstrap_ci.py`**
- `block_bootstrap_f1(y_true, y_pred, block_size, n_iterations, seed) -> dict[str, tuple[float, float]]` — returns `{metric: (lower_95, upper_95)}` for macro_f1, bull_f1, bear_f1. Block size = `window_size` (60) by default. Returns both the CI bounds and the point estimate.
- `effective_n(n_bars, window_size, stride) -> int` — computes `n_bars // window_size` for non-overlapping window count.

**`src/rigor/report_utils.py`**
- `write_rigor_report(results_df, output_dir, title)` — writes a markdown summary table and saves Plotly HTML charts (PR curves, confusion matrices, SHAP summaries) into `output_dir`.
- `timestamped_dir(base: Path) -> Path` — creates `base/<YYYY-MM-DD_HHMMSS>/` and returns path.

### Scripts (CLI-driven, `--help` documented)

**`scripts/tune_lstm.py`**
- Args: `--n-trials 50`, `--storage checkpoints/lstm/optuna_<date>.db`, `--study-name lstm_fvg`, `--output-dir reports/rigor/`, `--device auto`
- Loads train + val loaders from parquet. Runs Optuna study. Saves HTML visualisation of study (param importances, optimisation history) to output dir. Prints best config JSON.
- Uses `src/rigor/optuna_utils.LSTMObjective` + `run_study`.

**`scripts/tune_xgboost.py`**
- Args: `--n-trials 50`, `--storage checkpoints/xgboost/optuna_<date>.db`, `--study-name xgb_fvg`, `--output-dir reports/rigor/`
- Loads windowed XGB feature matrices. Runs Optuna study. Same HTML output pattern.

**`scripts/multiseed_run.py`**
- Args: `--model {lstm,xgboost}`, `--config <json-file or inline JSON>`, `--seeds 42 17 0 123 2024`, `--loss {weighted_ce,focal}`, `--gamma 2`, `--output-dir reports/rigor/`
- Delegates to `src/rigor/seed_sweep.run_seed_sweep`. Writes `reports/rigor/<ts>/multiseed_summary.md` with mean±std table.
- Gap 3 (focal ablation) = same script with `--loss focal --gamma N`.

**`scripts/threshold_sweep.py`**
- Args: `--model-path`, `--model-type {lstm,xgboost}`, `--output-dir reports/rigor/`
- Loads val set, computes per-class PR curves, finds F1-optimal thresholds, applies to test set, reports delta vs argmax. Saves PR curve HTML.
- Uses `src/rigor/threshold.py`.

**`scripts/shap_xgb.py`**
- Args: `--checkpoint checkpoints/xgboost/xgb_seed42.ubj`, `--output-dir reports/rigor/`, `--drop-threshold 1e-4`
- Loads best XGB checkpoint (post-tuning). Runs `shap.TreeExplainer`. Saves summary plot + top-5 dependence plots as HTML. Identifies features below threshold, prints drop list. Retrains XGB with pruned features (same config), compares F1.
- Requires `shap` library — add to project deps.

**`scripts/window_sweep.py`**
- Args: `--windows 30 60 90 120`, `--config <json-file>`, `--seed 42`, `--output-dir reports/rigor/`
- For each window size: rebuilds windowed dataset from processed parquets (does not re-download), trains LSTM once, records val + test Macro F1. Saves window-vs-F1 plot HTML.

**`scripts/bootstrap_ci.py`**
- Args: `--predictions <npz with y_true + y_pred>`, `--block-size 60`, `--n-iter 1000`, `--seed 42`, `--output-dir reports/rigor/`
- Loads precomputed test predictions. Runs block bootstrap. Writes CI table to stdout + appends to `docs/models-status.md`.
- Predictions saved from `multiseed_run.py` as `<output-dir>/<model>_seed<N>_preds.npz`.

### `tests/rigor/` — new test files

- `tests/rigor/test_optuna_utils.py` — smoke test: `LSTMObjective` and `XGBObjective` run for 2 trials without error; study is created and loaded from SQLite; best params dict has all expected keys.
- `tests/rigor/test_seed_sweep.py` — mock training: `SeedSweepConfig` validates types; `run_seed_sweep` returns DataFrame with correct columns and shape for N seeds.
- `tests/rigor/test_threshold.py` — unit tests for `compute_pr_curves`, `find_f1_optimal_threshold`, `apply_thresholds`. Uses synthetic proba arrays with known optimal thresholds.
- `tests/rigor/test_bootstrap_ci.py` — `block_bootstrap_f1` returns correct CI shape; `effective_n(7056, 60, 1) == 117`; CI lower < point estimate < CI upper for large n.

---

## Data Model / Types

### `SeedSweepConfig` (dataclass)

```python
@dataclass
class SeedSweepConfig:
    model_type: Literal["lstm", "xgb"]
    hyperparams: dict[str, Any]        # output of best trial params
    seeds: list[int]
    loss_type: Literal["weighted_ce", "focal"] = "weighted_ce"
    focal_gamma: float = 2.0
    output_dir: Path = Path("reports/rigor")
    checkpoint_dir: Path = Path("checkpoints")
    data_dir: Path = Path("data/processed")
```

### Meta JSON extension (per checkpoint)

Existing `meta.json` gains these keys when produced by rigor scripts:

```python
{
    # existing keys preserved
    "optuna_study": "checkpoints/lstm/optuna_2026-05-12.db",  # or null
    "optuna_trial_number": 23,                                  # or null
    "loss_type": "weighted_ce",
    "focal_gamma": null,                                        # float if focal
    "threshold_config": null,                                   # dict if thresholds applied
}
```

### `bootstrap_ci` output

```python
{
    "macro_f1":  {"point": float, "ci_lower": float, "ci_upper": float},
    "bull_f1":   {"point": float, "ci_lower": float, "ci_upper": float},
    "bear_f1":   {"point": float, "ci_lower": float, "ci_upper": float},
    "n_bootstrap": int,
    "block_size": int,
    "effective_n": int,
}
```

---

## Sequential vs Parallel

**Must build in order:**

```
Phase 0 (Foundation) → Phase 1 (Gap 1 LSTM tune) and Phase 1b (Gap 1 XGB tune, parallel) →
Phase 2 (Gap 2+3 multi-seed, depends on best configs) →
Phase 3 (Gap 4 threshold, depends on trained checkpoints) →
Phase 3b (Gap 5 SHAP, independent of Gap 4 but needs best XGB) →
Phase 4 (Gap 6 bull/bear analysis, reads multi-seed results) →
Phase 5 (Gap 7 window sweep, independent) →
Phase 6 (Gap 9 regularisation ablation, reads optuna results) →
Phase 7 (Gap 10 bootstrap CI, depends on multi-seed preds) →
Phase 8 (Gap 8 data scaling research, fully independent) →
Phase 9 (Final consolidation)
```

**Can run in parallel after Phase 0:**
- Gap 1 LSTM tuning + Gap 1 XGB tuning (independent compute)
- Gap 5 SHAP + Gap 7 window sweep (independent after Phase 0)
- Gap 8 research (no compute dependency)

---

## Error Handling

- Optuna study: if SQLite file already exists, `run_study` loads it and resumes from existing trials (idempotent). Never deletes existing study.
- Seed sweep: if a checkpoint already exists for a given seed, skip retraining and load — allows resume after interruption. Detected by presence of matching `.meta.json`.
- `bootstrap_ci.py`: validates that `y_true` and `y_pred` have matching length before bootstrapping; raises `ValueError` with clear message if not.
- `threshold_sweep.py`: if val set produces no positives for a class (possible with very sparse labels), falls back to default threshold 0.5 and logs a warning — does NOT crash.
- All scripts write to `reports/rigor/<timestamp>/` which is gitignored — no risk of polluting repo with large HTML files.
- SHAP: `shap` library import is guarded with a clear `ImportError` message instructing `pip install shap`.

---

## Edge Cases and Constraints

### No-lookahead enforcement
- Optuna objective uses train + val only. No test data touched until `bootstrap_ci.py` at the very end (and only to load pre-saved predictions from the multi-seed run).
- `run_seed_sweep` evaluates on test at the END of each seed run — not during training or tuning. This is the one permitted touch of test per seed.

### Temporal split semantics
- Window sweep rebuilds datasets per window size from the same temporal split parquets. It does not re-split. The split boundary dates (2018–2021 / 2022 / 2023–2024) are fixed regardless of window size.
- Purge gap: no stride-level purge is needed for train/val since we use a fixed temporal boundary (not cross-validation). If blocked CV on train+val is implemented as an optional extension in Gap 10, the purge gap must equal `window_size` bars (not samples).

### Apple MPS constraints
- All training in rigor scripts inherits the same MPS safety rules from `CLAUDE.md`: `nn.LSTM` sequence-batched only, post-attention `x = x + 0` if transformer added, deterministic mode off.
- Optuna's `MedianPruner` calls intermediate value after each epoch — this is compatible with the existing `EarlyStop` logic (pruner fires before patience is exhausted on bad trials).

### SHAP + XGBoost subprocess
- XGBoost inference uses `_xgb_worker.py` subprocess workaround (Python 3.14+ segfault). SHAP `TreeExplainer` must be run in the same subprocess context as the XGB booster. `shap_xgb.py` must spawn the subprocess or call the worker directly.

### Focal loss gamma ablation
- `FocalLoss` is already implemented in `src/training/loss.py`. The ablation adds `gamma` as a CLI param to `multiseed_run.py`; it does NOT modify `FocalLoss` source. The default `WeightedCE` remains the primary loss — focal is comparison-only.

### Feature pruning after SHAP
- Features with mean |SHAP| < 1e-4 are dropped from the XGB feature vector. `window_features.py` must NOT be modified in place — instead, `shap_xgb.py` passes a `feature_mask` argument to a new thin wrapper that re-uses `window_features.extract_features` but drops the masked columns. This avoids breaking the existing 35-feature interface used by the production adapter.

### Compute budget
Rough wall-clock estimates on M4 Pro (48 GB RAM):

| Phase | Task | Estimate |
|-------|------|----------|
| 0 | Foundation build + tests | 30 min |
| 1 | LSTM Optuna 50 trials (~60 epochs each, early stop) | 4–6 hrs |
| 1b | XGB Optuna 50 trials | 20–40 min |
| 2 | Multi-seed × 5 seeds (LSTM best config) | 2–3 hrs |
| 2b | Multi-seed × 5 seeds (focal γ=1,2,3) = 15 runs | 6–9 hrs |
| 3 | Threshold sweep | 5 min |
| 3b | SHAP on XGB | 5–10 min |
| 4 | Bull/bear analysis (read-only + plots) | 30 min |
| 5 | Window sweep × 4 sizes, 1 seed each | 1.5–2 hrs |
| 6 | Regularisation ablation (4 combos × 1 seed) | 1–2 hrs |
| 7 | Bootstrap CI | 2 min |
| 8 | Data scaling research | researcher time, no GPU |
| 9 | Consolidation | 30 min |

**Total compute-intensive:** ~14–22 hrs of training. Run phases 1 and 1b overnight.

---

## Phases (nb-build execution sequence)

### Phase 0 — Foundation: `src/rigor/` package

**Goal:** all utility code written and tested before any experiment script is built.

**New files:**
- `src/rigor/__init__.py`
- `src/rigor/optuna_utils.py`
- `src/rigor/seed_sweep.py`
- `src/rigor/threshold.py`
- `src/rigor/bootstrap_ci.py`
- `src/rigor/report_utils.py`

**New tests:**
- `tests/rigor/__init__.py`
- `tests/rigor/test_optuna_utils.py`
- `tests/rigor/test_seed_sweep.py`
- `tests/rigor/test_threshold.py`
- `tests/rigor/test_bootstrap_ci.py`

**Dependencies added:** `optuna` (TPE sampler, MedianPruner, SQLite backend), `shap` (used in Phase 3b script but imported lazily).

**Exit criteria:** `pytest tests/rigor/` passes. No existing tests broken.

---

### Phase 1 — Gap 1: Hyperparameter Search

**Goal:** best LSTM config and best XGB config, each persisted.

**New files:**
- `scripts/tune_lstm.py`
- `scripts/tune_xgboost.py`

**LSTM search space:**
- `hidden_size`: categorical {32, 64, 128}
- `num_layers`: categorical {1, 2, 3}
- `dropout`: float uniform [0.1, 0.5]
- `head_dropout`: float uniform [0.1, 0.6]
- `lr`: float log-uniform [1e-4, 1e-2]
- `weight_decay`: float log-uniform [1e-5, 1e-1]
- `batch_size`: categorical {16, 32, 64}

**XGB search space:**
- `n_estimators`: int uniform [100, 600]
- `max_depth`: categorical {3, 4, 5, 6}
- `learning_rate`: float log-uniform [0.01, 0.2]
- `min_child_weight`: int uniform [1, 10]
- `subsample`: float uniform [0.6, 1.0]
- `colsample_bytree`: float uniform [0.6, 1.0]

**Pruner:** `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)`. Intermediate values reported at each epoch (LSTM) or each boosting round (XGB).

**Outputs:**
- `checkpoints/lstm/optuna_<YYYY-MM-DD>.db`
- `checkpoints/xgboost/optuna_<YYYY-MM-DD>.db`
- `reports/rigor/<ts>/lstm_optuna_summary.html` (param importances + history)
- `reports/rigor/<ts>/xgb_optuna_summary.html`
- `reports/rigor/<ts>/best_lstm_config.json`
- `reports/rigor/<ts>/best_xgb_config.json`

**Exit criteria:** 50 trials completed for each model. `best_lstm_config.json` and `best_xgb_config.json` exist and are valid JSON with all expected keys.

---

### Phase 2 — Gap 2 + Gap 3: Multi-seed + Focal Ablation

**Goal:** variance quantified; focal vs weighted-CE comparison.

**New files:**
- `scripts/multiseed_run.py`

**Runs executed by nb-build:**
1. `python scripts/multiseed_run.py --model lstm --config reports/rigor/<ts>/best_lstm_config.json --seeds 42 17 0 123 2024 --loss weighted_ce` → 5 checkpoints
2. `python scripts/multiseed_run.py --model xgboost --config reports/rigor/<ts>/best_xgb_config.json --seeds 42 17 0 123 2024 --loss weighted_ce` → 5 checkpoints
3. Three focal runs (LSTM only, best config): `--loss focal --gamma 1`, `--gamma 2`, `--gamma 3` × 5 seeds = 15 additional checkpoints

**Outputs per run:**
- `checkpoints/lstm/lstm_seed<N>_<loss><gamma>.pt` + `.meta.json`
- `reports/rigor/<ts>/multiseed_summary_lstm_weighted_ce.md`
- `reports/rigor/<ts>/multiseed_summary_lstm_focal_g<N>.md`
- `reports/rigor/<ts>/multiseed_summary_xgb_weighted_ce.md`
- Each run also saves `<output-dir>/<model>_seed<N>_preds.npz` (y_true + y_pred arrays) for bootstrap CI

**Summary format:** markdown table with columns `[seed, macro_f1, none_f1, bull_f1, bear_f1]` + bottom row `mean ± std`.

**Exit criteria:** 5 + 5 + 15 checkpoints saved. All summary markdown files present. `pytest tests/rigor/test_seed_sweep.py` still passes.

---

### Phase 3 — Gap 4: Threshold Tuning

**Goal:** identify per-class F1-optimal decision thresholds on val, measure test uplift.

**New files:**
- `scripts/threshold_sweep.py`

**Procedure:**
1. Load best LSTM checkpoint (WeightedCE, mean seed from Phase 2 — pick seed with macro_f1 nearest to mean).
2. Run inference on val set, collect `y_proba` (N×3).
3. Compute one-vs-rest PR curves for classes 1 (bull) and 2 (bear). Class 0 threshold = residual.
4. Find F1-optimal threshold per class via `src/rigor/threshold.find_f1_optimal_threshold`.
5. Apply thresholds to test set, compute macro F1 delta vs argmax baseline.
6. Save PR curve HTML to `reports/rigor/<ts>/pr_curves_lstm.html`.
7. Save threshold config JSON to `reports/rigor/<ts>/thresholds_lstm.json`.

**Outputs:**
- `reports/rigor/<ts>/pr_curves_lstm.html`
- `reports/rigor/<ts>/thresholds_lstm.json`
- Printed table: argmax vs threshold-tuned macro/bull/bear F1 on test

**Exit criteria:** script runs end-to-end, HTML and JSON files created.

---

### Phase 3b — Gap 5: SHAP on XGBoost

**Goal:** feature importance ranked, low-value features pruned, delta F1 measured.

**New files:**
- `scripts/shap_xgb.py`

**Procedure:**
1. Load best XGB checkpoint from Phase 1 (or Phase 2 seed-mean checkpoint if available).
2. Run `shap.TreeExplainer` on val set (for speed — train set would be 7k samples × 35 features, fine for TreeExplainer).
3. Compute mean |SHAP| per feature.
4. Save summary plot (bar chart, 35 features ranked) and top-5 dependence plots as HTML.
5. Identify features with mean |SHAP| < 1e-4.
6. Retrain XGB with pruned feature set (same best config, seed 42 only). Compare val + test macro F1.
7. Append findings table to `reports/rigor/<ts>/shap_xgb_report.md`.

**Outputs:**
- `reports/rigor/<ts>/shap_summary_xgb.html`
- `reports/rigor/<ts>/shap_dep_<feature>.html` (top 5)
- `reports/rigor/<ts>/shap_xgb_report.md`

**Exit criteria:** HTML files created, report markdown exists.

---

### Phase 4 — Gap 6: Bull vs Bear Asymmetry

**Goal:** diagnose systematic model bias between bull and bear FVG classes.

**New files:**
- `reports/rigor/<ts>/asymmetry.md` (written by analysis script)
- `scripts/asymmetry_analysis.py`

**Procedure:**
1. Load all multi-seed prediction `.npz` files from Phase 2 (WeightedCE, LSTM only — 5 seeds).
2. Compute per-class precision/recall/F1 per seed. Compute confusion matrix per seed.
3. Aggregate: mean precision, recall, F1 per class; flag if bull_f1 > bear_f1 by >0.05 across all seeds.
4. Structural investigation: for windows where model predicts bull but truth is none (FP), compare mean gap size (close[N] - high[N-2]) to TP windows. Requires loading raw parquet and joining on window index — use test split only.
5. Write `asymmetry.md` with: per-class tables, confusion heat-maps (saved as HTML), structural gap size comparison, conclusion (systematic or random?).

**Outputs:**
- `reports/rigor/<ts>/asymmetry.md`
- `reports/rigor/<ts>/confusion_<seed>.html` (5 files)
- `reports/rigor/<ts>/gap_size_distribution.html`

**Exit criteria:** `asymmetry.md` exists with all three sections (tables, confusion, structural).

---

### Phase 5 — Gap 7: Window Size Sweep

**Goal:** measure sensitivity of LSTM Macro F1 to context window length.

**New files:**
- `scripts/window_sweep.py`

**Procedure:**
1. For each window size W ∈ {30, 60, 90, 120}: rebuild `SMCWindowDataset` from `spy_h1_train.parquet` / val / test parquets with `window_size=W, stride=1`. No re-downloading.
2. Train LSTM with best config from Phase 1 (seed=42 only).
3. Record val and test Macro F1 at convergence.
4. Plot line chart: W on x-axis, macro F1 (val + test) on y-axis.
5. Note the best W for CNN-LSTM kernel sizing guidance.

**Outputs:**
- `reports/rigor/<ts>/window_sweep_results.json` (raw numbers)
- `reports/rigor/<ts>/window_sweep.html` (Plotly line chart)
- Printed recommendation for CNN-LSTM kernel range

**Compute note:** 4 training runs × ~30 epochs average = ~1.5 hrs.

**Exit criteria:** JSON and HTML files exist. Four rows in results JSON.

---

### Phase 6 — Gap 9: Explicit Regularisation Ablation

**Goal:** isolate contribution of dropout and weight_decay to generalisation.

**Procedure (not a new script — extend `multiseed_run.py` with `--ablate-reg` flag or run directly via config JSON):**

Four configurations (seed=42 only, best other hyperparams from Phase 1):
1. Best config (control)
2. Best config with `dropout=0.0, head_dropout=0.0` (no dropout)
3. Best config with `weight_decay=0.0` (no L2)
4. Best config with `dropout=0.0, head_dropout=0.0, weight_decay=0.0` (no reg)

**Outputs:**
- 3 additional LSTM checkpoints
- `reports/rigor/<ts>/reg_ablation.md` — table: config × val_macro_f1 × test_macro_f1

**Exit criteria:** 4-row table in `reg_ablation.md`.

---

### Phase 7 — Gap 10: Bootstrap CI

**Goal:** honest uncertainty bounds on all final test metrics.

**New files:**
- `scripts/bootstrap_ci.py`

**Procedure:**
1. Load mean-seed WeightedCE LSTM predictions `.npz` (y_true + y_pred from Phase 2).
2. Compute `effective_n`: 3514 // 60 = 58 non-overlapping test windows.
3. Run `block_bootstrap_f1(block_size=60, n_iterations=1000, seed=42)`.
4. Print CI table.
5. Append CI block to `docs/models-status.md` (under new section "## Confidence Intervals").

**Outputs:**
- `reports/rigor/<ts>/bootstrap_ci_lstm.json`
- Updated `docs/models-status.md`

**Exit criteria:** `docs/models-status.md` contains CI section with all three metrics.

---

### Phase 8 — Gap 8: Data Scaling Research

**Goal:** determine feasibility of extending training data pre-2018.

**Output:** `.nb-suite/research/12-May-26/data-scaling.md`

**Research questions:**
1. Does Alpaca free-tier provide SPY minute bars back to 2010? (Check via small API probe — pull Jan 2010 1 day, verify response is non-empty.)
2. If yes: what volume data quality is available pre-2015 (fragmented markets, ETF tracking error)?
3. If yes: what is compute cost of retraining on 2010–2022 train split (~2× bars)?
4. Block bootstrap augmentation: is MixUp on returns mathematically sound for autoregressive label data? Document tradeoffs.
5. Multi-ticker augmentation: QQQ correlation to SPY H1 FVG frequency — does it add distribution coverage?

**If extension feasible:** write execution plan in same file. Do NOT execute as part of this sprint — flag for separate phase.

**Exit criteria:** research doc exists with a binary "extend / defer" recommendation + justification.

---

### Phase 9 — Final Consolidation

**Goal:** `docs/models-status.md` reflects all sprint results.

**Updates to `docs/models-status.md`:**
1. New section "## Rigor Sprint Results (12-May-26)"
2. Tuned model configs (best hyperparams table)
3. Multi-seed variance table (mean ± std, WeightedCE)
4. Focal vs WeightedCE comparison table
5. Threshold-tuned F1 vs argmax baseline
6. Window size sweep results table
7. Regularisation ablation table
8. Bootstrap CI section (already written by Phase 7)
9. Bull/bear asymmetry summary (pointer to `asymmetry.md`)
10. SHAP top-5 features + pruning delta
11. Data scaling research outcome (extend / defer)

**Exit criteria:** `docs/models-status.md` updated. All referenced `reports/rigor/<ts>/` files exist.

---

## Test Coverage

| Component | Test file | What's tested |
|-----------|-----------|---------------|
| `optuna_utils.py` | `tests/rigor/test_optuna_utils.py` | 2-trial smoke run for both objectives; study create/load idempotency; best params keys |
| `seed_sweep.py` | `tests/rigor/test_seed_sweep.py` | Config dataclass validation; DataFrame shape/columns from mock sweep; checkpoint-exists skip logic |
| `threshold.py` | `tests/rigor/test_threshold.py` | `compute_pr_curves` on known proba arrays; `find_f1_optimal_threshold` returns correct threshold; `apply_thresholds` produces correct labels including fallback-to-0 |
| `bootstrap_ci.py` | `tests/rigor/test_bootstrap_ci.py` | `effective_n(7056, 60, 1) == 117`; CI bounds ordered (lower < point < upper); reproducible with seed |
| Existing tests | unchanged | All 81+ existing tests must still pass after Phase 0 |

Model training scripts (`tune_lstm.py`, `multiseed_run.py`, etc.) are **not** unit-tested — they are integration scripts that are manually verified by output artefact existence. The utilities they call are tested.

---

## Risk Register

| Risk | Likelihood | Blast Radius | Reversibility | Mitigation |
|------|-----------|--------------|---------------|------------|
| Optuna LSTM study takes >8 hrs (50 trials × 60 epochs) | Medium | Phase 1 delayed | Reversible — resume from DB | Set `max_epochs=50` for search (not final training); pruner cuts bad trials early |
| Test set touched prematurely during Optuna search | Medium | Invalidates all downstream metrics | Irreversible — must re-run | `LSTMObjective` never loads test parquet; code review gate before running |
| SHAP subprocess incompatibility with XGB worker | Medium | Gap 5 blocked | Reversible | Use `_xgb_worker.py` subprocess pattern already in codebase; SHAP call inside subprocess |
| Focal loss multi-seed adds 15+ overnight runs, delays timeline | High | Gap 3 slow | Reversible — reduce seeds | Batch gamma=1,2,3 in a single script invocation; run overnight |
| Window sweep W=120 reduces test set sample count (bars - 120 + 1) | Low | Misleading comparison | Reversible — note in report | Report effective N per window size alongside F1 |
| Bootstrap CI block size = 60 too conservative, CI very wide | Medium | Misleading result | Reversible — re-run | Also report block_size=1 (no block) as reference; note autocorrelation concern |
| `shap` pip install conflicts with existing deps | Low | Gap 5 blocked | Reversible | Pin `shap>=0.46` in requirements; test install before Gap 5 begins |
| Data scaling research finds pre-2018 Alpaca bars are unavailable | Medium | Gap 8 closed | Reversible — pivot to augmentation doc | Research doc documents outcome; sprint is not blocked on it |

---

## Pre-mortem (imagining failure in 3 months)

1. **Test set was leaked during Optuna search.** We didn't notice that `tune_lstm.py` loaded all three parquets for convenience and the objective accidentally computed a test-set score as the return value. All downstream F1 numbers are optimistic and the models are overfit to test. Fix: strict code review on `LSTMObjective.__call__` before any study runs.

2. **Focal loss ran with wrong class weights.** `FocalLoss` was called with alpha=None (uniform) instead of the inverse-freq weights from `class_weights.json`, making the WeightedCE vs Focal comparison unfair. Fix: `multiseed_run.py` must load and pass `class_weights.json` to both loss constructors unconditionally.

3. **Multi-seed variance looks great but is MPS-pseudo-random.** MPS does not guarantee full determinism even with seed set. Mean±std across seeds partly reflects MPS non-determinism, not model variance. Fix: note this limitation in every report; run one cross-check on CPU for seed 42 vs MPS seed 42 to quantify the delta.

4. **SHAP feature pruning retraining used the old `n_estimators=300` not the tuned config.** The pruned-feature F1 comparison is against a sub-optimal XGB, making pruning look more/less impactful than it is. Fix: `shap_xgb.py` must accept `--config` pointing to `best_xgb_config.json`.

5. **Window sweep conclusion picked W=30 as "best" but test macro F1 was higher due to less overlap, not better signal.** Effective sample count is not held constant across W. Fix: `window_sweep.py` reports `effective_n` (test_bars // W) alongside F1 and notes that comparison is confounded.

---

## Out of Scope

- CNN-LSTM and xLSTM model architectures (separate phase)
- Migration to `ValidFVGLabeller` training labels (separate phase — see 11-May-26 plan)
- Live paper trading validation (separate phase)
- Multi-symbol generalisation (QQQ, individual equities)
- 2025+ out-of-sample evaluation (no data collected yet)
- Any changes to `src/data/`, `src/live/`, `src/inspect/` packages
- Notebook updates (separate nb-notebook task after sprint completes)

---

## Final Report Structure

After Phase 9, `docs/models-status.md` will contain:

1. Existing sections (unchanged except CI appended)
2. `## Rigor Sprint Results (12-May-26)` — top-level with subsections:
   - `### Hyperparameter Tuning` — best configs, trial count, Optuna HTML link
   - `### Seed Variance` — mean ± std table (WeightedCE LSTM + XGB)
   - `### Loss Function Ablation` — WeightedCE vs Focal γ=1,2,3 table
   - `### Decision Threshold Tuning` — argmax vs tuned F1 delta
   - `### SHAP Feature Importance` — top-5 features, pruning delta
   - `### Bull vs Bear Asymmetry` — summary + pointer to `asymmetry.md`
   - `### Window Size Sensitivity` — table + recommendation for CNN-LSTM
   - `### Regularisation Ablation` — 4-config table
   - `### Bootstrap Confidence Intervals` — CI bounds for final LSTM
   - `### Data Scaling` — research outcome (extend / defer)

Artefacts in `reports/rigor/<timestamp>/`:
- `lstm_optuna_summary.html`, `xgb_optuna_summary.html`
- `best_lstm_config.json`, `best_xgb_config.json`
- `multiseed_summary_*.md`
- `pr_curves_lstm.html`, `thresholds_lstm.json`
- `shap_summary_xgb.html`, `shap_dep_*.html`, `shap_xgb_report.md`
- `asymmetry.md`, `confusion_<seed>.html`, `gap_size_distribution.html`
- `window_sweep_results.json`, `window_sweep.html`
- `reg_ablation.md`
- `bootstrap_ci_lstm.json`

---

## Resolved Ambiguities

- **Test set access:** only permitted in `run_seed_sweep` (once per seed at end of training) and `bootstrap_ci.py`. Never in Optuna objective. Never in threshold_sweep (uses val for threshold selection, test only for reporting the delta — which is a final measurement, not a search).
- **"Best config" definition:** best trial from Optuna study by val Macro F1 (mean of last 3 epochs, matching EarlyStop smoothing). Written to `best_<model>_config.json`.
- **Focal alpha:** always `class_weights.json` inverse-freq weights — never uniform. Ablation varies gamma only.
- **Checkpoint naming:** `<model>_seed<N>.pt` for WeightedCE; `<model>_focal_g<gamma>_seed<N>.pt` for focal.
- **`reports/` gitignore:** add `reports/rigor/` to `.gitignore` if not already covered by `reports/` entry.
- **`shap` version:** `shap>=0.46` — `TreeExplainer` API stable at this version.

---

## What to Flag to User

- After Phase 1: share `best_lstm_config.json` and `best_xgb_config.json` for sanity check before committing to Phase 2 compute.
- After Phase 2: share mean±std table before kicking off focal ablation (15 runs). Confirm gamma range is still desired.
- After Phase 8 research: share data-scaling.md recommendation before any data pull is attempted.
