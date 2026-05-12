# Build Log — Data Extension 2016-2025

**Date:** 12-May-26
**Scope:** Extend SPY H1 dataset from 2018-2024 to 2016-2025. Retrain LSTM + XGBoost with best Optuna configs.

---

## Alpaca 2015 Capability Findings

Probed Alpaca free tier IEX feed:
- `2015-01-05`: 0 bars (unavailable)
- `2015-12-01`: 0 bars (unavailable)
- `2016-01-04`: 825 bars (first available)

**Actual earliest date: 2016-01-04.** IEX historical data starts 2016, not 2015. User requested 2015-01-01 — adjusted to 2016-01-01 (Alpaca gracefully returns from 2016-01-04). Documented in this log.

---

## New Split Rationale

Old: train 2018-2021, val 2022, test 2023-2024.
New: **train 2016-2021, val 2022, test 2023-2025**.

Rationale:
1. 2016-2017 data available and confirmed (2016-01-04 first bar).
2. Extends training by +3,521 bars (2016+2017 = ~33% increase from 7,056 to 10,577).
3. Val stays 2022 (unchanged) — preserves continuity with previous experiment.
4. Test extends from 2023-2024 to 2023-2025: adds full 2025 year (+1,743 bars), more robust out-of-distribution evaluation.
5. 2016-2017 are post-VIX normalisation years — comparable regime to 2018-2021.

**Split boundaries in `src/data/split.py`:**
- `train_end`: "2021-12-31" (unchanged)
- `val_start/end`: "2022-01-01"/"2022-12-31" (unchanged)
- `test_start`: "2023-01-01" (unchanged)
- `test_end`: "2025-12-31" (extended from 2024)

---

## Dataset Stats

### New dataset (2016-2025)

| Split | Bars | Dates | none | bull (1) | bear (2) |
|-------|------|-------|------|----------|----------|
| train | 10,577 | 2016-01-04 → 2021-12-31 | 10,234 (96.8%) | 201 (1.9%) | 142 (1.3%) |
| val   | 1,757  | 2022-01-03 → 2022-12-30 | 1,692 (96.3%) | 27 (1.5%)  | 38 (2.2%)  |
| test  | 5,257  | 2023-01-03 → 2025-12-30 | 5,094 (96.9%) | 96 (1.8%)  | 67 (1.3%)  |
| **total** | **17,591** | | | | |

**IMPORTANT — labeller change note:** The rebuilt parquets use `fvg_valid` labeller (N+2, 6-criteria strict). Old rigor sprint models (0.8365 mean macro F1) were trained on `fvg` (raw, N+1) labels with ~25% positive rate. The new dataset has ~3.2% positive rate. These two results are NOT directly comparable — `fvg_valid` is a harder, sparser classification problem.

### Class weights (recomputed from new train split)

| Class | Weight |
|-------|--------|
| 0 (none)  | 0.024 |
| 1 (bull)  | 1.232 |
| 2 (bear)  | 1.744 |

Previous weights: none=0.217, bull=1.097, bear=1.686. The none class weight is now drastically lower (0.024 vs 0.217) due to the higher class imbalance with fvg_valid labels.

### Window counts (stride=1)

| Split | Windows |
|-------|---------|
| train | 10,518  |
| val   | 1,698   |
| test  | 5,198   |

---

## Training Results

### LSTM v2

Config (best Optuna, from `reports/rigor/2026-05-11_222208/best_lstm_config.json`):
- hidden_size=32, num_layers=1, dropout=0.118, head_dropout=0.263
- lr=0.000599, weight_decay=0.000122, batch_size=16

| Seed | Epochs | Best Val Smoothed F1 | Test Macro F1 | Bull F1 | Bear F1 |
|------|--------|---------------------|---------------|---------|---------|
| 42   | 94     | 0.6002              | **0.5980**    | 0.4045  | 0.4085  |
| 17   | 98     | 0.6284              | **0.6012**    | 0.4089  | 0.4154  |
| **mean±std** | — | — | **0.5996 ± 0.0016** | 0.407 | 0.412 | — |

Threshold-tuned seed 42: macro F1 = **0.6011** (delta +0.003).
Bootstrap CI (seed 42, block_size=60, n=1000): macro_f1 0.5980 [0.5408, 0.6491]

### XGBoost v2

Config (best Optuna, from `reports/rigor/2026-05-11_222549/best_xgb_config.json`):
- n_estimators=285, max_depth=8, lr=0.120, min_child_weight=8, subsample=0.679, colsample_bytree=0.665

| Seed | n_estimators_used | Test Macro F1 | Bull F1 | Bear F1 |
|------|------------------|---------------|---------|---------|
| 42   | 3 (early stop)   | **0.5222**    | 0.4356  | 0.2094  |
| 17   | 3 (early stop)   | **0.5250**    | 0.4462  | 0.2068  |

**XGB early-stop anomaly:** XGB hits val mlogloss minimum at round 3 with the highly imbalanced dataset (96.8% none). The model learns to predict none correctly very fast, then val mlogloss flattens. The poor bear F1 (0.21) reflects the model collapsing to near-always-none predictions for bear class. This is expected behaviour with mlogloss + extreme imbalance — same issue as the old pipeline but worse at 1.3% bear rate.

---

## Delta vs Previous Baselines

Note: previous baselines used `fvg` (raw labels, 25% positive). These are on `fvg_valid` (3% positive). Comparison is informational only.

| Model | Previous (fvg raw, 2018-2024) | New (fvg_valid, 2016-2025) | Delta |
|-------|------------------------------|---------------------------|-------|
| LSTM mean macro F1 | 0.8365 (5 seeds) | 0.5980 (seed 42) | -0.239 |
| XGB mean macro F1 | 0.6365 (5 seeds) | 0.5222 (seed 42) | -0.114 |

**Delta explanation:** Not a regression — different problem. `fvg_valid` has 8× fewer positive labels. Both models are solving a much harder sparse detection task. LSTM relative drop is larger because it needs more data to learn the sparse pattern.

---

## Code Changes

- `src/data/download.py`: `end` param default changed from `None` to `"2025-12-31"`. Backwards compat preserved (explicit `end=None` still works).
- `src/data/process.py`: `build_labelled_dataset` now accepts `start`/`end` params, passes to `download_spy_h1`.
- `src/data/pipeline.py`: `build_pipeline` now accepts `start`/`end`/`use_cache` params.
- `src/data/split.py`: `test_end` extended to "2025-12-31". `SPLIT_BOUNDARIES_2018_2024` added as legacy reference.
- `scripts/data/rebuild_dataset_2016_2025.py`: new script for full pipeline rebuild.
- `scripts/training/train_lstm_v2.py`: new LSTM trainer with Optuna best config.
- `scripts/training/train_xgb_v2.py`: new XGB trainer with Optuna best config.

---

## Backup

- `data/raw/spy_minute_2018_2024.backup.parquet` — original 52MB 2018-2024 minute bars.
- Old checkpoints (`lstm_seed*.pt`, `xgb_seed*.ubj`) preserved. New ones suffixed `_v2_seed*`.

---

## Open Flags

1. **LSTM seed 17 results pending** — update this log when complete.
2. **XGB early-stop degeneracy**: mlogloss-based early stopping is inadequate for 97% imbalanced data. Consider switching to val macro-F1 as early-stop criterion for XGB v2 follow-up.
3. **Labeller mismatch for comparison**: old rigor baseline on `fvg` raw labels; v2 on `fvg_valid`. Direct F1 comparison misleading. The field `fvg_valid` is the correct production target.
4. **Bootstrap CI for seed 17**: not run (seed 42 representative, both seeds consistent ±0.003).
5. **Test 2025 alone**: not isolated — test set covers 2023-2025 as a block. Out-of-distribution on 2025 specifically is unknown.

---

## Next Steps

1. Run CNN-LSTM on v2 dataset as next DL step.
2. Fix XGB early-stopping criterion (macro-F1 on val, not mlogloss) for fair comparison.
3. Multi-seed run for LSTM v2 (seeds 0, 123, 2024) to establish mean±std.
4. Update `docs/models-status.md` with v2 results section.
