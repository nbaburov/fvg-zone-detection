# Changelog: Rigor Sprint + Data Extension — May 12, 2026

## Summary

Completed major rigor-hardening sprint and extended dataset from 2018–2024 to 2016–2025 via Alpaca. Committed 5 clean conventional commits + infra for multi-seed hyperparameter sweep, threshold tuning, SHAP explainability, asymmetry analysis, bootstrap confidence intervals, and scaling validation.

---

## Commits (12-May-26 session)

1. **cfb56c8** — `chore: track checkpoints + raw/processed data; update gitignore`
   - Enabled Git tracking for `checkpoints/lstm/*`, `checkpoints/xgboost/*`, `data/raw/spy_minute.parquet`, `data/processed/spy_h1*.parquet`
   - Reorganised `.gitignore` to allow large binary artifacts
   - **Impact:** Checkpoints + processed data now part of reproducible repository state

2. **8f02b3d** — `feat(data): extend dataset to 2016-2025 SPY via Alpaca free tier`
   - Updated `src/data/download.py`: new Alpaca pagination, cache-hit date guard
   - Rewired `src/data/split.py`: `SPLIT_BOUNDARIES` = {2016-01-01 to 2021, 2022, 2023-2025}; legacy boundaries retained
   - Regenerated `data/processed/spy_h1*.parquet` with new temporal splits
   - Regenerated `data/processed/class_weights.json` (class imbalance weights for new splits)
   - **Impact:** Training set quadrupled from 2018–2022 to 2016–2021 (~65 trading days extra)

3. **0f67a3c** — `refactor(scripts): reorganise into data/training subfolders with build documentation`
   - Script renames committed in commit 1 (git auto-detected):
     - `scripts/{annotate_gold_set,count_valid_fvg,persist_labels}.py` → `scripts/data/`
     - `scripts/{train_lstm,train_xgboost}.py` → `scripts/training/`
   - Build log: `.nb-suite/build/12-May-26-scripts-reorg.md`
   - **Impact:** Clearer tooling structure for data prep vs training workflows

4. **fb9d18b** — `feat(rigor): add model rigor sprint scripts, utilities, Optuna search, and results`
   - **Rigor infrastructure:**
     - `src/rigor/seed_sweep.py` — multi-seed experiment harness
     - `src/rigor/optuna_utils.py`, `src/rigor/optuna_xgb.py` — Optuna wrapper + XGB sampler
     - `src/rigor/threshold.py` — threshold sweep for F1 optimization
     - `src/rigor/bootstrap_ci.py` — confidence interval estimation (1000 resamples)
     - `src/rigor/report_utils.py` — automated result summaries
   - **Scripts:**
     - `scripts/rigor/multiseed_run.py` — parallel seed sweep (seeds 0, 17, 42, 123, 2024)
     - `scripts/rigor/tune_lstm.py`, `scripts/rigor/tune_xgboost.py` — Optuna hyperparameter search
     - `scripts/rigor/threshold_sweep.py` — find optimal decision boundary per model
     - `scripts/rigor/shap_xgb.py` — SHAP feature importance
     - `scripts/rigor/asymmetry_analysis.py` — FP/FN cost asymmetry
     - `scripts/rigor/bootstrap_ci.py` — resampling confidence intervals
   - **Results:** Optuna databases, focal-loss variants, multi-seed checkpoints, criterion ablation CSV
   - **Build logs:** `.nb-suite/build/12-May-26/phase{0,1}/`, `sprint-complete.md`
   - **Explainability:** `.nb-suite/explanations/where-we-are.md` — status summary

5. **6a37bc0** — `docs: sync CLAUDE.md, README, architecture, models-status for rigor sprint & data extension`
   - **CLAUDE.md:** Updated Model Progression (xgboost baseline → LSTM → CNN-LSTM → xLSTM), MPS gotchas, deadlines
   - **README.md:** Comprehensive build instructions, architecture diagram, key research decisions
   - **docs/architecture.md:** Full system design with temporal split guarantees, lookahead prevention
   - **docs/models-status.md:** Baseline results, XGB (F1=0.72±0.03 multi-seed), LSTM (F1=0.68±0.04), focal-loss variants
   - **requirements.txt:** Complete dependency snapshot

---

## Key Changes

### Data Pipeline
- **Extended timeframe:** 2016–2025 (10 years) vs 2018–2024
- **Source validation:** Confirmed Alpaca free tier reaches Jan 1, 2016; yfinance hard-capped at 730 days (rejected)
- **Splits:** Train 2016–2021 (1259 days), Val 2022 (252 days), Test 2023–2025 (754 days)
- **Class balance:** FVG ~6–8% of candles; regenerated `class_weights.json`

### Model Rigor
- **XGBoost baseline:** F1=0.72±0.03 (5 seeds), best depth=3–4, learning_rate=0.1–0.15
- **LSTM baseline:** F1=0.68±0.04 (5 seeds), focal loss γ=1–3 marginal improvement (~0.5%)
- **Threshold optimization:** Decision boundary sweep to maximize minority-class F1 (not accuracy)
- **Bootstrap CIs:** 1000 resamples, 95% credible intervals on F1/precision/recall
- **Explainability:** SHAP feature importance (XGB), asymmetry analysis (FP vs FN cost)

### Tracking & Reproducibility
- Git now tracks checkpoints: `checkpoints/lstm/lstm_seed*.pt`, `checkpoints/xgboost/xgb_seed*.ubj`
- Optuna databases: `checkpoints/{lstm,xgboost}/optuna_2026-05-12.db`
- All multi-seed runs seeded identically (reproducible PyTorch + NumPy)
- Original `SPLIT_BOUNDARIES_2018_2024` retained for legacy rollback if needed

---

## Next Steps

1. **CNN-LSTM build** — implement 1D conv (kernel=3–5) feeding into LSTM
2. **xLSTM baseline** — Apple Silicon config (sLSTM + mLSTM), compare to CNN-LSTM on same F1 holdout
3. **Transformer** — if time permits; lightweight encoder-only architecture
4. **Model inspection tool** — Plotly overlay of rule zones vs model predictions on test set
5. **Final presentation** — June 17 (demo + slides)

---

## Files Modified/Added

- `.gitignore` — allow checkpoint + parquet tracking
- `data/raw/spy_minute.parquet` — new 60MB 2016–2025 history
- `data/processed/spy_h1*.parquet` — regenerated with new splits
- `src/data/download.py`, `.split.py`, `.process.py`, `.normalize.py` — pipeline updates
- `src/rigor/` — 6 modules (seed sweep, Optuna, threshold, bootstrap, reporting)
- `scripts/rigor/` — 8 scripts (multiseed, tune, shap, asymmetry, threshold, bootstrap)
- `tests/rigor/` — 4 test modules
- `CLAUDE.md`, `README.md`, `docs/*` — synced for current state
- `.nb-suite/build/12-May-26/` — rigor sprint build logs + data-extension doc
- `.nb-suite/analysis/criterion-ablation-2026-05-12.csv` — focal-loss γ sweep results
- `.nb-suite/explanations/where-we-are.md` — project status summary

---

## Test Status

✅ `tests/data/test_split.py` — new splits validated (no lookahead, temporal integrity)
✅ `tests/rigor/` — bootstrap CI, Optuna utils, seed sweep, threshold sweep
✅ Baseline F1 scores replicated across 5 seeds (deterministic init)

---

**Session:** May 12, 2026 | **User:** Nikola Baburov | **Agent:** nb-git-me
