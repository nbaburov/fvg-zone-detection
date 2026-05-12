# Phase 1 — Gap 1: Hyperparameter Search

**Date:** 12-May-26
**Status:** Scripts built, ready to run overnight

## Files created

- `scripts/tune_lstm.py` — LSTM Optuna search (50 trials, train+val only)
- `scripts/tune_xgboost.py` — XGB Optuna search via subprocess (Python 3.14 workaround)
- `scripts/multiseed_run.py` — Multi-seed sweep + focal ablation + reg ablation
- `scripts/threshold_sweep.py` — Per-class PR curve + threshold tuning
- `scripts/shap_xgb.py` — SHAP feature importance + pruned retrain
- `scripts/window_sweep.py` — Window size sensitivity (W=30,60,90,120)
- `scripts/asymmetry_analysis.py` — Bull/bear asymmetry investigation
- `scripts/bootstrap_ci.py` — Block bootstrap 95% CI

## XGBoost subprocess pattern

Python 3.14 segfaults XGBoost in-process. All XGB scripts use subprocess.
Workers written inline at runtime: `_xgb_tune_worker.py`, `_xgb_sweep_worker.py`, `_xgb_infer_worker.py`.

## Run commands

```bash
# Phase 1a: LSTM tuning (run overnight, ~4-6 hrs)
.venv/bin/python scripts/tune_lstm.py --n-trials 50 --device auto

# Phase 1b: XGB tuning (fast, ~30 min)
.venv/bin/python scripts/tune_xgboost.py --n-trials 50
```

## Decision gate

After Phase 1: inspect best_lstm_config.json and best_xgb_config.json.
Check that winning config is NOT at search-space edge.
If edge: expand space and rerun. Otherwise proceed to Phase 2.
