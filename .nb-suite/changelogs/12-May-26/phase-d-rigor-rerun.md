# Changelog — Phase D (10-Gap Rigor Rerun) — 12-May-26

| SHA | Type | Subject |
|-----|------|---------|
| 6070a92 | feat(rigor) | complete 10-gap rigor rerun on ValidFVG (py3.12+CPU stack) |

## Files (99 total)
- Source: `scripts/rigor/{multiseed_run,shap_xgb,threshold_sweep,tune_xgboost,window_sweep,bootstrap_ci_multiseed,threshold_multiseed}.py`
- Workers: `scripts/rigor/_workers/{_shap,_xgb_sweep,_xgb_tune}_worker.py`, `scripts/_xgb_infer_worker.py`
- Core: `src/rigor/{seed_sweep,optuna_utils,report_utils}.py`
- Checkpoints: LSTM (canonical 5, focal 15, reg ablation 15) + XGB (canonical 5) + 4 Optuna DBs
- Build log: `.nb-suite/build/12-May-26/rerun-sprint-build.md`

## Summary
Closed all 10 rigor gaps on canonical ValidFVG target. XGB 0.721 ± 0.001 dominates LSTM 0.599 ± 0.025. Focal loss and threshold tuning both worse than baseline. Reg helps marginally. W=90 best window. Fixed reg-ablation checkpoint cache bug in `seed_sweep.py`.
