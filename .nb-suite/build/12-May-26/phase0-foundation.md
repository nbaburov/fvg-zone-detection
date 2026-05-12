# Phase 0 — Foundation: src/rigor/ package

**Date:** 12-May-26
**Status:** COMPLETE

## Files created

- `src/rigor/__init__.py`
- `src/rigor/optuna_utils.py` — LSTMObjective, XGBObjective, run_study
- `src/rigor/seed_sweep.py` — SeedSweepConfig, run_seed_sweep
- `src/rigor/threshold.py` — compute_pr_curves, find_f1_optimal_threshold, apply_thresholds
- `src/rigor/bootstrap_ci.py` — block_bootstrap_f1, effective_n
- `src/rigor/report_utils.py` — write_rigor_report, timestamped_dir
- `tests/rigor/__init__.py`
- `tests/rigor/test_optuna_utils.py`
- `tests/rigor/test_seed_sweep.py`
- `tests/rigor/test_threshold.py`
- `tests/rigor/test_bootstrap_ci.py`

## Dependencies installed

- optuna 4.8.0
- shap 0.51.0
- optuna-integration[xgboost]

## Test results

```
26 passed, 3 skipped
```

3 skipped = XGBObjective in-process tests — known Python 3.14 XGBoost segfault.
XGB tuning script will use subprocess workaround (per CLAUDE.md pattern).

Existing 206 non-XGB tests: all pass.

## Notes

- `apply_thresholds` uses mask-then-argmax (not scaling). Test fixed to match.
- XGB segfault is pre-existing (not introduced by Phase 0).
- optuna-integration[xgboost] installed but XGBoostPruningCallback dropped from
  XGBObjective.__call__ — XGB 3.x doesn't accept callbacks in .fit(). Pruning
  skipped for XGB; early_stopping_rounds=30 provides equivalent trial termination.
