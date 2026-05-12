# Changelog — Phase B (Stack Fix) — 12-May-26

| SHA | Type | Subject |
|-----|------|---------|
| 4f64f03 | fix(stack) | py3.12 + CPU-only LSTM + isolated XGB tests for rerun |

## Files
- `scripts/training/train_lstm.py` — `select_device` returns CPU (MPS warning + fallback)
- `src/rigor/optuna_utils.py` — `run_study` adds `timeout=300, n_jobs=1`
- `tests/rigor/test_optuna_utils.py` — 3 XGB skips removed via subprocess helper `_run_xgb_study_subprocess`
- `tests/models/test_xgboost_baseline.py` — docstring updated, behind `pytest.ini --ignore`
- `pytest.ini` (new) — default `pytest` ignores XGB baseline file
- `Makefile` (new) — `make test` runs both halves in separate pytest invocations
- `CLAUDE.md` — MPS gotchas section synced for py3.12 + CPU-only LSTM + XGB test isolation
- `.nb-suite/build/12-May-26/rerun-sprint-build.md` (new) — Phase B build log

## Summary
Phase B closed: Python 3.12 stack live, MPS LSTM deadlock bypassed via forced CPU device, all rigor tests passing (245 total, 0 skipped) under `make test`.
