# Build Log — Post-Research Rerun Sprint
**Date:** 12-May-26
**Phase:** B — Close XGB test gap (subprocess fix)

---

## Done

- Rewrote `tests/rigor/test_optuna_utils.py` to remove all 3 `@pytest.mark.skipif(_XGB_SKIP, ...)` decorators and the `_XGB_SKIP` variable.
- Added `_run_xgb_study_subprocess()` helper that:
  - Saves arrays as `.npz` to `tmp_path`
  - Launches `scripts/rigor/_workers/_xgb_tune_worker.py` in a fresh subprocess via `sys.executable`
  - Sets `PYTHONPATH=<project_root>` in the subprocess env (worker's `sys.path.insert` resolves to `scripts/rigor`, not project root)
  - Loads the resulting study from SQLite for assertion
- Removed `import platform`, `import sys` module-level torch-loading tests now use subprocess; replaced with `import os`, `import subprocess`, `import optuna`
- Removed `XGBObjective` from the `optuna_utils` import (no longer imported — XGB runs via subprocess + `optuna_xgb.py`)
- Updated module docstring in `test_optuna_utils.py` to explain subprocess pattern
- Updated `CLAUDE.md` MPS gotchas section: replaced old LSTM/MultiHeadAttention warnings with accurate constraints (CPU-only LSTM, subprocess XGB workers)
- Added `tests/models/conftest.py` with `pytest_collection_modifyitems` to order `test_xgboost_baseline.py` before `test_lstm.py` within that directory

## Good

- 6/6 tests in `tests/rigor/test_optuna_utils.py` pass, 0 skipped
- 235/235 tests pass when `test_xgboost_baseline.py` is excluded from the combined run
- `test_xgboost_baseline.py` passes 10/10 in isolation
- PYTHONPATH fix was needed: worker's ROOT path resolved to `scripts/rigor/` not project root

## Bad / Pre-existing

- `tests/models/test_xgboost_baseline.py` segfaults when run in the same pytest process as ANY torch test file — because pytest imports all test modules during collection (triggering `import torch`), then `clf.fit()` in XGB model tests segfaults due to the arm64 libgomp issue. This is pre-existing (test_lstm.py docstring says "Run separately"). The conftest.py ordering fix doesn't help because the issue is collection-time import, not execution order.
- This is a separate issue from Phase B scope. Phase B fix was `test_optuna_utils.py` — fully resolved.
- Running `pytest --ignore=tests/models/test_xgboost_baseline.py` gives 235 green. Running `test_xgboost_baseline.py` alone gives 10 green.

## Open flags

- `tests/models/test_xgboost_baseline.py` combined-suite segfault: needs subprocess isolation of `clf.fit()` in that test file (same pattern as the rigor fix) OR pytest-subprocess-worker plugin. Flagged for a follow-up fix before Phase E.
- `tests/models/conftest.py` was added but doesn't fully resolve the combined segfault — it's still useful for future if collection-time import is ever fixed.

## Next steps

- User to review and approve Phase B
- Phase C: train XGB seed42 + LSTM seed42 fresh baselines
