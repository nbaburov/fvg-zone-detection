# Test Log: Part B ValidFVG Migration — 2026-05-12

## Result
PASS — 251 passed / 0 failed / 0 skipped

## Coverage added
- `tests/data/test_pipeline.py`: 5 occurrences of `labeller_name="fvg"` updated to `"fvg_valid"` — pipeline integration tests now exercise canonical labeller
- `tests/data/test_pipeline_default.py` (new, 2 tests):
  - `test_default_labeller_is_fvg_valid`: asserts `build_pipeline` default is `"fvg_valid"` via signature inspection
  - `test_fvg_valid_labeller_positive_rate_is_sparse`: asserts ValidFVG positive rate < 10% on flat series (guards against wrong labeller wired under "fvg_valid")

## Failures fixed
None — all 251 tests passed on first run after migration.

## Bugs found
None.

## Flagged
None — all planned coverage was achievable.

## Next steps
- Phase D rigor rerun on ValidFVG only (seed variance, HP tuning, threshold, bootstrap CI)
