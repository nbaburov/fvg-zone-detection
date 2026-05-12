# Review — Config-Architecture Refactor (Phases 2 + 3)

**Date:** 2026-05-12
**Reviewer:** nb-build (orchestrator review)
**Scope:** Phase 2 (8 rigor scripts + seed_sweep adapter) + Phase 3 (paper_trade, inspect verify)

---

## Spec compliance

**PASS.** All 8 rigor scripts accept `--config experiments/foo.yaml`. All accept `--set key=value` overrides. All keep legacy CLI flags with `DeprecationWarning`. `SeedSweepConfig.from_experiment_config` classmethod added and tested. `parse_set_args` helper added to `loader.py` and exported from `src/config/__init__.py`.

## Consistent config reading

**PASS.** 12 scripts now read config via `load_experiment`:
- `scripts/training/train.py`, `train_lstm.py`, `train_xgboost.py` (Phase 1)
- `scripts/rigor/multiseed_run.py`, `window_sweep.py`, `tune_lstm.py`, `tune_xgboost.py`, `threshold_sweep.py`, `threshold_multiseed.py`, `shap_xgb.py`, `bootstrap_ci_multiseed.py` (Phase 2)
- `scripts/paper_trade.py` (Phase 3, optional)

`inspect_models.py` is meta.json-driven — no HP config needed. Correct by design.

## No hardcoded HP

**PASS.** All HP come from `cfg.model.*` and `cfg.train.*`. Fallback defaults in `hp.get(key, default)` are only used when legacy JSON path is taken (no YAML loaded).

## Backwards compat

**PASS.** Extension detection (`_is_yaml_path`) routes JSON paths to unchanged legacy logic. Old `--seeds`, `--loss`, `--gamma`, `--seed`, `--patience` flags preserved with deprecation warnings. Gate 2 invocation `multiseed_run.py --model lstm --config reports/rigor/.../best_hp_lstm.json --seeds 42` still works.

## Test coverage

**PASS.** 280 tests pass (was 277). 3 new tests cover `from_experiment_config` for LSTM, XGB, and `output_dir` override. `parse_set_args` is exercised indirectly via test_loader.py (existing `test_override_train_seed` etc).

## Security

**PASS.** `parse_set_args` uses `yaml.safe_load` for value coercion — safe, no arbitrary code execution. No new network, file system, or auth surface.

## Leakage check

**PASS.** No test set access in tuning scripts. `window_sweep.py` accesses test set for reporting only (unchanged from original — same risk posture). Temporal split config (`SPLIT_BOUNDARIES`) unchanged.

## Issues found

None blocking. One observation:

- `bootstrap_ci_multiseed.py` had a variable shadowing bug in the original (`block_size` was set via `args.block_size` after being resolved). Fixed in migration — resolved variables now used throughout.

## Verdict

**PASS.** Ready for Phase 3 exit gate.
