# Changelog — Config Refactor Phase 2 + 3 — 13-May-26

| SHA | Type | Subject |
|-----|------|---------|
| c3819ce | feat(config) | migrate rigor + paper_trade scripts to YAML-driven config |

## Files
- `src/rigor/seed_sweep.py` — `from_experiment_config()` adapter
- `src/config/loader.py`, `src/config/__init__.py` — `parse_set_args()` helper exported
- 8 rigor scripts under `scripts/rigor/` — `--config` + `--set` support, legacy CLI preserved
- `scripts/paper_trade.py` — optional `--config` flag
- `tests/rigor/test_seed_sweep.py` — 3 new tests
- nb-suite build log + review log + Phase 1 changelog

## Summary
Phase 2 + 3 closed: 8 rigor scripts + paper_trade now YAML-driven via `--config experiments/foo.yaml --set key=value`. Legacy CLI paths preserved. `bootstrap_ci_multiseed.py` block_size shadow bug fixed opportunistically. 280/280 tests pass. nb-review pass.
