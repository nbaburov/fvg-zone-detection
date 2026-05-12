# Review: Part B ValidFVG Migration — spec + architecture — 2026-05-12

REVIEW DOMAIN: code + docs
REVIEW PASS: spec, quality, tests
RESULT: PASS (with 2 minor issues)
ISSUES: 2 total — 0 blocker / 0 major / 2 minor / 0 nit

## Issues

[MINOR] src/data/pipeline.py:63 — Docstring says "Persisted to {PROCESSED_DIR}/class_weights.json" but B2 now writes TWO files (suffixed + legacy). Docstring is partially stale. → Update to: "Persisted to {PROCESSED_DIR}/class_weights_{labeller_name}.json (and legacy alias class_weights.json)."

[MINOR] tests/data/test_pipeline_default.py:8 — `import pytest` is present but unused (no `pytest.mark`, `pytest.raises`, or fixtures used). → Remove unused import to keep test file clean and avoid linter warnings.

## Spec compliance — all B1–B8 requirements verified

| Requirement | Status |
|-------------|--------|
| B1: pipeline default → "fvg_valid" | PASS — line 51 confirmed |
| B2: dual class_weights write | PASS — suffixed + legacy both written |
| B3: 5× labeller_name "fvg" → "fvg_valid" in test_pipeline.py | PASS — replace_all confirmed |
| B4: FVGLabeller deprecation header | PASS — 8-line header with rationale |
| B5: spy_h1_labeled.parquet deleted; persist_labels.py deprecated | PASS — file removed; stub returns 0 |
| B6: CLAUDE.md updated (canonical target, arch tree, constraints) | PASS — 3 sections updated |
| B7: docs/models-status.md header + dual-FVG section + historical section | PASS — all three present |
| B8: archive annotation prepended to "Key findings" | PASS — annotation present |
| Critical bug fix: build_rawfvg_splits.py output_path isolation | PASS — writes to rawfvg/ subdir; canonical spy_h1.parquet untouched |

## Key questions answered

1. **Dual class_weights write — correct?** Yes. B2 iterates over (suffixed_path, legacy_path) and writes the same dict to both. Any consumer reading `class_weights.json` gets ValidFVG weights. Any new consumer reading `class_weights_fvg_valid.json` also gets them. No divergence possible.

2. **FVGLabeller deprecation adequate?** Yes. Header comment is specific: states why kept (rawfvg checkpoint compatibility), what not to do (no new training runs), and where to find authoritative numbers. `@register("fvg")` and `@register("fvg_raw")` remain functional — inspect tool can still load rawfvg checkpoints.

3. **Legacy spy_h1_labeled.parquet references — silent failures?** All three active references (`train_lstm.py:127`, `train_xgboost.py:132`, `naive_baselines.py:62`) are gated behind `if splits == "legacy":`. They will `FileNotFoundError` if `--splits legacy` is passed — but this is correct and expected: the legacy mode is explicitly retired. The help text for `--splits` still mentions the legacy path in prose, which is now misleading, but this is a minor cosmetic issue, not a breakage risk.

4. **New test assertions strong enough?** Yes for regression detection. `test_default_labeller_is_fvg_valid` catches any accidental revert of the default via signature inspection — not just string comparison. `test_fvg_valid_labeller_positive_rate_is_sparse` catches wrong labeller wiring (actual rate=0.000% vs threshold<10% — loose but safe since FVGLabeller on same input would produce ~25%). The loose threshold (<10%) is intentional: it avoids fragility from tiny geometry differences in the synthetic frame.

## UNVERIFIED

- `docs/models-status.md` dual-FVG table numbers: reviewed as verbatim copy from `reports/rigor/2026-05-13/baselines/dual_fvg_compare.md` — assumed correct, not re-derived from raw meta.json.
- `--splits legacy` help text still mentions `spy_h1_labeled.parquet` by name in train_lstm.py and train_xgboost.py. These were not in Part B scope — flagged for optional cleanup in Phase D.

APPROVE: yes
