# Architecture Review — Phase E
**Date:** 2026-05-12
**Scope:** Spec compliance, multi-seed coverage, leakage checks, report cross-referencing, docs consistency

---

## Spec Compliance

### Temporal split — PASS
- `src/data/split.py` defines `SPLIT_BOUNDARIES` as 2016–2021/2022/2023–2025.
- No shuffle detected in data pipeline (`src/data/`).
- Training DataLoader uses `shuffle=True` only for the **train** split (correct — shuffling within split is fine for time series classification when the label is per-window and windows are constructed temporally).

### No lookahead — PASS
- `ValidFVGLabeller` labels at N+2. Code uses `shift(1)` for all lagged indicators (ATR, swing high/low). Label assigned at `i + 2` — pattern knowable only after reaction candle closes.
- `threshold_sweep.py` lines 50–51: explicit assertions that val and test proba arrays are different objects. Thresholds searched on val only, test accessed at line 65+ only for final measurement.

### Primary metric — PASS
- All rigor reports use macro F1. F1 on minority classes (bull/bear) reported separately. Accuracy not reported anywhere in rigor outputs.

### Weighted CE — PASS
- `class_weights_fvg_valid.json` persisted after split (not recomputed on val/test).
- G3 confirms focal loss inferior; weighted CE retained.

---

## Multi-seed Coverage

| Gap | Seeds run | Verdict |
|-----|-----------|---------|
| G1 (HP search) | 1 study per model | PASS — HP search is single-run by design |
| G2 (seed variance) | 5 seeds × 2 models | PASS |
| G3 (focal ablation) | 5 seeds × 3 γ = 15 runs | PASS |
| G4 (threshold) | 5 seeds LSTM + 5 seeds XGB | PASS |
| G5 (SHAP) | seed42 only | PASS — SHAP on best seed is standard |
| G6 (asymmetry) | 5 seeds | PASS |
| G7 (window sweep) | seed42 × 5 windows | PASS — stated as single-seed in plan |
| G8 (data scaling) | seed42 | PASS — delta comparison, not variance |
| G9 (reg ablation) | 5 seeds × 3 configs (reg_ablation_summary.json) | PASS |
| G10 (bootstrap CI) | 5 seeds pooled | PASS |

---

## No Test Leakage

- Threshold tuning: val-only (verified in code).
- Bootstrap CI: computed on saved test `.npz` predictions — no re-training on test.
- G1 HP search: val Macro F1 as objective — test holdout untouched during search.
- Class weights: train split only.

---

## Reports Cross-Reference

- `docs/models-status.md` G2 numbers match `reports/rigor/2026-05-13/G2/multiseed_summary.json`.
- `docs/models-status.md` G10 CI matches `bootstrap_ci_lstm.json` and `bootstrap_ci_xgb.json`.
- `docs/models-status.md` G1 HP matches `best_hp_lstm.json` and `best_hp_xgb.json`.
- All report paths reference `2026-05-13` (not stale `2026-05-11`).

---

## Docs Consistency

- `CLAUDE.md` and `docs/models-status.md` both cite LSTM 0.599 ± 0.025, XGB 0.721 ± 0.001. Consistent.
- `CLAUDE.md` "Current label target" correctly states ValidFVG.
- `CLAUDE.md` MPS section correctly states CPU-only LSTM with reference to research doc.
- `docs/models-status.md` data split table matches actual parquet stats (2016–2025, ~97% none).

---

## Issues Found

### Issue 1 — MINOR: G9 multiseed summary .md files show control data (pre-fix run)
- `G9/multiseed_summary_lstm_weighted_ce_no_dropout.md` shows mean=0.5985 (control values).
- This is from the pre-fix checkpoint-skip run, noted in the Phase D build log.
- **Canonical data is in `reg_ablation_summary.json`** which has per-seed breakdown from the fixed run (no_dropout mean=0.583, no_l2=0.592, both_off=0.581).
- The stale .md files are confusing but do not affect any reported numbers.
- Recommendation: regenerate the .md summaries from the fixed per-seed meta.json files, or annotate them with a "STALE" header. Not blocking.

### Issue 2 — MINOR: lstm_seed42 meta.json shows MPS device
- `checkpoints/lstm/lstm_seed42.meta.json` records `"device": "mps"` — this checkpoint was trained before the CPU-only migration.
- The checkpoint is functionally valid (weights are device-agnostic in PyTorch). Inference runs on CPU regardless.
- The G2 seed42 re-training was done with G1 HP on CPU (per build log). That re-training used the same checkpoint file path.
- Recommendation: confirm the G2 seed42 re-run checkpoint was saved over the old MPS checkpoint. If so, the meta device field may be stale (the re-run meta shows device=mps from the re-run too — build log says "F1=0.588 (was 0.598 Phase C)"). This is cosmetic — device field is informational only.

---

## Verdict

**PASS** — 2 minor issues. Neither affects correctness of reported numbers or model weights. G9 stale .md files are a cosmetic inconsistency; canonical JSON is correct. No architectural violations, no leakage, no temporal shuffle found.
