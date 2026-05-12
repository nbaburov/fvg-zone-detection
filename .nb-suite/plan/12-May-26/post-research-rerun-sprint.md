# Plan: Post-Research Rerun Sprint

**Date:** 12-May-26
**Type:** Multi-phase sprint — cleanup + stack migration + fresh baselines + full rigor rerun
**Feeds:** @nb-build directly. Each phase has a decision gate before proceeding.
**Prior art read:** mps-gpu-fix.md, archive snapshot, sprint-complete build log, original rigor plan, models-status.md, CLAUDE.md

---

## Sprint Goal

End state:
- Python 3.12 venv, CPU-only LSTM, no hanging processes
- Fresh XGB + LSTM baselines on 2016–2025 data (raw FVG labels, N+1)
- All 10 rigor gaps closed with no partials — full 5-seed coverage everywhere
- `docs/models-status.md` reflects reality (all stale numbers replaced)
- `checkpoints/` and `reports/rigor/` contain only 2026-05-13 outputs
- Ready for Status Update 1 (May 17) and CNN-LSTM phase

**What this is NOT:** ValidFVG training, CNN-LSTM, paper trading, 2025 forward test. Those are separate phases.

---

## Executor assignment

| Phase | Owner | Subagents spawned |
|-------|-------|-------------------|
| A — Cleanup | @nb-build | none |
| B — Stack migration | @nb-build | @nb-test (smoke suite) |
| C — Fresh baselines | @nb-build | @nb-test (reproducibility) |
| D — Full rigor rerun | @nb-build | @nb-test (gap validation) |
| E — Inspect + docs | @nb-build | @nb-review (docs quality gate) |

---

## Decision gates

```
Phase A complete?
  → git log shows archive commit + cleanup commit, no stale files → proceed to B

Phase B complete?
  → pytest 0 skipped, 0 failures, CPU smoke test passes → proceed to C

Phase C complete?
  → seed42 XGB run twice = identical F1; seed42 LSTM run twice = identical F1;
    data scaling Δ measured and recorded → proceed to D

Phase D complete?
  → all 10 gap JSON outputs exist under reports/rigor/2026-05-13/;
    multi-seed std computed from 5 seeds (not 2); window sweep 5 points → proceed to E

Phase E complete?
  → docs/models-status.md numbers match reports; no stale "2026-05-11" paths remain
```

---

## Phase A — Cleanup

**Goal:** commit the archive doc, delete all stale artifacts in one commit, verify repo is clean.

**Estimated wall-clock:** 10 minutes

### Steps

1. Stage and commit `.nb-suite/archive/2026-05-12-pre-rerun-snapshot.md`:
   ```
   git add .nb-suite/archive/2026-05-12-pre-rerun-snapshot.md
   git commit -m "docs(archive): snapshot pre-rerun state with preserved findings"
   ```

2. Delete the following (all gitignored or will be replaced):
   - `checkpoints/lstm/*.pt`
   - `checkpoints/lstm/*.meta.json`
   - `checkpoints/lstm/optuna_*.db`
   - `checkpoints/xgboost/*.ubj`
   - `checkpoints/xgboost/*.ubj.metadata.json` (if present)
   - `checkpoints/xgboost/optuna_*.db`
   - `reports/rigor/2026-05-11_*/` (all subdirs — ~75 MB)
   - `reports/rigor/v2_2016_2025/2026-05-12_*/` (partial new-data run)
   - `reports/inspect/2026-05-11_*/`

3. Commit:
   ```
   git add -u
   git commit -m "chore(checkpoints): wipe stale artifacts pre-rerun (see archive doc)"
   ```

4. Do NOT delete: `data/raw/`, `data/processed/`, `data/gold_labels.csv`, all `.nb-suite/` logs.

### Test strategy
- `git status` shows clean. `ls checkpoints/lstm/` is empty. `ls reports/rigor/` contains no `2026-05-11` dirs.

---

## Phase B — Stack migration to Python 3.12

**Goal:** working Python 3.12 venv, CPU-forced LSTM trainer, Optuna timeout guard, 0 skipped tests.

**Estimated wall-clock:** 45–60 minutes (install + smoke runs)

### Steps

#### B1 — Create Python 3.12 venv
```bash
brew install python@3.12
# verify
python3.12 --version  # expect 3.12.x

deactivate  # if currently in 3.14 venv
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install xgboost numpy pandas scikit-learn optuna optuna-integration[xgboost] shap alpaca-py exchange_calendars plotly pytest
```

**Note on torch install:** Use CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling the full CUDA/MPS build. Saves ~2 GB and removes MPS device from availability check entirely.

Verify:
```python
import torch; print(torch.__version__, torch.backends.mps.is_available())
# expect: 2.x.x False  (MPS not available = no risk of accidental device selection)
import xgboost; print(xgboost.__version__)
import numpy, shap; print(numpy.__version__, shap.__version__)
```

#### B2 — Force CPU in LSTM trainer

File: `scripts/training/train_lstm.py`

Current `select_device` (lines 85–90) returns `mps` when available. Change to:
```python
def select_device(requested: str) -> torch.device:
    # MPS LSTM gradient kernel is broken (see .nb-suite/research/12-May-26/mps-gpu-fix.md)
    # CPU is the only reliable device for multi-layer LSTM backprop.
    if requested == "mps":
        print("WARNING: MPS requested but LSTM training is CPU-only. Using CPU.")
    return torch.device("cpu")
```

This hardcodes CPU regardless of CLI arg. The `--device` arg is kept in the argparse interface for future re-enablement but overridden in logic.

Also update the metadata logger to record `"device": "cpu (forced — MPS LSTM bug)"` so `.meta.json` files are honest.

#### B3 — Add Optuna timeout guard

File: `src/rigor/optuna_utils.py`

Current `run_study` calls `study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)` with no timeout (line 233).

Change to:
```python
study.optimize(
    objective,
    n_trials=n_remaining,
    show_progress_bar=True,
    timeout=300,          # 5 min per trial — prevents infinite MPS/CPU hang
    n_jobs=1,             # always serial (MPS thread-safety + XGB subprocess safety)
)
```

Also add `timeout=300` as a parameter to `run_study` signature with default 300.

#### B4 — Restore 3 skipped XGB rigor tests

Locate skipped tests:
```bash
pytest tests/rigor/ -v 2>&1 | grep SKIP
```

The 3 skipped tests were guarded with `pytest.mark.skip` due to Python 3.14 XGB subprocess issues. On Python 3.12, the subprocess segfault risk is eliminated. Remove the skip decorators. Confirm the tests pass.

#### B5 — Smoke tests

Run these manually to verify the stack before committing to long Phase C runs:

```bash
# 1. CPU LSTM forward + backward (should complete in <30s)
python - <<'EOF'
import torch, torch.nn as nn
device = torch.device("cpu")
model = nn.LSTM(input_size=5, hidden_size=64, num_layers=3, dropout=0.3, batch_first=True).to(device)
x = torch.randn(32, 60, 5, device=device)
out, _ = model(x)
loss = out.mean()
loss.backward()
print("LSTM CPU smoke: PASSED")
EOF

# 2. XGB minimal train (should complete in <10s)
python - <<'EOF'
import xgboost as xgb, numpy as np
X = np.random.randn(500, 35)
y = np.random.randint(0, 3, 500)
dtrain = xgb.DMatrix(X, label=y)
bst = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 3}, dtrain, num_boost_round=10)
print("XGB smoke: PASSED")
EOF

# 3. Full test suite
pytest --tb=short
```

**Decision gate for Phase B:** pytest shows 0 skipped, 0 failures. Both smoke tests print PASSED.

#### B6 — Update CLAUDE.md

Remove from `CLAUDE.md` the existing "MPS gotchas" bullet that says "nn.LSTMCell slower than CPU → use sequence-batched nn.LSTM". Replace the entire MPS gotchas section with:

```markdown
**Device:** LSTM training is CPU-only (MPS LSTM gradient kernel bug — unfixed as of torch 2.11, see `.nb-suite/research/12-May-26/mps-gpu-fix.md`). XGBoost also CPU. Python 3.12 stack (3.14 caused XGB subprocess segfaults).
```

Keep the MPS gotchas that apply to future CNN-LSTM and Transformer work:
- `nn.MultiheadAttention` + bool mask + dropout = NaN on MPS (keep, relevant for Transformer)
- Deterministic mode = 8× slowdown (keep)
- xLSTM Apple Silicon config (keep)

### Commit for Phase B
```
git add scripts/training/train_lstm.py src/rigor/optuna_utils.py tests/rigor/ CLAUDE.md
git commit -m "fix(training): force CPU device for LSTM; add Optuna timeout guard; restore XGB tests on py3.12"
```

---

## Phase C — Fresh baselines on 2016–2025

**Goal:** reproducible seed42 baselines for both models on new data; naive baseline; data scaling Δ measured.

**Estimated wall-clock:** 2–4 hours (XGB fast, LSTM ~1.5 hrs per run × 2 for reproducibility check)

### C1 — Verify data is already built

```bash
ls data/processed/spy_h1_train.parquet
ls data/processed/spy_h1_val.parquet
ls data/processed/spy_h1_test.parquet
ls data/processed/class_weights.json
```

If any file is missing, run:
```bash
python -c "from src.data.pipeline import build_pipeline; build_pipeline()"
```

The 2016–2025 data was already downloaded and processed in the previous sprint. It should still be on disk.

### C2 — XGB seed42 baseline

```bash
python scripts/training/train_xgboost.py --seed 42 --output-dir checkpoints/xgboost
```

Record F1 from `.meta.json`. Run a second time with a fresh output path to verify identical result:
```bash
python scripts/training/train_xgboost.py --seed 42 --output-dir checkpoints/xgboost/repro_check
diff <(python -c "import json; d=json.load(open('checkpoints/xgboost/xgb_seed42.meta.json')); print(d['test_macro_f1'])") \
     <(python -c "import json; d=json.load(open('checkpoints/xgboost/repro_check/xgb_seed42.meta.json')); print(d['test_macro_f1'])")
# Should print nothing (identical)
```

Delete the repro_check dir after confirming.

### C3 — LSTM seed42 baseline (CPU)

```bash
python scripts/training/train_lstm.py --seed 42 --device cpu --output-dir checkpoints/lstm
```

Record wall-clock (use `time` prefix). This establishes how long each seed takes on CPU — critical for Phase D planning. Expected: 45–90 min for 50–100 epochs with patience=15.

Run reproducibility check:
```bash
python scripts/training/train_lstm.py --seed 42 --device cpu --output-dir checkpoints/lstm/repro_check
```

Both runs must produce identical `test_macro_f1`. If they differ by >1e-4, investigate — deterministic seeding may need `torch.use_deterministic_algorithms(True)` for CPU.

Delete repro_check after confirming.

### C4 — Naive baseline

Write a one-off script or run inline to compute majority-class and uniform-random F1 on the test split. This is a 5-minute task:

```python
import pandas as pd, numpy as np
from sklearn.metrics import f1_score
from src.data.split import load_test_split  # or load parquet directly

y_test = ...  # load from spy_h1_test.parquet["label"].values
majority = np.zeros(len(y_test), dtype=int)
uniform = np.random.RandomState(42).randint(0, 3, len(y_test))
print("majority F1:", f1_score(y_test, majority, average="macro", zero_division=0))
print("uniform F1:", f1_score(y_test, uniform, average="macro", zero_division=0))
```

Record both numbers for `docs/models-status.md`.

### C5 — Data scaling Δ

This gap was skipped in the previous sprint. Train seed42 XGB and LSTM using the OLD split boundaries (`SPLIT_BOUNDARIES_2018_2024`) from `src/data/split.py` and compare to the new 2016–2025 baseline. Use the exact same hyperparams.

```bash
python scripts/training/train_xgboost.py --seed 42 --splits legacy --output-dir checkpoints/xgboost/legacy_splits
python scripts/training/train_lstm.py --seed 42 --device cpu --splits legacy --output-dir checkpoints/lstm/legacy_splits
```

If `--splits legacy` doesn't exist as a CLI arg, add it. It should switch `build_pipeline()` to use `SPLIT_BOUNDARIES_2018_2024` instead of the default. Alternatively, temporarily swap the constant in `src/data/split.py` and rerun (less clean — prefer CLI arg).

Report Δ = new_f1 − old_f1 for each model. Store in `reports/rigor/2026-05-13/data_scaling_delta.json`.

**Decision gate for Phase C:**
- `checkpoints/lstm/lstm_seed42.pt` and `checkpoints/xgboost/xgb_seed42.ubj` exist with fresh `.meta.json`
- Reproducibility confirmed (two runs identical)
- Naive baseline numbers recorded
- Data scaling Δ JSON written

---

## Phase D — Full rigor rerun (all 10 gaps, no partials)

**Goal:** every gap produces complete multi-seed results under `reports/rigor/2026-05-13/`.

**Estimated wall-clock:**
- G1 LSTM HP (50 trials × ~60 epochs, CPU): **4–8 hrs** → run overnight
- G1 XGB HP (50 trials): **30–45 min**
- G2 Multi-seed LSTM WeightedCE (5 seeds × ~80 epochs): **4–7 hrs** → run overnight
- G2 Multi-seed XGB (5 seeds): **10–15 min**
- G3 Focal ablation (15 LSTM runs): **8–14 hrs** → run overnight
- G4 Threshold: **5 min**
- G5 SHAP: **10 min**
- G6 Asymmetry: **20 min**
- G7 Window sweep (5 points, 1 seed each): **4–6 hrs**
- G8 Data scaling Δ: done in Phase C
- G9 Reg ablation (4 configs × 5 seeds): **5–8 hrs**
- G10 Bootstrap CI: **5 min**

**Total compute:** ~30–45 hrs. All compute-heavy gaps run overnight. Order: run G1 XGB first (30 min), then kick off G1 LSTM overnight. Phase D spans 2–3 days of overnight runs.

All outputs go to `reports/rigor/2026-05-13/<gap>/`. Each gap dir gets its own subdirectory (not timestamp-based — use gap label for clarity).

Output structure:
```
reports/rigor/2026-05-13/
  g1_hp_lstm/           best_lstm_config.json, optuna_summary.html
  g1_hp_xgb/            best_xgb_config.json, optuna_summary.html
  g2_multiseed/         multiseed_lstm.md, multiseed_xgb.md, *_preds.npz
  g3_focal/             focal_g{1,2,3}_summary.md
  g4_threshold/         thresholds_lstm.json, pr_curves.html
  g5_shap/              shap_summary.html, shap_report.md
  g6_asymmetry/         asymmetry.md, confusion_*.html
  g7_window_sweep/      window_sweep.json, window_sweep.html
  g8_data_scaling/      → already written in Phase C as data_scaling_delta.json (move/copy here)
  g9_reg_ablation/      reg_ablation.md (5 seeds × 4 configs)
  g10_bootstrap_ci/     bootstrap_ci_lstm.json
```

### G1 — HP Search (50/50 trials)

**LSTM:** `python scripts/training/tune_lstm.py --n-trials 50 --output-dir reports/rigor/2026-05-13/g1_hp_lstm`

LSTM search space must **exclude `num_layers >= 2`** since CPU training with large num_layers is slow and the prior sprint confirmed `num_layers=1` wins consistently on this dataset. Constrain to:
- `num_layers`: categorical {1} (confirmed winner — don't burn 50 trials on dead configs)
- All other params as before

This halves the search space variance and allows 50 meaningful trials in reasonable time on CPU.

**XGB:** `python scripts/training/tune_xgboost.py --n-trials 50 --output-dir reports/rigor/2026-05-13/g1_hp_xgb`

New Optuna DBs go to `checkpoints/lstm/optuna_2026-05-13.db` and `checkpoints/xgboost/optuna_2026-05-13.db`. Old DBs from 2026-05-12 are already deleted (Phase A).

**Decision gate G1:** `best_lstm_config.json` and `best_xgb_config.json` exist, 50 trials each in DB, no trial in PRUNED state due to timeout (if many timeouts, reduce trial complexity or increase timeout).

### G2 — Multi-seed (5 seeds: 0, 17, 42, 123, 2024)

Depends on G1 configs.

```bash
python scripts/multiseed_run.py --model lstm --config reports/rigor/2026-05-13/g1_hp_lstm/best_lstm_config.json \
  --seeds 0 17 42 123 2024 --loss weighted_ce --output-dir reports/rigor/2026-05-13/g2_multiseed

python scripts/multiseed_run.py --model xgboost --config reports/rigor/2026-05-13/g1_hp_xgb/best_xgb_config.json \
  --seeds 0 17 42 123 2024 --loss weighted_ce --output-dir reports/rigor/2026-05-13/g2_multiseed
```

Each seed saves `<model>_seed<N>_preds.npz` (y_true + y_pred arrays) to the output dir for G10.

**Decision gate G2:** 5+5 checkpoints in `checkpoints/lstm/` and `checkpoints/xgboost/`. All 5 seeds present in summary table with no NaN. Std computed from 5 seeds.

### G3 — Focal ablation (full 5 seeds × γ ∈ {1, 2, 3})

15 LSTM runs. Depends on G1 LSTM config. XGB focal not applicable (XGB uses sample_weight, not loss function change).

```bash
for gamma in 1 2 3; do
  python scripts/multiseed_run.py --model lstm \
    --config reports/rigor/2026-05-13/g1_hp_lstm/best_lstm_config.json \
    --seeds 0 17 42 123 2024 --loss focal --gamma $gamma \
    --output-dir reports/rigor/2026-05-13/g3_focal
done
```

**Critical:** `FocalLoss` in `src/training/loss.py` must receive `alpha` from `class_weights.json` (not uniform). Verify before running — this was a pre-mortem risk in the original plan.

**Decision gate G3:** 15 checkpoint files. Summary table covers all 3 gammas × 5 seeds. No seeds hung (CPU should prevent this).

### G4 — Threshold tuning

Depends on G2 multi-seed LSTM.

Select the seed with test macro F1 nearest to the G2 mean. Run:
```bash
python scripts/threshold_sweep.py \
  --model-path checkpoints/lstm/lstm_seed<nearest_mean_seed>.pt \
  --model-type lstm \
  --output-dir reports/rigor/2026-05-13/g4_threshold
```

Val set used for threshold selection. Test set touched once for delta measurement — not during search.

**Decision gate G4:** `thresholds_lstm.json` exists. Delta printed to stdout and captured in report.

### G5 — SHAP on XGB

Depends on G2 XGB checkpoints. Run on the Optuna-tuned seed42 checkpoint (NOT pre-tuning baseline).

```bash
python scripts/shap_xgb.py \
  --checkpoint checkpoints/xgboost/xgb_seed42.ubj \
  --config reports/rigor/2026-05-13/g1_hp_xgb/best_xgb_config.json \
  --output-dir reports/rigor/2026-05-13/g5_shap
```

**Decision gate G5:** `shap_summary.html` and `shap_report.md` exist.

### G6 — Asymmetry

Depends on G2 multi-seed LSTM preds (all 5 seeds).

```bash
python scripts/asymmetry_analysis.py \
  --preds-dir reports/rigor/2026-05-13/g2_multiseed \
  --model lstm \
  --output-dir reports/rigor/2026-05-13/g6_asymmetry
```

Multi-seed std used (not single-seed). Report must include: per-class F1 table with mean±std, confusion heat-maps, structural gap size comparison, conclusion.

**Decision gate G6:** `asymmetry.md` has all three sections. std computed from 5 seeds (check the table has 5 rows + mean row).

### G7 — Window sweep (W ∈ {30, 45, 60, 90, 120})

Added W=45 to the original {30, 60, 90, 120} for finer resolution around the optimal range. Previous sprint found W=30 and W=60 completed in 15–20 epochs. Use seed 42 only, CPU, reduced patience=7 (from default 15) to speed up sweeping.

```bash
python scripts/window_sweep.py \
  --windows 30 45 60 90 120 \
  --config reports/rigor/2026-05-13/g1_hp_lstm/best_lstm_config.json \
  --seed 42 \
  --patience 7 \
  --device cpu \
  --output-dir reports/rigor/2026-05-13/g7_window_sweep
```

`window_sweep.py` must pass `device=cpu` explicitly. It must also report `effective_n` (test_bars // W) alongside F1 so the comparison is honest (different W = different sample counts).

**Decision gate G7:** `window_sweep.json` has 5 rows (one per W). HTML plot renders. Note in report: reduced patience=7 used for sweep; final model uses patience=15.

### G8 — Data scaling Δ

Already completed in Phase C. Copy `data_scaling_delta.json` to `reports/rigor/2026-05-13/g8_data_scaling/`.

### G9 — Reg ablation (multi-seed, 5 seeds × 4 configs)

Previous sprint used single seed (seed=999). Rerun with full 5 seeds.

4 configurations × 5 seeds = 20 LSTM runs:
1. Control (best config from G1)
2. No dropout (`dropout=0, head_dropout=0`)
3. No L2 (`weight_decay=0`)
4. No reg (both off)

```bash
python scripts/reg_ablation.py \
  --config reports/rigor/2026-05-13/g1_hp_lstm/best_lstm_config.json \
  --seeds 0 17 42 123 2024 \
  --device cpu \
  --output-dir reports/rigor/2026-05-13/g9_reg_ablation
```

`reg_ablation.py` is a new thin wrapper around `multiseed_run.py` that overrides dropout/weight_decay fields before passing to `SeedSweepConfig`. If it doesn't exist as a separate script, extend `multiseed_run.py` with `--ablate-reg {none,no_dropout,no_l2,no_reg}` flag.

Report table: 4 configs × mean±std macro F1 (computed from 5 seeds). Single-seed deltas from previous sprint are not used.

**Decision gate G9:** `reg_ablation.md` has 4-row table. Each row has mean±std from 5 seeds.

### G10 — Bootstrap CI (CI of mean across seeds, not within one seed)

Previous sprint computed CI from single-seed predictions. Correct approach: compute CI of the multi-seed mean.

Method (block bootstrap, overlap-aware effective N):
1. Stack predictions from all 5 G2 WeightedCE seeds.
2. For each bootstrap draw: resample blocks of size 60 from the test sequence. Apply to each seed's predictions. Compute macro F1 per seed per bootstrap draw. Take mean across seeds.
3. 2.5/97.5 percentiles of the 1000 bootstrap means = CI.

This requires a change to `bootstrap_ci.py` or a new `--multi-seed` mode. The current implementation takes a single npz file.

**Effective N calculation fix:** current `effective_n` uses `n_bars // window_size` which ignores stride=1 overlap. The correct overlap-aware formula:

```python
def effective_n(n_bars: int, window_size: int, stride: int = 1) -> int:
    """Non-overlapping window count — honest effective sample size."""
    return n_bars // window_size  # only non-overlapping windows count as independent
```

This is already the formula used (it does the right thing — stride doesn't change independence count). The existing `effective_n(3514, 60, 1) == 58` is correct. Confirm the bootstrap_ci.py docstring says so explicitly.

```bash
python scripts/bootstrap_ci.py \
  --preds-dir reports/rigor/2026-05-13/g2_multiseed \
  --model lstm \
  --mode multi-seed \
  --block-size 60 \
  --n-iter 1000 \
  --seed 42 \
  --output-dir reports/rigor/2026-05-13/g10_bootstrap_ci
```

**Decision gate G10:** `bootstrap_ci_lstm.json` has `{"method": "ci_of_seed_mean", "n_seeds": 5, ...}`. CI lower < point < CI upper.

---

## Phase E — Inspect + docs

**Goal:** visual overlays on new checkpoints; `docs/models-status.md` fully updated; CLAUDE.md patched; feedpulse ready for May 17.

**Estimated wall-clock:** 1–2 hours

### E1 — Rerun inspector

```bash
python scripts/inspect_models.py \
  --models lstm xgboost \
  --lookahead-bars 20 \
  --output-dir reports/inspect/2026-05-13
```

Uses new checkpoints from Phase C (seed42 baselines). Generates Plotly overlays on 2023–2025 test data. Confirm HTML files render and contain overlay traces.

### E2 — Update `docs/models-status.md`

Complete rewrite of the "Rigor Sprint Results" section. Replace all `2026-05-11` report paths with `2026-05-13`. Replace all stale numbers.

New section structure (replaces current rigor sprint section):

```
## Rigor Sprint Results (2026-05-13 Rerun)

Sprint basis: 2016–2025 data, raw FVG labels (N+1), Python 3.12, CPU-only LSTM.

### Naive Baseline (new — honest comparison)
### Hyperparameter Tuning (G1)
### Seed Variance (G2) — 5 seeds each
### Loss Function Ablation — Focal (G3) — full 5 seeds
### Decision Threshold Tuning (G4)
### SHAP Feature Importance (G5)
### Bull vs Bear Asymmetry (G6) — multi-seed std
### Window Size Sensitivity (G7) — 5 window sizes
### Data Scaling Delta (G8)
### Regularisation Ablation (G9) — 5 seeds
### Bootstrap CI (G10) — CI of mean across seeds
### Summary Table
```

Delete the V2 section (that was ValidFVG — wrong labeller for current sprint). The V2 experiment is archived in `.nb-suite/archive/`.

Update the "Data splits" table at the top to reflect 2016–2025 boundaries.

Update the LSTM and XGBoost trained model sections with fresh checkpoint paths and new F1 numbers from Phase C.

### E3 — Patch CLAUDE.md

Remove stale MPS LSTM gotchas (done in Phase B already, but verify). Add:
```
**CPU-only for LSTM** — see .nb-suite/research/12-May-26/mps-gpu-fix.md
```

Remove the "MPS gotchas" entry "nn.LSTMCell slower than CPU → use sequence-batched nn.LSTM" (irrelevant now; we're on CPU).

Keep Transformer/CNN-LSTM relevant MPS notes for future phases.

### E4 — Spawn @nb-feedpulse for Status Update 1

After E2 and E3 are complete, spawn `@nb-feedpulse` with the following context:
- Sprint completed: full rigor rerun on 2016–2025 data
- MPS LSTM bug resolved via CPU training + Python 3.12 migration
- All 10 gaps closed with proper multi-seed coverage
- Fresh baselines: LSTM macro F1 = [from Phase C seed42 result], XGB = [from Phase C]
- Status Update 1 deadline: May 17

### E5 — Final review gate

Spawn `@nb-review` on:
- `docs/models-status.md` — check no stale 2026-05-11 paths, numbers are internally consistent
- `CLAUDE.md` — check CPU-only rule is clearly stated

**Decision gate for Phase E (= sprint complete):**
- Inspector HTML renders
- `docs/models-status.md` passes review (no stale paths, numbers match reports)
- `CLAUDE.md` reflects CPU-only rule
- feedpulse written

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MPS LSTM hang recurs | Low (CPU torch wheel eliminates MPS availability) | Use CPU-only torch install (`--index-url .../cpu`) |
| Python 3.12 dependency conflict | Low | All ML wheels battle-tested on 3.12; SHAP 0.51 + numpy 2.4 combination untested — run `import shap` smoke immediately after install |
| LSTM HP search takes >12 hrs on CPU (50 trials × 100 epochs) | Medium | Constrain num_layers=1 (confirmed winner); set MedianPruner to kill bad trials at epoch 10; timeout=300 per trial |
| XGB early-stops at round 3 on 2016–2025 data (seen in V2 run) | High if mlogloss used | For raw FVG labels (~14% bull/10% bear), mlogloss is fine; the V2 issue was extreme ValidFVG sparsity. Confirm class balance before running |
| Reproducibility fails on CPU under PyTorch (different from MPS) | Medium | Use `torch.use_deterministic_algorithms(True)` for seed42 baseline runs; document 8× slowdown for future runs; fast sweep uses non-deterministic |
| G9 reg ablation 20 runs takes too long | High | Run overnight; reduce patience to 10 for ablation runs (control uses full patience=15) |
| Data scaling Δ CLI arg doesn't exist | Low | Implement `--splits {default,legacy}` in both trainers before running C5; 30-min task |
| bootstrap_ci.py needs multi-seed mode | Certain (current impl is single-npz) | Implement `--mode multi-seed --preds-dir <dir>` before Phase D closes |
| feedpulse before all results are in | Low | E4 is last step; all numbers available before spawning |

---

## File paths changed in this sprint

| File | Change |
|------|--------|
| `scripts/training/train_lstm.py` | CPU forced, device warning added (Phase B2) |
| `src/rigor/optuna_utils.py` | timeout=300 added to run_study (Phase B3) |
| `tests/rigor/` (3 test files) | skip decorators removed (Phase B4) |
| `CLAUDE.md` | MPS section updated, CPU-only rule added (Phase B6) |
| `src/data/split.py` | `--splits legacy` support, if needed (Phase C5) |
| `scripts/training/train_lstm.py` | `--splits` CLI arg (Phase C5) |
| `scripts/training/train_xgboost.py` | `--splits` CLI arg (Phase C5) |
| `scripts/bootstrap_ci.py` | `--mode multi-seed` added (Phase D/G10) |
| `scripts/reg_ablation.py` | new script or multiseed_run extension (Phase D/G9) |
| `scripts/window_sweep.py` | `--patience` and `--device` CLI args added; effective_n reported (Phase D/G7) |
| `docs/models-status.md` | full rewrite of rigor section (Phase E2) |

---

## Overnight run schedule (3 nights)

**Night 1:** G1 LSTM HP search (50 trials, CPU). Start before bed, check in the morning.
```bash
nohup python scripts/training/tune_lstm.py --n-trials 50 \
  --storage checkpoints/lstm/optuna_2026-05-13.db \
  --output-dir reports/rigor/2026-05-13/g1_hp_lstm > /tmp/lstm_hp.log 2>&1 &
```

**Night 2:** G2 LSTM WeightedCE 5 seeds + G3 focal ablation first 2 gammas (γ=1, γ=2). Kick off after G1 config confirmed.
```bash
nohup python scripts/multiseed_run.py --model lstm ... > /tmp/g2_lstm.log 2>&1 &
```

**Night 3:** G3 focal γ=3 + G9 reg ablation 5 seeds × 4 configs. These can run in sequence via a shell script.

**Daytime tasks (fast, no GPU):** G4 threshold, G5 SHAP, G6 asymmetry, G7 window sweep (5 points, CPU, fast), G10 bootstrap CI, Phase E docs.

---

## Definition of done

The sprint is done when all of the following are true:

1. `checkpoints/lstm/` contains only 2026-05-13 checkpoints (5 WeightedCE + 15 focal + 4 reg ablation + 1 seed42 baseline)
2. `checkpoints/xgboost/` contains only 2026-05-13 checkpoints (5 WeightedCE + 1 seed42 baseline)
3. `reports/rigor/2026-05-13/` has all 10 gap subdirs with required output files
4. `docs/models-status.md` has no references to 2026-05-11 paths or stale numbers
5. pytest passes with 0 skipped, 0 failures
6. `CLAUDE.md` explicitly states CPU-only LSTM with link to research doc
7. @nb-feedpulse doc written and ready for May 17 Status Update 1
