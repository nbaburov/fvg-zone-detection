# Changelog

All notable changes to the SMC Data Challenge. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

No SemVer releases tagged yet — sections are dated working-tree milestones (newest first). Consolidated from per-session logs formerly under `.nb/changelogs/`.

## [2026-06-09] — Documentation overhaul: plain-language showcase set

Restructured `docs/` into a clean, plain-language, showcase-grade documentation set. Standard applied: explain concepts, don't name-drop; define jargon on first use; no hallucinations or speculative claims; readable by non-technical and non-financial readers.

### New docs
- **`docs/overview.md`** — plain-language project overview: what the system does, why, and how it is evaluated.
- **`docs/data.md`** — merges the former `data-model.md` and `fvg-label-guide.md` into one coherent document. Covers data acquisition, resampling, the FVG label definition (all 6 validity criteria explained in plain English), class imbalance, and the temporal split.
- **`docs/evaluation.md`** — new document. Explains why F1 is the primary metric, the G1–G10 validation sprint (10 controlled experiments across 5 seeds each), and how results are interpreted honestly.
- **`docs/models.md`** — replaces `models-status.md`. Includes per-model plain-language architecture descriptions, ASCII/Mermaid diagrams, numerical results table (5-seed F1 ± std), and an honest data-bound diagnosis.

### Renamed
- `docs/fvg-trading-simulation.md` → `docs/trading-simulation.md` — name no longer ties the document to a specific model; matches the broadened multi-arch scope.

### Removed (content merged into new set + `reports/rigor/`)
- `docs/idea.md` — superseded by `docs/overview.md`
- `docs/data-model.md` — merged into `docs/data.md`
- `docs/fvg-label-guide.md` — merged into `docs/data.md`
- `docs/models-status.md` — merged into `docs/models.md`; raw results preserved in `reports/rigor/`

### Changed
- `README.md` — rewritten with a clean docs index pointing to the new set; corrected `build_pipeline` and training commands.
- `CLAUDE.md` — doc-references updated to the new filenames.
- `docs/architecture.md` — language and cross-references aligned to the new set.

---

## [2026-06-09] — FVG exit-strategy trade-simulation + realism guards

Tests whether the FVG detector's signals are actually tradeable on unseen 2023–2025 data, using four cited SMC/ICT exit strategies with realistic execution assumptions.

### Strategy layer

- **`src/strategy/exits.py`** — single source of truth for trade-outcome logic. Defines `ExitConfig`, `TradeOutcome`, and `compute_exit`. Four FVG exit strategies implemented and cited: `fixed_2r` (market-order bracket, 2× risk TP), `ict_iofed` (ICT inversion-of-FVG entry, gap-midpoint TP), `ce_50pct` (50% Gann midpoint CE pullback), `tradinglab` (TradingLab gap-cover TP). Realism guards: ATR min-stop floor (default 0.3×ATR, sweepable), transaction costs (half-spread + slippage), confidence filter (skip signals below threshold), optimistic vs conservative fill mode (optimistic = assume limit fills; conservative = degrade limit-entry strategies to market-order P&L).
- **`src/strategy/__init__.py`** — package init exposing `compute_exit`, `ExitConfig`, `TradeOutcome`.

### Inspect integration

- **`src/inspect/outcomes.py`** — delegates to `compute_exit`; adds after-cost, median, winsorized, and outlier metrics alongside the existing raw-R metrics.
- **`src/inspect/report.py`** — multi-strategy × all-seeds (5-seed mean±std) tables. `--all-seeds` aggregation path.
- **`scripts/inspect_models.py`** — new flags: `--exit-strategy`, `--all-exit-strategies`, `--realistic`, `--min-stop-atr-k`, `--min-stop-atr-k-sweep`, `--confidence-threshold`, `--confidence-sweep`, `--all-seeds`, `--fill-mode`, `--sensitivity-sweep`. Backward compatible: defaults reproduce prior behaviour byte-identical.

### Results (`docs/fvg-trading-simulation.md`)

Full trade-sim on unseen 2023–2025 data: 4 tickers (SPY, QQQ, IWM, DIA) × 4 model archs × 4 exit strategies × 5 seeds + sweeps. **Honest finding:** edges are thin and uncertain. Only `fixed_2r` (market-order bracket) survives a realistic fill assumption — limit-entry strategies (`ict_iofed`, `ce_50pct`, `tradinglab`) show positive R only under optimistic fill; degrading to conservative (market-order) P&L eliminates the edge. Best defensible single cell: XGBoost + `fixed_2r`, QQQ, **+18.7±2.0R (sim)**. This is exploratory analysis — not a proven profitable strategy. 573 tests pass.

### Added
- `src/strategy/__init__.py`, `src/strategy/exits.py`
- `tests/strategy/test_fvg_exits.py`, `tests/strategy/test_realism_guards.py`, `tests/strategy/test_sim_enhancements.py`
- `tests/inspect/test_outcomes_regression.py`
- `docs/fvg-trading-simulation.md`
- `data/processed/qqq_h1_test.parquet`, `data/processed/iwm_h1_test.parquet`, `data/processed/dia_h1_test.parquet` — per-symbol unseen-test slices (~200 KB each; regenerable from multisym pooled test)
- `.nb/plan/09-Jun-26/exit-sim-realism.md`, `.nb/plan/09-Jun-26/sim-enhancements.md`
- `.nb/research/09-Jun-26/fvg-exit-strategies.md`

### Changed
- `src/inspect/outcomes.py` — delegates to `compute_exit`; after-cost/median/winsorized/outlier metrics
- `src/inspect/report.py` — multi-strategy + all-seeds tables
- `scripts/inspect_models.py` — exit-strategy flags + sensitivity sweeps
- Docs synced: `CLAUDE.md`, `README.md`, `docs/architecture.md`

### Notes
- `.nb/plan/09-Jun-26/fvg-exit-strategies.md` and `.nb/plan/09-Jun-26/multi-symbol-expansion.md` are NOT staged here — they are plan files for future work, not part of this feature.
- `checkpoints/xlstm_multisym/` NOT staged — training in progress.

---

## [2026-06-09] — inspect/paper-trade all-5-arch tooling

All 5 model architectures are now inspectable and paper-tradeable. Previously only lstm, cnn_lstm, and xgboost had adapters; transformer and xlstm could be trained but not post-hoc analysed or wired into the paper-trading loop.

### Inspect / paper-trade adapter layer

- **`TransformerAdapter`** (`src/inspect/adapters/transformer_adapter.py`) and **`XLSTMAdapter`** (`src/inspect/adapters/xlstm_adapter.py`) — new auto-discovered adapters. Both implement the `ModelAdapter` ABC; the registry discovers them at import time, so `scripts/inspect_models.py` and `scripts/paper_trade.py` support all 5 archs with no whitelist changes.
- **Per-model checkpoint override** — `scripts/inspect_models.py --models name[:checkpoint_path]` mirrors the existing `paper_trade.py` `name:path` syntax. Each arch's specific checkpoint (e.g. best multisym seed) loads from its own directory in a single command. Backward compatible: bare `name` falls back to `--checkpoint-dir` + default seed.
- All 5 adapters (`lstm`, `cnn_lstm`, `xgboost`, `transformer`, `xlstm`) gained an optional `checkpoint_path` kwarg. `registry.load_adapters` gained a `checkpoint_paths` mapping so callers can supply per-adapter paths without touching the registry internals.
- **Registry resilience** — `_discover` now tolerates a single adapter import failure (logs a warning, continues loading the remaining adapters) instead of aborting all inspect/paper-trade runs on a missing optional dep.
- **xLSTM context_length fix** — `XLSTMAdapter` resolves `context_length` from checkpoint meta `hyperparams.window_size` so it matches the window the model was actually trained on, rather than relying on a hardcoded default.

### Tests
5 new test files in `tests/inspect/`: `test_transformer_adapter.py`, `test_xlstm_adapter.py`, `test_registry.py` (updated + new coverage), `test_checkpoint_override.py`, `test_xlstm_context_length_fallback.py`. Full suite green.

### Config
- `experiments/xlstm_multisym.yaml` — xLSTM multisym retrain config (mirrors the pattern of the other `*_multisym.yaml` configs already committed). Training in progress; checkpoints land in a later commit.

### Added
- `src/inspect/adapters/transformer_adapter.py`
- `src/inspect/adapters/xlstm_adapter.py`
- `tests/inspect/test_transformer_adapter.py`, `test_xlstm_adapter.py`, `test_registry.py`, `test_checkpoint_override.py`, `test_xlstm_context_length_fallback.py`
- `experiments/xlstm_multisym.yaml`

### Changed
- `src/inspect/adapters/{lstm,cnn_lstm,xgboost}_adapter.py` — added optional `checkpoint_path` kwarg
- `src/inspect/registry.py` — `_discover` resilience + `checkpoint_paths` support in `load_adapters`
- `scripts/inspect_models.py` — `name:path` parsing, duplicate-name guard, docstring updates
- Docs synced: `CLAUDE.md`, `README.md`, `docs/architecture.md`

### Notes
- Trade-outcome simulation (`--lookahead-bars`) uses a **generic 2R bracket** (gap-edge stop-loss, fixed 2× risk take-profit) for all architectures. This is a placeholder; a researched, SMC-aligned FVG exit strategy is future work and is not claimed as canonical.
- `checkpoints/xlstm_multisym/` is NOT committed here — training in progress; will land in a subsequent commit alongside the results.

---

## [2026-06-09]

Phase 4 multi-symbol expansion COMPLETE. Data-bound hypothesis SUPPORTED (directionally strong; not proven at strict 95% due to effective_n=86).

### Multi-symbol training pipeline
- `build_multi_symbol_pipeline` pools SPY + QQQ + IWM + DIA (2016–2021 train, 2022 val, 2023–2025 test), each symbol processed independently then concatenated. Minority bull/bear-FVG positives multiplied **4.06×/3.74×** vs SPY-only baseline.
- `download_h1(symbol)` generalised — raw minute parquets `data/raw/{qqq,iwm,dia}_minute.parquet` tracked per repo convention (~131 MB total, all <100 MB each).
- Cross-symbol window guards: `src/data/window.py` + `src/features/window_features.py` XGB path now enforce no window spans a symbol boundary. `_sort_if_multisym` reload guard prevents silent sort-loss on parquet round-trip.
- `adjustment="raw"` fix applied consistently across all symbol pulls.
- `dataset_meta.json` sidecar extended with `symbols` field; pooled processed parquets under `data/processed/multisym/`.

### New tooling
- `scripts/data/depth_probe.py` — D9 gate confirming each symbol has sufficient history before pooling (already committed 2026-06-09 in prior commit).
- `scripts/rigor/eval_spy_test.py` + `scripts/rigor/_workers/_xgb_eval_spy_worker.py` — fixed SPY-only test-set evaluator so multi-symbol-trained models are benchmarked on the same held-out SPY slice as the single-symbol baselines (apples-to-apples).
- `experiments/transformer_multisym.yaml`, `experiments/xgb_multisym.yaml` (cnn_lstm + lstm multisym configs committed in pipeline commit).
- `scripts/rigor/multiseed_run.py` — XGB cross-symbol reload sort fix.

### Results (fixed SPY-only test, bootstrap CI 1000-iter block=60, effective_n≈86)
| Model | SPY-only F1 | Multi-sym F1 | Δ | bear_f1 |
|---|---|---|---|---|
| CNN-LSTM | 0.639 [0.603, 0.674] | **0.675 [0.637, 0.710]** | +0.036 | 0.439 → 0.505 |
| LSTM | 0.595 | **0.640** | +0.045 | — |
| XGB | 0.721 | **0.738** | +0.017 | — |
| Transformer | 0.601 | 0.577 | −0.024 | seed42 collapse persists |

**Verdict:** data-bound hypothesis SUPPORTED. CNN-LSTM clears the pre-registered 0.674 threshold and all 5 seeds exceed 0.666. However bootstrap CIs overlap at strict 95% (effective_n=86 on the SPY test slice) — result is "supported, directionally strong, not proven at 95%". Transformer failed to improve, confirming it is instability-bound rather than data-bound. Full rationale in `reports/rigor/09-Jun-26/phase4_verdict.md`.

### Added
- `scripts/rigor/eval_spy_test.py`, `scripts/rigor/_workers/_xgb_eval_spy_worker.py`
- `experiments/transformer_multisym.yaml`, `experiments/xgb_multisym.yaml`
- `checkpoints/{cnn_lstm,lstm,transformer,xgb}_multisym/` — all per-seed `.pt`/`.ubj` + meta (tracked per repo convention)
- `data/processed/multisym/` — pooled train/val/test parquets + class_weights + dataset_meta
- `data/raw/{qqq,iwm,dia}_minute.parquet` — raw minute bars (tracked per repo convention)
- `reports/rigor/09-Jun-26/phase4_verdict.md`, bootstrap CI JSONs (×4), `spy_test_eval.json`

### Changed
- `scripts/rigor/multiseed_run.py` — XGB cross-symbol reload sort guard
- Docs synced: `CLAUDE.md`, `README.md`, `docs/architecture.md`, `docs/data-model.md`, `docs/models-status.md`

### Notes
- Ephemeral artifacts NOT committed: `spytest_preds/*.npz`, `*.log`, `depth_probe.json` (ephemeral probe output), run-output dirs under `reports/rigor/09-Jun-26/{cnn_lstm,lstm,transformer,xgb}_multisym/`.
- Next lever: additional symbols (e.g. GLD, TLT) or longer history could push past the 95% CI threshold with higher effective_n.

## [2026-06-07]

DL ladder completed (5 archs) + data-vs-capacity diagnosis. CNN-LSTM confirmed carrier; task is data-bound.

### Status Update 2 deliverables + Transformer fair-shot tuning
- **Status Update 2** (`notebooks/06-status-update-2.ipynb` + reveal deck `07-status-update-2-presentation.ipynb`, generated by `scripts/_build_su2.py`): self-contained, education-grade, organised around the question "why did classical ML beat deep learning?". 16 figures from real committed numbers, the four official learning outcomes mapped, references cited inline, tooling/product surface, honest limitations, humanized first-person prose. Exported to HTML + reveal slides.
  - Prose polish: every named technique (focal loss, gradient clipping, Cohen's kappa, class weights, XGBoost, LSTM, Transformer, xLSTM, Optuna, threshold, dropout/weight-decay, bootstrap, softmax) now carries a one-sentence plain-language mechanism, so the deck reads name-drop-free for a non-technical reader.
  - Transformer fair-shot result folded in: a tuned, stabilised Transformer (warm-up + gradient clipping + Optuna) reaches 0.601 ± 0.045 (5-seed) / 0.604 ± 0.050 (10-seed, no collapse), up from the untuned 0.548. It ties LSTM and stays below CNN-LSTM (0.639): even properly tuned, the larger architecture reaches parity not a win, confirming the data-bound (not capacity-bound) verdict. Part 8 + next-steps + limitations reframed from in-progress to done.
- **Transformer fair-shot tuning**: `scripts/rigor/tune_transformer.py` (Optuna), `scripts/rigor/transformer_fairshot_pipeline.sh` (sweep -> 5-seed -> collapse probe), `experiments/transformer_tuned.yaml`, `TransformerObjective` in `optuna_utils.py`, `warmup_steps`/`max_grad_norm` in schema. LR warm-up + gradient clipping are **gated transformer-only** in `seed_sweep._train_torch_generic` so the committed LSTM/CNN-LSTM/xLSTM baselines are byte-for-byte unchanged. `tests/rigor/test_backward_compat_gating.py` enforces the gating.
- **Fix**: `_train_torch_generic` now filters model kwargs to the constructor signature, so Optuna best-HP JSON metadata (`n_trials_completed`, `study_name`, ...) can no longer reach the model constructor. Suite: 365 tests pass.


### Added
- **Transformer** (`src/models/transformer.py`, `FVGTransformerClassifier` — encoder + mean/CLS pool) and **xLSTM** (`src/models/xlstm_model.py`, `FVGxLSTMClassifier` — sLSTM stack, `xlstm==2.0.5` v2 API, vanilla backend for Apple Silicon). Both wired into schema/registry/dispatcher; configs `experiments/{transformer,xlstm}_g1.yaml`. Untuned baselines.
- **Learning-curve harness** `scripts/rigor/learning_curve.py` (val F1 vs train fraction, any arch) with an A2b guard asserting `--train-fraction` is honoured (window count scales with fraction).
- **`--train-fraction`** flag + per-epoch **train macro-F1 logging** in `scripts/training/train.py` (shared loop → all 4 torch archs), enabling the data-bound/capacity-bound diagnosis.
- `scripts/rigor/run_dl_diagnosis_pipeline.sh` — orchestrates the full ladder + learning curves (CPU, background-safe).
- Diagnosis gate report `reports/rigor/07-Jun-26/dl_diagnosis_decision.md` (per-model cards, bootstrap CIs, decision matrix) + bootstrap CI artifacts.
- Tests: `tests/models/test_{transformer,xlstm_model}.py`, `tests/config/test_schema_roundtrip.py`, `tests/rigor/test_{seed_sweep_routing,train_fraction}.py`. Suite 284 → **353**.

### Fixed
- **Segfault (macOS arm64):** `src/rigor/seed_sweep.py` routed transformer/xlstm through the `else` branch into in-process XGBoost, which segfaults after a torch import. Added `_train_torch_generic` (registry-built, CPU-forced); unknown arch now raises instead of misrouting. Regression-tested.
- `seed_sweep.py` `_get_device()` could return MPS (broken gradient kernel) — forced CPU.
- `seed_sweep.py` cache loader silently zeroed minority-class F1 for `train.py`-format metas (`test_bull_f1`/`test_bear_f1` vs `test_per_class_f1` array) — now reads both.
- `bootstrap_ci_multiseed.py` / `multiseed_run.py` — added `transformer`/`xlstm` to `--model` choices.

### Results (ValidFVG, 5 seeds, raw (60,5), 95% bootstrap CI)
- CNN-LSTM **0.639 ± 0.013** [0.603, 0.674] — carrier · LSTM 0.595 ± 0.015 · Transformer 0.548 ± 0.095 (unstable, seed42 collapse) · xLSTM 0.369 ± 0.007 (underfits). XGB 0.721 reference (engineered feats, input mismatch).
- **Diagnosis:** task is **data-bound, not capacity-bound** — bigger archs did worse; recurrent learning curves show no plateau. Lever = multi-symbol data. Full rationale + caveats in the decision report.

### Notes
- All experiment YAMLs CPU-only (MPS bug). xLSTM learning curve abbreviated to full-data anchor (CPU cost; full-data score already its ceiling). CNN-LSTM/LSTM ladder checkpoints overwritten by learning-curve runs — canonical results live in `reports/`.

## [2026-05-13]

CNN-LSTM baseline + config-system refactor.

### Added
- CNN-LSTM classifier (`src/models/cnn_lstm.py`) + registry integration. Topology: 2× Conv1d(16, k=5) → 2-layer LSTM(32). Canonical 5-seed mean macro F1 = 0.614 ± 0.021 — beats LSTM 0.599 (+0.015), trails XGB 0.721 (−0.107). Full rigor pass: threshold, asymmetry, window sweep, reg ablation, bootstrap CI (`1b7b030`, `1619f8c`).
- Pydantic v2 experiment config system: `src/config/` (schema, loader, registry, model + loss registrations), `experiments/*.yaml` (`_base`, `_template`, `lstm_g1`, `xgb_g1`, raw-FVG variants), unified `scripts/training/train.py` entry, 26 config tests (`2947b84`).
- YAML-driven migration of 8 `scripts/rigor/` scripts + `paper_trade.py` via `--config foo.yaml --set key=value`; legacy CLI preserved (`c3819ce`).
- `notebooks/03-baselines.ipynb` — naive predict-none floor, XGBoost (macro F1 0.722), LSTM (macro F1 0.588) on test seed=42 G2, via cached predictions to dodge macOS arm64 libgomp segfault (`7945e7f`).

### Changed
- `scripts/training/{train_lstm,train_xgboost}.py` — `--config`/`--set` flags, delegate to `train.py`.
- Notebook 01 synced to CNN-LSTM (40 cell changes, 0 exec errors).

### Fixed
- LSTM optimiser mismatch — Adam canonical (was AdamW + OneCycleLR).
- `bootstrap_ci_multiseed.py` `block_size` shadow bug.

### Notes
- 284/284 tests pass. G9 flagged dropout=0.222 as possible over-regularization (no_dropout +0.009).

## [2026-05-12]

Rigor-hardening sprint + dataset extension 2018–2024 → 2016–2025.

### Added
- Dataset extended to 2016–2025 SPY H1 via Alpaca free tier (confirmed reaches 2016-01-01). Splits: train 2016–2021, val 2022, test 2023–2025. Regenerated `spy_h1*.parquet` + `class_weights.json` (`8f02b3d`).
- Rigor infrastructure `src/rigor/` (seed sweep, Optuna wrapper + XGB sampler, threshold sweep, bootstrap CI @ 1000 resamples, report utils) + 8 `scripts/rigor/` scripts (multiseed, tune, SHAP, asymmetry, threshold, bootstrap) + 4 rigor test modules (`fb9d18b`).
- Subprocess workers `scripts/rigor/_workers/` for XGB tune/sweep/SHAP + `_xgb_infer_worker.py` — isolate XGB from torch to avoid arm64 segfault.
- `Makefile` + `pytest.ini` — two-pass test run isolating XGB baseline tests.

### Changed
- 10-gap rigor rerun on canonical ValidFVG target (py3.12 + CPU stack): **XGB 0.721 ± 0.001 dominates LSTM 0.599 ± 0.025**. Focal loss and threshold tuning both worse than baseline; reg helps marginally; W=90 best window (`6070a92`).
- Git now tracks checkpoints (`checkpoints/{lstm,xgboost}/*`) + raw/processed parquet + Optuna DBs for reproducibility (`cfb56c8`).
- `scripts/` reorganised into `data/` + `training/` subfolders (`0f67a3c`).
- `src/data/{download,split,process,normalize}.py` — Alpaca pagination, cache-hit date guard, `SPLIT_BOUNDARIES` rewired (legacy `SPLIT_BOUNDARIES_2018_2024` retained).
- Docs synced: `models-status.md` rewritten with G1–G10 + dual-FVG table, `architecture.md`, `data-model.md`, `CLAUDE.md`, `README.md` (`6a37bc0`, `6521a5f`).

### Fixed
- Phase B stack: forced CPU device in `train_lstm.py` (MPS LSTM deadlock), `run_study` gains `timeout=300, n_jobs=1`, XGB tests un-skipped via subprocess helper. 245 tests pass, 0 skipped (`4f64f03`).
- `seed_sweep.py` reg-ablation checkpoint cache bug.
- `src/inspect/adapters/{lstm,xgboost}_adapter.py` — read HP from `meta.json`, use subprocess for XGB (`6521a5f`). 22 Plotly overlays generated.

## [2026-05-11]

ValidFVGLabeller + gold annotation + inspect/live toolkits.

### Added
- `ValidFVGLabeller` (`src/data/labels/valid_fvg.py`) — vectorised 6-criteria SMC rule (geometric gap, N+2 reaction, S/R confluence [optional], priority, Gann midpoint, BOS), label index **N+2**, full ablation support. Helpers `sr.py` (pivot S/R), `bos.py` (break-of-structure). Registered `'fvg_valid'` (`939d250`).
- Gold annotation campaign: `src/data/annotate.py` (stratified sampler, anchoring-bias-free Plotly UI, Cohen's κ gate), `notebooks/02-gold-annotation.ipynb`, `docs/fvg-label-guide.md`. 75 samples, **κ=1.0** vs labeller (`8383311`, `3392b24`, `8e013b2`, `bb46e3c`).
- `src/inspect/` — offline toolkit: TP/SL outcome simulation + per-model timelines (`03bac5a`).
- `src/live/` — Alpaca paper-trading harness (`5fab708`).
- Top-level `README.md` + `docs/{architecture,data-model,models-status}.md`; session fixtures for inspect/live tests (`9e88039`, `1f8c185`). 92 tests pass.

### Changed
- `FVGLabeller` aliased under both `'fvg'` and `'fvg_raw'` — old geometric rule coexists as baseline (`1bf28e5`). Label index moved N+1 → N+2 for ValidFVG.
- Notebook 01 updated to discuss validated FVG target.

### Fixed
- Unified Alpaca secret env var as `ALPACA_SECRET_KEY` (`3d7a021`).

### Removed
- Deprecated gold labels + stale CSV/HTML report artifacts (`be0fd68`).

## [2026-05-08]

Project foundation — Phase 2 research + Phase 3 data pipeline.

### Added
- Phase 3 SMC data pipeline: 13 source modules (Alpaca acquisition + custom vectorised FVG labeller), 64 tests, 70-cell CRISP-DM Status-Update-1 notebook with non-finance primer.
- Phase 2 (Data Understanding) research: 6 artifacts, master phasing plan, Phase 3 sub-plan, scope + timeframe explanation docs.
- Local `.gitignore` override to track `CLAUDE.md` and nb artifacts.

### Changed
- Plan + `CLAUDE.md` revised for Alpaca pivot (yfinance rejected) and R6 architecture order — added XGBoost baseline, reordered Phase 4.
