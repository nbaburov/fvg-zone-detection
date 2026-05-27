# Changelog

All notable changes to the SMC Data Challenge. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

No SemVer releases tagged yet — sections are dated working-tree milestones (newest first). Consolidated from per-session logs formerly under `.nb/changelogs/`.

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
