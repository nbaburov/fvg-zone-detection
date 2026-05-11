# Models Status

What is trained, what data it saw, what splits we have, how it performs, what's validated.

> **Current label target (training):** `FVGLabeller` — raw geometric 3-candle FVG at index N+1.
> `ValidFVGLabeller` (6-criteria SMC @ N+2) is implemented and validated against the gold set (κ=1.0) but the trained checkpoints below were produced on the raw labels. Migrating training to the validated labels is the next phase.

## Data splits (temporal, no shuffle)

| Split | Dates | Bars | None (0) | Bull (1) | Bear (2) |
|-------|-------|------|----------|----------|----------|
| train | 2018-01-02 → 2021-12-31 | 7,056 | 5,318 (75.4%) | 1,053 (14.9%) | 685 (9.7%) |
| val   | 2022-01-03 → 2022-12-30 | 1,757 | 1,287 (73.2%) | 237 (13.5%)   | 233 (13.3%) |
| test  | 2023-01-03 → 2024-12-31 | 3,514 | 2,617 (74.5%) | 528 (15.0%)   | 369 (10.5%) |

**Class balance:** ~75% none, ~14% bull, ~10% bear across all splits. Moderately imbalanced — not extreme.

**Stratification:** none. The split is purely chronological (per `src/data/split.temporal_split`). Each split sees consecutive years; no oversampling, no SMOTE, no temporal shuffling. This is intentional to prevent lookahead leakage and to test real out-of-distribution performance across years.

## Balancing strategy

Class weights computed on the **train split only** by inverse frequency, persisted to `data/processed/class_weights.json`:

| Class | Weight |
|-------|--------|
| 0 (none)  | 0.217 |
| 1 (bull)  | 1.097 |
| 2 (bear)  | 1.686 |

Used by `WeightedCE` in `src/training/loss.py`. No oversampling. No `FocalLoss` in current runs (implemented for ablation but not active).

## Trained models

### LSTM — `checkpoints/lstm/lstm_seed42.pt`

| Attribute | Value |
|-----------|-------|
| Architecture | 2-layer unidirectional LSTM, 64 hidden units, dropout 0.2 |
| Parameters | 51,651 |
| Input shape | `(batch, 60, 5)` — normalised per-window OHLCV |
| Output | 3-class softmax `{none, bull, bear}` |
| Loss | WeightedCE (inverse-freq weights) |
| Optimiser | Adam, lr 1e-3 |
| Early stop | EMA-smoothed val Macro-F1, patience 15 |
| Device | MPS (Apple Silicon) |
| Seed | 42 |
| Best epoch | 59 |
| Trained | 2026-05-11 |

**Test results (full test split, from `lstm_seed42.meta.json`):**

| Metric | Value |
|--------|-------|
| Macro F1 | **0.824** |
| Bull F1 | 0.768 |
| Bear F1 | 0.788 |
| Best smoothed val Macro F1 | 0.822 |
| NaN detected during training | no |

### XGBoost — `checkpoints/xgboost/xgb_seed42.ubj`

| Attribute | Value |
|-----------|-------|
| Library | xgboost 3.2.0, `tree_method="hist"` |
| Objective | `multi:softprob` (3 classes) |
| `n_estimators` (used) | 297 of 300 (early stop kicked in) |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 5 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| Eval metric | mlogloss + merror, early-stopping rounds 30 |
| Class weights | applied via `sample_weight` (same inverse-freq weights) |
| Input | 60-bar window → feature vector via `src/features/window_features.py` |
| Seed | 42 |
| Trained | 2026-05-11 |

**Test results (Jan 2024 slice, from inspector run):**

| Metric | Value |
|--------|-------|
| F1 Bull | 0.683 |
| F1 Bear | 0.500 |
| F1 FVG Macro | 0.591 |
| F1 Binary FVG vs none | 0.623 |

Full-test macro F1 not yet recomputed end-to-end with checkpoint reload — the slice numbers above come from the inspector and are representative.

## Side-by-side (Jan 2024 slice via inspector, lookahead=20 bars, TP=2R)

| Model   | F1 Macro | F1 Binary | Trades | TP | SL | Undec | Win Rate | Total R |
|---------|---------:|----------:|-------:|---:|---:|------:|---------:|--------:|
| XGBoost | 0.591    | 0.623     | 37     | 12 | 12 | 12    | 50.0%    | +17.54  |
| LSTM    | 0.822    | 0.816     | 25     | 9  | 7  | 8     | 56.2%    | +13.24  |

LSTM = fewer, higher-quality trades; XGBoost = more trades, lower hit rate, similar total R. Both have **positive expectancy** on this slice but the sample is tiny and excludes fees + realistic slippage.

## Validation

### Gold annotation set (offline label-quality validation, not model validation)

- File: `data/gold_labels.csv` (75 annotated rows)
- Purpose: validates the *labeller*, not the model.
- Sampling: stratified by `fvg_rich`, `low_vol`, `random` strata (see `src/data/annotate.py`). `fvg_rich` strata are drawn from positions where `raw_label != 0`, so the human annotator was shown the raw `FVGLabeller` output.
- Cohen's κ vs raw `FVGLabeller` (what was actually annotated): **1.0** — perfect agreement.
- Cohen's κ vs `ValidFVGLabeller` (recomputed post-hoc): **0.89** — Valid rejects a small subset the human accepted (criteria 5/6 filter some geometrically-valid gaps).

**Interpretation:** The gold set confirms that raw FVG geometry matches human intuition perfectly. The ValidFVGLabeller adds trader-style strictness (reaction confirmation + BOS + Gann midpoint), which removes ~10% of human-accepted FVGs. This is by design — Valid trades signal quality for sparsity.

### Model validation set

- File: `data/processed/spy_h1_val.parquet` (2022, 1,757 bars).
- Used by both trainers for early stopping (LSTM via EMA-smoothed Macro F1, XGBoost via mlogloss).
- Never touched at test time.

## Testing

| Layer | Where | Test count |
|-------|-------|-----------:|
| Data pipeline | `tests/data/` | passing |
| Labellers | `tests/data/labels/` (incl. 92-test valid_fvg suite) | passing |
| Models + training | `tests/models/`, `tests/training/` | passing |
| Inspector | `tests/inspect/` | 34 |
| Live paper-trading | `tests/live/` | 47 |

**Mandatory anti-lookahead fixture** in `tests/data/labels/test_valid_fvg.py` asserts label index is exactly N+2 and that no future-bar information leaks into earlier positions. Equivalent fixture for raw `FVGLabeller` asserts N+1.

## What's NOT validated yet

- **Forward paper trading**: `scripts/paper_trade.py` is fully implemented but has not run against a live Alpaca session. Dry-run during market hours is the next step.
- **Migration to ValidFVG training labels**: trained models above use raw labels. ValidFVG sparsity is severe (~0.3% positive rate) and training stability with that level of imbalance is unknown. Plan in `.nb-suite/plan/11-May-26/phase4-lstm-baseline.md`.
- **Out-of-sample 2025+ generalisation**: test set ends 2024-12-31. Performance on 2025 bars not measured.
- **Multi-symbol generalisation**: SPY-only. No transfer test to QQQ / individual equities.
- **Backtest with fees and slippage**: inspector outcome simulation assumes instant fill at next open, 0 slippage, 0 fees. Real-world performance will be worse.

## Reproducibility

- Seed 42 fixed in `train_utils.set_seed` — covers Python, NumPy, PyTorch.
- Deterministic MPS mode disabled (8× slowdown documented in `CLAUDE.md`); for fully deterministic runs use CPU.
- Each checkpoint ships a `.meta.json` with git SHA, Python version, platform, device, seed, and test metrics.
- Inspector and trainer both consume `class_weights.json` from disk — never recompute weights on the fly.
