# Build Log — Phase 4 LSTM Baseline

**Date:** 2026-05-11
**Branch:** master
**Plan:** `.nb-suite/plan/11-May-26/phase4-lstm-baseline.md`

## Done

- `src/models/lstm.py` — FVGLSTMClassifier (51,651 params, nn.LSTM sequence-batched, unidirectional)
- `src/training/__init__.py` — package marker
- `src/training/loss.py` — WeightedCE + FocalLoss
- `src/training/early_stop.py` — EarlyStop with EMA smoothing (patience, min_delta, mode)
- `src/training/train_utils.py` — set_seed, log_run_metadata, eval_epoch
- `tests/models/test_lstm.py` — 21 unit tests, all passing
- `scripts/train_lstm.py` — full end-to-end training script
- `checkpoints/lstm/lstm_seed42.pt` — best checkpoint (epoch 59)
- `checkpoints/lstm/lstm_seed42.meta.json` — run metadata sidecar
- `logs/lstm_seed42.csv` — training curve CSV (74 epochs)
- `.nb-suite/test-logs/11-May-26/lstm-baseline.md` — evaluation log

## Test results

- 21/21 unit tests pass (includes MPS device test)
- Full training run: 74 epochs, early stop at patience=15

## Key results

| Metric | Naive | XGBoost | LSTM | Delta vs XGB |
|--------|-------|---------|------|-------------|
| Test macro-F1 | 0.2846 | 0.6084 | 0.8242 | +0.2158 |
| Bull F1 | 0.0000 | 0.5519 | 0.7676 | +0.2157 |
| Bear F1 | 0.0000 | 0.5183 | 0.7881 | +0.2698 |

## Good

- LSTM substantially beat XGBoost (+0.2158 macro-F1) — contra R6/R7 expectations for small data
- No NaN on MPS — gradient clipping (max_norm=1.0) + OneCycleLR kept training stable
- Training converged cleanly: train_loss from 1.09 → 0.10, early stop at epoch 74
- EMA smoothing worked well (patience=15, alpha=0.3) — noise-robust stopping
- All 21 tests pass including MPS device test
- Infrastructure (loss.py, early_stop.py, train_utils.py) designed for direct reuse in CNN-LSTM

## Bad / Fixed

- Labeller key: plan specified `"valid_fvg"` but actual registry key is `"fvg_valid"`. Fixed in script before writing — checked with `list(LABELLERS.keys())` per plan pre-mortem item 5.
- Plan's `build_pipeline()` approach skipped per plan decision (Option A) — loaded parquets directly with stride=1, no pipeline modification needed.

## Observations

- LSTM on raw OHLCV (5 features) substantially outperforms XGBoost on 35 engineered features. The 60-candle sequence gives LSTM enough temporal context that it rediscovers the pattern structure without hand-crafted features.
- Val macro-F1 peaked around epoch 24-36 (0.83 range), test macro-F1 matched val closely (0.8242) — no extreme overfitting.
- The "LSTM won't beat XGBoost on ~117 effective samples" pre-mortem item did not materialise. Effective N with stride=1 (highly overlapping windows, 6,997 train) was sufficient for LSTM to generalise.
- Bear class F1 (0.7881) > Bull F1 (0.7676) on test — surprising given fewer bear samples (363 vs 519). Class weights may over-correct.

## Open flags

- stride=1 produces 59/60 bar overlap between adjacent windows. This is acknowledged in the eval log. F1 is still computed on temporal holdout (2023-2024) so leakage is not a concern for the reported numbers.
- Multi-seed evaluation (5 seeds) is out of scope for this baseline — use point estimate F1.
- FocalLoss implemented but not used — available for CNN-LSTM ablation if minority-F1 gets stuck.

## Next steps

CNN-LSTM baseline plan. DL floor is now 0.8242 macro-F1 (not XGBoost's 0.6084). The CNN-LSTM architecture must beat this to justify kernel-based locality encoding.

The training infrastructure (loss.py, early_stop.py, train_utils.py) is template-ready — copy scripts/train_lstm.py and swap model class + hyperparams only.
