# XGBoost Baseline Evaluation — Phase 4 Step 0

**Date:** 2026-05-11
**Git SHA:** 10f6d1d
**Python:** 3.14.4
**XGBoost:** 3.2.0
**Seed:** 42

## Dataset summary

| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|--------------|
| Train | 6997 | 1718 | 24.6% |
| Val   | 1698 | 458 | 27.0% |
| Test  | 3455 | 882 | 25.5% |

Note: stride=1, window=60 for all splits. Overlapping windows are expected.
Test positives sanity: 882 (threshold: >= 200).

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 (max) |
| max_depth | 4 |
| learning_rate | 0.05 |
| min_child_weight | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | multi:softprob |
| num_class | 3 |
| early_stopping_rounds | 30 |
| tree_method | hist |
| seed | 42 |

## Training

- Final n_estimators used: 297 (early stopping on val mlogloss)
- Best val mlogloss: 0.680628
- Val merror at best: 0.3357
- Val mlogloss curve (first 5): ['1.0826', '1.0716', '1.0608', '1.0488', '1.0394']
- Val mlogloss curve (last 5):  ['0.6817', '0.6814', '0.6812', '0.6809', '0.6806']

## Naive baseline (always-majority, test set)

F1 macro: 0.2846  ← majority-class predictor for reference

## Validation results

```
              precision    recall  f1-score   support

        none     0.8743    0.6339    0.7349      1240
        bull     0.4571    0.7652    0.5724       230
        bear     0.4010    0.7281    0.5171       228

    accuracy                         0.6643      1698
   macro avg     0.5775    0.7091    0.6081      1698
weighted avg     0.7542    0.6643    0.6837      1698

```

Confusion matrix (val) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       786       208       246
        bull        52       176         2
        bear        61         1       166
```

## Test results (primary)

```
              precision    recall  f1-score   support

        none     0.8863    0.6576    0.7550      2573
        bull     0.4354    0.7534    0.5519       519
        bear     0.4043    0.7218    0.5183       363

    accuracy                         0.6787      3455
   macro avg     0.5754    0.7109    0.6084      3455
weighted avg     0.7680    0.6787    0.6996      3455

```

Confusion matrix (test) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none      1692       504       377
        bull       119       391         9
        bear        98         3       262
```

- **Macro-F1: 0.6084**
- Minority class F1 (bull): 0.5519
- Minority class F1 (bear): 0.5183
- Positive count sanity: 882 windows (>= 200 OK)

## Comparison to naive baseline

| Metric | Naive (majority) | XGBoost | Delta |
|--------|-----------------|---------|-------|
| Macro-F1 | 0.2846 | 0.6084 | +0.3238 |
| Bull F1  | 0.0000 | 0.5519 | +0.5519 |
| Bear F1  | 0.0000 | 0.5183 | +0.5183 |

## Top-10 feature importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | above_ma20 | 0.271602 |
| 2 | ret_5 | 0.119506 |
| 3 | gap_bull | 0.034513 |
| 4 | vol_spike | 0.032081 |
| 5 | gap_norm_bull | 0.030029 |
| 6 | gap_bear | 0.028691 |
| 7 | mid_body_bull | 0.027617 |
| 8 | mid_body_bear | 0.025091 |
| 9 | pos_in_range | 0.024688 |
| 10 | vol_zscore_5 | 0.023550 |


## Phase 4 handoff

- **XGBoost test macro-F1: 0.6084** — this is the floor DL models must beat.
- Bull F1 floor: 0.5519
- Bear F1 floor: 0.5183
- Model saved at: checkpoints/xgboost/xgb_seed42.ubj
- Naive baseline macro-F1: 0.2846 (delta to XGBoost: +0.3238)

Note on overlapping windows: stride=1 produces highly overlapping windows (59/60 shared bars between adjacent windows). This is standard for XGBoost tabular evaluation and matches the distribution DL models will train on. F1 is computed on a held-out temporal test split (2023-2024), so temporal leakage is not a concern despite overlap.
