# XGBoost Baseline Evaluation — Phase 4 Step 0

**Date:** 2026-05-12
**Git SHA:** 4f64f03
**Python:** 3.12.13
**XGBoost:** 3.2.0
**Seed:** 42

## Dataset summary

| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|--------------|
| Train | 10518 | 2545 | 24.2% |
| Val   | 1698 | 458 | 27.0% |
| Test  | 5198 | 1312 | 25.2% |

Note: stride=1, window=60 for all splits. Overlapping windows are expected.
Test positives sanity: 1312 (threshold: >= 200).

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

- Final n_estimators used: 300 (early stopping on val mlogloss)
- Best val mlogloss: 0.706146
- Val merror at best: 0.3498
- Val mlogloss curve (first 5): ['1.0847', '1.0723', '1.0630', '1.0529', '1.0437']
- Val mlogloss curve (last 5):  ['0.7067', '0.7067', '0.7065', '0.7064', '0.7061']

## Naive baseline (always-majority, test set)

F1 macro: 0.2852  ← majority-class predictor for reference

## Validation results

```
              precision    recall  f1-score   support

        none     0.8868    0.6000    0.7157      1240
        bull     0.4378    0.7957    0.5648       230
        bear     0.4014    0.7763    0.5291       228

    accuracy                         0.6502      1698
   macro avg     0.5753    0.7240    0.6032      1698
weighted avg     0.7608    0.6502    0.6702      1698

```

Confusion matrix (val) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       744       234       262
        bull        45       183         2
        bear        50         1       177
```

## Test results (primary)

```
              precision    recall  f1-score   support

        none     0.9030    0.5991    0.7203      3886
        bull     0.4128    0.7955    0.5436       792
        bear     0.3711    0.7808    0.5031       520

    accuracy                         0.6472      5198
   macro avg     0.5623    0.7251    0.5890      5198
weighted avg     0.7751    0.6472    0.6716      5198

```

Confusion matrix (test) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none      2328       890       668
        bull       142       630        20
        bear       108         6       406
```

- **Macro-F1: 0.5890**
- Minority class F1 (bull): 0.5436
- Minority class F1 (bear): 0.5031
- Positive count sanity: 1312 windows (>= 200 OK)

## Comparison to naive baseline

| Metric | Naive (majority) | XGBoost | Delta |
|--------|-----------------|---------|-------|
| Macro-F1 | 0.2852 | 0.5890 | +0.3038 |
| Bull F1  | 0.0000 | 0.5436 | +0.5436 |
| Bear F1  | 0.0000 | 0.5031 | +0.5031 |

## Top-10 feature importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | above_ma20 | 0.323089 |
| 2 | ret_5 | 0.128077 |
| 3 | gap_norm_bull | 0.034607 |
| 4 | gap_bull | 0.028149 |
| 5 | gap_bear | 0.027397 |
| 6 | vol_spike | 0.025293 |
| 7 | pos_in_range | 0.022979 |
| 8 | vol_zscore_5 | 0.022470 |
| 9 | mid_body_bull | 0.021702 |
| 10 | vol_zscore_20 | 0.021375 |


## Phase 4 handoff

- **XGBoost test macro-F1: 0.5890** — this is the floor DL models must beat.
- Bull F1 floor: 0.5436
- Bear F1 floor: 0.5031
- Model saved at: checkpoints/xgboost/xgb_seed42.ubj
- Naive baseline macro-F1: 0.2852 (delta to XGBoost: +0.3038)

Note on overlapping windows: stride=1 produces highly overlapping windows (59/60 shared bars between adjacent windows). This is standard for XGBoost tabular evaluation and matches the distribution DL models will train on. F1 is computed on a held-out temporal test split (2023-2024), so temporal leakage is not a concern despite overlap.
