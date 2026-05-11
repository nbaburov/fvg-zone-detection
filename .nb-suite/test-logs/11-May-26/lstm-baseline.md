# LSTM Baseline Evaluation — Phase 4 Step 1

**Date:** 2026-05-11
**Git SHA:** 8c09794
**Python:** 3.14.4
**PyTorch:** 2.11.0
**Device:** mps
**Seed:** 42

## Model architecture

- hidden_size: 64, num_layers: 2, dropout: 0.3, head_dropout: 0.5
- bidirectional: False (unidirectional — safe at label position 59)
- Total parameters: 51,651

## Dataset summary

| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|--------------|
| Train | 6997 | 1718 | 24.6% |
| Val   | 1698 | 458 | 27.0% |
| Test  | 3455 | 882 | 25.5% |

Note: stride=1 for all splits — matches XGBoost baseline evaluation for fair comparison.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 64 |
| num_layers | 2 |
| dropout (LSTM inter-layer) | 0.3 |
| head_dropout | 0.5 |
| batch_size (train) | 32 |
| batch_size (eval) | 256 |
| optimizer | AdamW |
| lr | 0.003 |
| weight_decay | 0.01 |
| scheduler | OneCycleLR(max_lr=0.003) |
| max_epochs | 100 |
| early_stop patience | 15 |
| early_stop ema_alpha | 0.3 |
| grad_clip max_norm | 1.0 |
| seed | 42 |

## Training

- Stopped at epoch: 74 (early_stopping)
- Best smoothed val macro-F1: 0.8224 at epoch 59
- Best checkpoint: checkpoints/lstm/lstm_seed42.pt
- NaN detected: False

## Training curve (first 5 epochs)

| Epoch | Train Loss | Val Macro-F1 | Smoothed F1 | Bull F1 | Bear F1 |
|-------|-----------|-------------|-------------|---------|---------|
|   1 | 1.0898 | 0.2743 | 0.2743 | 0.0086 | 0.2632 |
|   2 | 1.0603 | 0.3842 | 0.3073 | 0.2618 | 0.3067 |
|   3 | 0.9738 | 0.4152 | 0.3396 | 0.3564 | 0.4984 |
|   4 | 0.7714 | 0.6173 | 0.4229 | 0.5407 | 0.5699 |
|   5 | 0.6215 | 0.6583 | 0.4936 | 0.5663 | 0.6296 |


## Training curve (last 5 epochs)

| Epoch | Train Loss | Val Macro-F1 | Smoothed F1 | Bull F1 | Bear F1 |
|-------|-----------|-------------|-------------|---------|---------|
|  70 | 0.1164 | 0.8019 | 0.8108 | 0.7839 | 0.7065 |
|  71 | 0.1118 | 0.8235 | 0.8146 | 0.8031 | 0.7506 |
|  72 | 0.1000 | 0.7956 | 0.8089 | 0.7717 | 0.7059 |
|  73 | 0.1083 | 0.8124 | 0.8099 | 0.7831 | 0.7447 |
|  74 | 0.1040 | 0.8119 | 0.8105 | 0.8008 | 0.7186 |


## Validation results (best checkpoint)

```
              precision    recall  f1-score   support

        none     0.9396    0.8911    0.9147      1240
        bull     0.6938    0.9261    0.7933       230
        bear     0.8093    0.7632    0.7856       228

    accuracy                         0.8787      1698
   macro avg     0.8142    0.8601    0.8312      1698
weighted avg     0.8888    0.8787    0.8809      1698

```

Confusion matrix (val) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       1105        94        41
        bull         17       213         0
        bear         54         0       174
```

## Test results (primary)

```
              precision    recall  f1-score   support

        none     0.9357    0.8990    0.9169      2573
        bull     0.6936    0.8593    0.7676       519
        bear     0.8147    0.7631    0.7881       363

    accuracy                         0.8787      3455
   macro avg     0.8147    0.8405    0.8242      3455
weighted avg     0.8866    0.8787    0.8810      3455

```

Confusion matrix (test) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       2313       197        63
        bull         73       446         0
        bear         86         0       277
```

## Comparison to XGBoost floor and naive baseline

| Metric | Naive | XGBoost | LSTM | LSTM vs XGBoost |
|--------|-------|---------|------|-----------------|
| Macro-F1 | 0.2846 | 0.6084 | 0.8242 | +0.2158 |
| Bull F1  | 0.0000 | 0.5519 | 0.7676 | +0.2157 |
| Bear F1  | 0.0000 | 0.5183 | 0.7881 | +0.2698 |

## Interpretation

LSTM test macro-F1 (0.8242) meets or exceeds XGBoost floor (0.6084). DL pipeline viable.

## Phase 4 handoff to CNN-LSTM

- LSTM test macro-F1: 0.8242
- Model saved at: checkpoints/lstm/lstm_seed42.pt
- Infrastructure (loss.py, early_stop.py, train_utils.py) reusable unchanged for CNN-LSTM.
- CNN-LSTM must beat LSTM macro-F1 (0.8242) on the same test split to justify the architecture.
