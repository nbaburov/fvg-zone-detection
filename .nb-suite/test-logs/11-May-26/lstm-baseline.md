# LSTM Baseline Evaluation — Phase 4 Step 1

**Date:** 2026-05-12
**Git SHA:** 4f64f03
**Python:** 3.12.13
**PyTorch:** 2.11.0
**Device:** cpu (forced — MPS LSTM bug, see .nb-suite/research/12-May-26/mps-gpu-fix.md)
**Seed:** 42

## Model architecture

- hidden_size: 64, num_layers: 2, dropout: 0.3, head_dropout: 0.5
- bidirectional: False (unidirectional — safe at label position 59)
- Total parameters: 51,651

## Dataset summary

| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|--------------|
| Train | 10518 | 2545 | 24.2% |
| Val   | 1698 | 458 | 27.0% |
| Test  | 5198 | 1312 | 25.2% |

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

- Stopped at epoch: 100 (max_epochs)
- Best smoothed val macro-F1: 0.8904 at epoch 100
- Best checkpoint: checkpoints/lstm/lstm_seed42_rawfvg.pt
- NaN detected: False

## Training curve (first 5 epochs)

| Epoch | Train Loss | Val Macro-F1 | Smoothed F1 | Bull F1 | Bear F1 |
|-------|-----------|-------------|-------------|---------|---------|
|   1 | 1.0923 | 0.2959 | 0.2959 | 0.0085 | 0.2655 |
|   2 | 1.0590 | 0.4242 | 0.3344 | 0.3719 | 0.3747 |
|   3 | 0.8535 | 0.5711 | 0.4054 | 0.4810 | 0.5907 |
|   4 | 0.6306 | 0.6824 | 0.4885 | 0.6579 | 0.6000 |
|   5 | 0.5341 | 0.7197 | 0.5579 | 0.6515 | 0.6889 |


## Training curve (last 5 epochs)

| Epoch | Train Loss | Val Macro-F1 | Smoothed F1 | Bull F1 | Bear F1 |
|-------|-----------|-------------|-------------|---------|---------|
|  96 | 0.0213 | 0.8907 | 0.8895 | 0.8850 | 0.8371 |
|  97 | 0.0241 | 0.8907 | 0.8898 | 0.8850 | 0.8371 |
|  98 | 0.0193 | 0.8907 | 0.8901 | 0.8850 | 0.8371 |
|  99 | 0.0209 | 0.8907 | 0.8903 | 0.8850 | 0.8371 |
| 100 | 0.0269 | 0.8907 | 0.8904 | 0.8850 | 0.8371 |


## Validation results (best checkpoint)

```
              precision    recall  f1-score   support

        none     0.9449    0.9548    0.9499      1240
        bull     0.8831    0.8870    0.8850       230
        bear     0.8645    0.8114    0.8371       228

    accuracy                         0.9264      1698
   macro avg     0.8975    0.8844    0.8907      1698
weighted avg     0.9258    0.9264    0.9259      1698

```

Confusion matrix (val) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       1184        27        29
        bull         26       204         0
        bear         43         0       185
```

## Test results (primary)

```
              precision    recall  f1-score   support

        none     0.9420    0.9532    0.9476      3886
        bull     0.8538    0.8409    0.8473       792
        bear     0.8601    0.8038    0.8310       520

    accuracy                         0.9211      5198
   macro avg     0.8853    0.8660    0.8753      5198
weighted avg     0.9204    0.9211    0.9206      5198

```

Confusion matrix (test) — rows=actual, cols=predicted (none/bull/bear):
```
                 none      bull      bear
        none       3704       114        68
        bull        126       666         0
        bear        102         0       418
```

## Comparison to XGBoost floor and naive baseline

| Metric | Naive | XGBoost | LSTM | LSTM vs XGBoost |
|--------|-------|---------|------|-----------------|
| Macro-F1 | 0.2846 | 0.6084 | 0.8753 | +0.2669 |
| Bull F1  | 0.0000 | 0.5519 | 0.8473 | +0.2954 |
| Bear F1  | 0.0000 | 0.5183 | 0.8310 | +0.3127 |

## Interpretation

LSTM test macro-F1 (0.8753) meets or exceeds XGBoost floor (0.6084). DL pipeline viable.

## Phase 4 handoff to CNN-LSTM

- LSTM test macro-F1: 0.8753
- Model saved at: checkpoints/lstm/lstm_seed42_rawfvg.pt
- Infrastructure (loss.py, early_stop.py, train_utils.py) reusable unchanged for CNN-LSTM.
- CNN-LSTM must beat LSTM macro-F1 (0.8753) on the same test split to justify the architecture.
