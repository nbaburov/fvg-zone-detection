# Bull vs Bear FVG Asymmetry Analysis

## Per-Seed Per-Class Metrics

| Seed | Bull F1 | Bear F1 | Bull-Bear Gap | Macro F1 |
|------|---------|---------|---------------|----------|
| cnn_lstm_seed0 | 0.4424 | 0.3394 | +0.1030 | 0.5829 |
| cnn_lstm_seed123 | 0.4472 | 0.5175 | -0.0703 | 0.6466 |
| cnn_lstm_seed17 | 0.4769 | 0.4052 | +0.0717 | 0.6198 |
| cnn_lstm_seed2024 | 0.4418 | 0.3883 | +0.0534 | 0.6033 |
| cnn_lstm_seed42 | 0.4444 | 0.4341 | +0.0103 | 0.6177 |

Mean bull F1: 0.4506 ± 0.0133
Mean bear F1: 0.4169 ± 0.0589
Mean bull-bear gap: +0.0336

## Conclusion

Systematic asymmetry (bull>bear by >0.05 across all seeds): **False**

If systematic: model likely benefits from bull FVG structural clarity vs bear.
Potential causes: SPY long-term upward bias creates more bull FVG instances in
training data; bull gap features (gap_bull, gap_norm_bull) may be more discriminative.

## Structural Gap Size Analysis

See `gap_size_distribution.html` for TP vs FP gap size comparison.
See `confusion_seed*.html` for per-seed confusion matrices.