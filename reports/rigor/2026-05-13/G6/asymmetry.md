# Bull vs Bear FVG Asymmetry Analysis

## Per-Seed Per-Class Metrics

| Seed | Bull F1 | Bear F1 | Bull-Bear Gap | Macro F1 |
|------|---------|---------|---------------|----------|
| lstm_seed0 | 0.4277 | 0.3866 | +0.0412 | 0.5934 |
| lstm_seed123 | 0.3937 | 0.3232 | +0.0705 | 0.5603 |
| lstm_seed17 | 0.4393 | 0.4649 | -0.0255 | 0.6257 |
| lstm_seed2024 | 0.4711 | 0.4278 | +0.0433 | 0.6254 |
| lstm_seed42 | 0.4199 | 0.3744 | +0.0455 | 0.5880 |

Mean bull F1: 0.4304 ± 0.0253
Mean bear F1: 0.3954 ± 0.0482
Mean bull-bear gap: +0.0350

## Conclusion

Systematic asymmetry (bull>bear by >0.05 across all seeds): **False**

If systematic: model likely benefits from bull FVG structural clarity vs bear.
Potential causes: SPY long-term upward bias creates more bull FVG instances in
training data; bull gap features (gap_bull, gap_norm_bull) may be more discriminative.

## Structural Gap Size Analysis

See `gap_size_distribution.html` for TP vs FP gap size comparison.
See `confusion_seed*.html` for per-seed confusion matrices.