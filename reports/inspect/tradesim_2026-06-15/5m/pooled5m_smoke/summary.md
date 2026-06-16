# Model Inspection Summary

- Windows: 233979
- FVG-positive windows: 7022 (3.0%)
- Models: cnn_lstm

## Per-model Metrics

| Model | F1 Bull | F1 Bear | F1 FVG Macro | F1 Binary FVG |
|-------|---------|---------|--------------|---------------|
| cnn_lstm | 0.5998 | 0.5552 | 0.5775 | 0.5785 |

## Precision / Recall (FVG classes)

| Model | Prec Bull | Rec Bull | Prec Bear | Rec Bear |
|-------|-----------|----------|-----------|----------|
| cnn_lstm | 0.5283 | 0.6938 | 0.4496 | 0.7257 |

## Confusion Matrices

### cnn_lstm

```
              pred_none  pred_bull  pred_bear
true_none        221761       2401       2795
true_bullish       1187       2689          0
true_bearish        863          0       2283
```

## Top Disagreement Windows

| Rank | Window | Timestamp | Disagreement | GT Label | cnn_lstm |
|------|--------|-----------|-------------|----------|---|
| 1 | 233978 | 2025-12-30 15:55 | 0.000 | none | none |
| 2 | 78001 | 2024-01-04 12:50 | 0.000 | none | none |
| 3 | 77999 | 2024-01-04 12:40 | 0.000 | none | none |
| 4 | 77998 | 2024-01-04 12:35 | 0.000 | none | none |
| 5 | 77997 | 2024-01-04 12:30 | 0.000 | none | none |
| 6 | 77996 | 2024-01-04 12:25 | 0.000 | none | none |
| 7 | 77995 | 2024-01-04 12:20 | 0.000 | none | none |
| 8 | 77994 | 2024-01-04 12:15 | 0.000 | none | none |
| 9 | 77993 | 2024-01-04 12:10 | 0.000 | none | none |
| 10 | 77992 | 2024-01-04 12:05 | 0.000 | none | none |
| 11 | 77991 | 2024-01-04 12:00 | 0.000 | none | none |
| 12 | 77990 | 2024-01-04 11:55 | 0.000 | none | bearish |
| 13 | 77989 | 2024-01-04 11:50 | 0.000 | none | none |
| 14 | 77988 | 2024-01-04 11:45 | 0.000 | none | none |
| 15 | 77987 | 2024-01-04 11:40 | 0.000 | none | none |
| 16 | 77986 | 2024-01-04 11:35 | 0.000 | none | none |
| 17 | 77985 | 2024-01-04 11:30 | 0.000 | none | none |
| 18 | 77984 | 2024-01-04 11:25 | 0.000 | none | none |
| 19 | 77983 | 2024-01-04 11:20 | 0.000 | none | none |
| 20 | 77982 | 2024-01-04 11:15 | 0.000 | none | none |

## Trade Outcomes (simulated)

Each positive prediction → bracket trade: entry at next H1 open, SL beyond opposite gap edge, TP = R×R-multiple. Win rate counts only decided trades.

| Model | Signals | Filled | TP | SL | Undecided | Win Rate | Total R | Avg R | after_cost_total_R | median_R | n_outlier_R |
|-------|---------|--------|----|----|-----------|---------|---------|------|--------------------|---------|------------|
| cnn_lstm | 10168 | 10168 | 2756 | 5869 | 1543 | 0.320 | +221.88 | +0.022 | -628.49 | -1.000 | 0 |

## Exit-Strategy Comparison

**Realistic mode — pre-registered primary metric: `after_cost_total_R`** (total R after ATR min-stop floor + round-trip costs).  `winsorized_total_R`, `median_R`, `n_outlier_R` are mandatory context.  Raw `total_R` shown for continuity with the pre-realism comparison.

> **Caveat:** limit entries modelled as filled at exact limit price on first bar touching that level (optimistic OHLC bar-level backtest assumption).  Overstates fill quality vs live execution.

### cnn_lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 10168 | 10168 | N/A | 0.320 | 2756 | 5869 | 1543 | +221.88 | +0.022 | -628.49 | -0.062 | -1.000 | 0 | +221.88 | 0.000 |
| ict_iofed | 10168 | 7079 | 0.696 | 0.312 | 1835 | 4053 | 1191 | -547.69 | -0.077 | -1047.76 | -0.148 | -1.000 | 1 | -550.75 | 0.000 |
| ce_50pct | 10168 | 6432 | 0.633 | 0.255 | 1384 | 4033 | 1015 | -412.83 | -0.064 | -939.74 | -0.146 | -1.000 | 1 | -414.42 | 0.000 |
| tradinglab | 10168 | 7079 | 0.696 | 0.246 | 1558 | 4776 | 745 | -385.45 | -0.054 | -1118.86 | -0.158 | -1.000 | 5 | -391.56 | 0.000 |
