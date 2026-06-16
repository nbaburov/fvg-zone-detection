# Model Inspection Summary

- Windows: 5198
- FVG-positive windows: 163 (3.1%)
- Models: cnn_lstm, lstm, transformer, xgboost

## Per-model Metrics

| Model | F1 Bull | F1 Bear | F1 FVG Macro | F1 Binary FVG |
|-------|---------|---------|--------------|---------------|
| cnn_lstm | 0.4444 | 0.4341 | 0.4393 | 0.4415 |
| lstm | 0.4228 | 0.3922 | 0.4075 | 0.4089 |
| transformer | 0.3651 | 0.2833 | 0.3242 | 0.3249 |
| xgboost | 0.6047 | 0.5780 | 0.5913 | 0.5940 |

## Precision / Recall (FVG classes)

| Model | Prec Bull | Rec Bull | Prec Bear | Rec Bear |
|-------|-----------|----------|-----------|----------|
| cnn_lstm | 0.3158 | 0.7500 | 0.4516 | 0.4179 |
| lstm | 0.3467 | 0.5417 | 0.2920 | 0.5970 |
| transformer | 0.3034 | 0.4583 | 0.1988 | 0.4925 |
| xgboost | 0.4815 | 0.8125 | 0.4717 | 0.7463 |

## Confusion Matrices

### cnn_lstm

```
              pred_none  pred_bull  pred_bear
true_none          4845        156         34
true_bullish         24         72          0
true_bearish         39          0         28
```

### lstm

```
              pred_none  pred_bull  pred_bear
true_none          4840         98         97
true_bullish         44         52          0
true_bearish         27          0         40
```

### transformer

```
              pred_none  pred_bull  pred_bear
true_none          4801        101        133
true_bullish         52         44          0
true_bearish         34          0         33
```

### xgboost

```
              pred_none  pred_bull  pred_bear
true_none          4895         84         56
true_bullish         18         78          0
true_bearish         17          0         50
```

## Agreement Matrix

Fraction of windows where each pair of models predicts the same class.

| | cnn_lstm | lstm | transformer | xgboost |
|---|---|---|---|---|
| **cnn_lstm** | 1.000 | 0.948 | 0.921 | 0.956 |
| **lstm** | 0.948 | 1.000 | 0.928 | 0.946 |
| **transformer** | 0.921 | 0.928 | 1.000 | 0.931 |
| **xgboost** | 0.956 | 0.946 | 0.931 | 1.000 |

## Top Disagreement Windows

| Rank | Window | Timestamp | Disagreement | GT Label | cnn_lstm | lstm | transformer | xgboost |
|------|--------|-----------|-------------|----------|---|---|---|---|
| 1 | 3033 | 2024-10-04 14:30 | 0.500 | none | none | none | bullish | none |
| 2 | 3400 | 2024-12-19 10:30 | 0.500 | none | none | none | bearish | none |
| 3 | 3976 | 2025-04-22 12:30 | 0.500 | none | none | none | none | bullish |
| 4 | 2056 | 2024-03-18 10:30 | 0.500 | none | bullish | bullish | none | none |
| 5 | 478 | 2023-04-24 14:30 | 0.500 | bullish | bullish | bullish | none | bullish |
| 6 | 479 | 2023-04-24 15:30 | 0.500 | none | bullish | none | none | none |
| 7 | 2055 | 2024-03-18 09:30 | 0.500 | none | none | none | bullish | none |
| 8 | 2054 | 2024-03-15 15:30 | 0.500 | none | none | none | bullish | none |
| 9 | 1581 | 2023-12-07 11:30 | 0.500 | bullish | bullish | bullish | none | bullish |
| 10 | 2869 | 2024-09-03 11:30 | 0.500 | none | none | bearish | bearish | none |
| 11 | 3401 | 2024-12-19 11:30 | 0.500 | none | none | none | bearish | none |
| 12 | 3399 | 2024-12-19 09:30 | 0.500 | none | none | none | bearish | none |
| 13 | 2092 | 2024-03-25 11:30 | 0.500 | none | none | bearish | none | none |
| 14 | 3398 | 2024-12-18 15:30 | 0.500 | none | none | none | bearish | none |
| 15 | 3397 | 2024-12-18 14:30 | 0.500 | none | none | none | bearish | none |
| 16 | 490 | 2023-04-26 12:30 | 0.500 | none | bullish | bullish | bullish | none |
| 17 | 4647 | 2025-09-09 11:30 | 0.500 | none | none | none | bearish | none |
| 18 | 2046 | 2024-03-14 14:30 | 0.500 | none | none | none | bearish | none |
| 19 | 2876 | 2024-09-04 11:30 | 0.500 | none | none | none | bullish | none |
| 20 | 2044 | 2024-03-14 12:30 | 0.500 | none | none | none | bearish | none |

## Trade Outcomes (simulated)

Each positive prediction → bracket trade: entry at next H1 open, SL beyond opposite gap edge, TP = R×R-multiple. Win rate counts only decided trades.

| Model | Signals | Filled | TP | SL | Undecided | Win Rate | Total R | Avg R | after_cost_total_R | median_R | n_outlier_R |
|-------|---------|--------|----|----|-----------|---------|---------|------|--------------------|---------|------------|
| cnn_lstm | 290 | 290 | 84 | 141 | 65 | 0.373 | +49.43 | +0.170 | +45.18 | -0.471 | 0 |
| lstm | 287 | 287 | 73 | 156 | 58 | 0.319 | +10.88 | +0.038 | +6.23 | -1.000 | 0 |
| transformer | 311 | 311 | 74 | 158 | 79 | 0.319 | +9.23 | +0.030 | +4.06 | -1.000 | 0 |
| xgboost | 268 | 268 | 69 | 153 | 46 | 0.311 | +1.99 | +0.007 | -2.09 | -1.000 | 0 |

## Exit-Strategy Comparison

**Realistic mode — pre-registered primary metric: `after_cost_total_R`** (total R after ATR min-stop floor + round-trip costs).  `winsorized_total_R`, `median_R`, `n_outlier_R` are mandatory context.  Raw `total_R` shown for continuity with the pre-realism comparison.

> **Caveat:** limit entries modelled as filled at exact limit price on first bar touching that level (optimistic OHLC bar-level backtest assumption).  Overstates fill quality vs live execution.

### cnn_lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 290 | 290 | N/A | 0.373 | 84 | 141 | 65 | +49.43 | +0.170 | +45.18 | +0.156 | -0.471 | 0 | +49.43 | 0.000 |
| ict_iofed | 290 | 209 | 0.721 | 0.365 | 69 | 120 | 20 | +9.64 | +0.046 | +7.39 | +0.035 | -1.000 | 0 | +9.64 | 0.000 |
| ce_50pct | 290 | 170 | 0.586 | 0.252 | 39 | 116 | 15 | +8.34 | +0.049 | +6.05 | +0.036 | -1.000 | 0 | +8.34 | 0.000 |
| tradinglab | 290 | 209 | 0.721 | 0.316 | 61 | 132 | 16 | +56.52 | +0.270 | +52.62 | +0.252 | -1.000 | 2 | +52.59 | 0.000 |

### lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 287 | 287 | N/A | 0.319 | 73 | 156 | 58 | +10.88 | +0.038 | +6.23 | +0.022 | -1.000 | 0 | +10.88 | 0.000 |
| ict_iofed | 287 | 212 | 0.739 | 0.332 | 62 | 125 | 25 | -23.52 | -0.111 | -26.02 | -0.123 | -1.000 | 0 | -23.52 | 0.000 |
| ce_50pct | 287 | 186 | 0.648 | 0.268 | 44 | 120 | 22 | -10.21 | -0.055 | -12.79 | -0.069 | -1.000 | 0 | -10.21 | 0.000 |
| tradinglab | 287 | 212 | 0.739 | 0.251 | 50 | 149 | 13 | -10.87 | -0.051 | -15.06 | -0.071 | -1.000 | 2 | -15.13 | 0.000 |

### transformer

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 311 | 311 | N/A | 0.319 | 74 | 158 | 79 | +9.23 | +0.030 | +4.06 | +0.013 | -1.000 | 0 | +9.23 | 0.000 |
| ict_iofed | 311 | 208 | 0.669 | 0.354 | 62 | 113 | 33 | +41.87 | +0.201 | +38.48 | +0.185 | -1.000 | 0 | +41.87 | 0.000 |
| ce_50pct | 311 | 196 | 0.630 | 0.251 | 44 | 131 | 21 | +18.91 | +0.097 | +15.23 | +0.078 | -1.000 | 0 | +18.91 | 0.000 |
| tradinglab | 311 | 208 | 0.669 | 0.291 | 53 | 129 | 26 | +41.31 | +0.199 | +36.91 | +0.177 | -1.000 | 1 | +40.20 | 0.000 |

### xgboost

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 268 | 268 | N/A | 0.311 | 69 | 153 | 46 | +1.99 | +0.007 | -2.09 | -0.008 | -1.000 | 0 | +1.99 | 0.000 |
| ict_iofed | 268 | 197 | 0.735 | 0.306 | 52 | 118 | 27 | -25.32 | -0.129 | -27.68 | -0.141 | -1.000 | 0 | -25.32 | 0.000 |
| ce_50pct | 268 | 168 | 0.627 | 0.224 | 33 | 114 | 21 | -21.68 | -0.129 | -24.19 | -0.144 | -1.000 | 0 | -21.68 | 0.000 |
| tradinglab | 268 | 197 | 0.735 | 0.220 | 39 | 138 | 20 | -0.88 | -0.004 | -4.76 | -0.024 | -1.000 | 2 | -4.81 | 0.000 |

## All-Seeds Aggregate

5-seed mean ± std across multisym checkpoints.  Pre-registered primary metric: `after_cost_total_R` (mean±std).  Per-seed rows follow each strategy block.

### cnn_lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | +13.15 ± 12.61 | 0.327 ± 0.019 | 308.8 |
| ict_iofed | 5 | -26.25 ± 8.42 | 0.317 ± 0.017 | 226.4 |
| ce_50pct | 5 | -20.78 ± 11.29 | 0.240 ± 0.021 | 193.6 |
| tradinglab | 5 | +3.56 ± 8.89 | 0.259 ± 0.013 | 226.4 |

#### cnn_lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +31.50 | 0.352 | 311 |
| 17 | +8.24 | 0.321 | 287 |
| 42 | -2.69 | 0.300 | 289 |
| 123 | +17.60 | 0.333 | 324 |
| 2024 | +11.09 | 0.327 | 333 |

#### cnn_lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -14.58 | 0.342 | 227 |
| 17 | -24.73 | 0.318 | 215 |
| 42 | -29.82 | 0.309 | 209 |
| 123 | -24.53 | 0.319 | 235 |
| 2024 | -37.61 | 0.296 | 246 |

#### cnn_lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -5.91 | 0.269 | 194 |
| 17 | -18.93 | 0.241 | 184 |
| 42 | -27.01 | 0.220 | 175 |
| 123 | -16.22 | 0.253 | 203 |
| 2024 | -35.82 | 0.219 | 212 |

#### cnn_lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +16.68 | 0.278 | 227 |
| 17 | +0.26 | 0.254 | 215 |
| 42 | -2.59 | 0.247 | 209 |
| 123 | +8.37 | 0.265 | 235 |
| 2024 | -4.93 | 0.250 | 246 |

### lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | +21.26 ± 14.66 | 0.343 ± 0.017 | 305.8 |
| ict_iofed | 5 | -7.65 ± 18.39 | 0.343 ± 0.024 | 218.2 |
| ce_50pct | 5 | +4.20 ± 19.86 | 0.282 ± 0.028 | 191.2 |
| tradinglab | 5 | +19.73 ± 25.73 | 0.282 ± 0.020 | 218.2 |

#### lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +32.93 | 0.352 | 332 |
| 17 | +23.87 | 0.345 | 315 |
| 42 | +36.58 | 0.364 | 299 |
| 123 | +1.81 | 0.322 | 304 |
| 2024 | +11.11 | 0.330 | 279 |

#### lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +2.50 | 0.335 | 234 |
| 17 | +13.73 | 0.379 | 220 |
| 42 | -13.41 | 0.340 | 213 |
| 123 | -5.91 | 0.347 | 225 |
| 2024 | -35.15 | 0.315 | 199 |

#### lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +11.47 | 0.281 | 206 |
| 17 | +35.95 | 0.327 | 193 |
| 42 | -7.22 | 0.275 | 186 |
| 123 | -8.02 | 0.277 | 196 |
| 2024 | -11.20 | 0.250 | 175 |

#### lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +44.52 | 0.296 | 234 |
| 17 | +42.82 | 0.304 | 220 |
| 42 | +20.23 | 0.286 | 213 |
| 123 | +8.35 | 0.264 | 225 |
| 2024 | -17.29 | 0.259 | 199 |

### transformer

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | +13.63 ± 20.15 | 0.266 ± 0.152 | 202.8 |
| ict_iofed | 5 | -9.56 ± 9.98 | 0.290 ± 0.163 | 145.4 |
| ce_50pct | 5 | -0.83 ± 11.87 | 0.222 ± 0.125 | 124.4 |
| tradinglab | 5 | +8.83 ± 9.46 | 0.234 ± 0.133 | 145.4 |

#### transformer / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +12.79 | 0.326 | 225 |
| 17 | -11.37 | 0.286 | 211 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | +29.37 | 0.370 | 213 |
| 2024 | +37.38 | 0.347 | 365 |

#### transformer / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -15.03 | 0.383 | 159 |
| 17 | -16.43 | 0.359 | 151 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | +2.46 | 0.363 | 155 |
| 2024 | -18.81 | 0.347 | 262 |

#### transformer / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -3.31 | 0.281 | 136 |
| 17 | -15.37 | 0.254 | 129 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | +17.62 | 0.294 | 134 |
| 2024 | -3.09 | 0.281 | 223 |

#### transformer / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +14.85 | 0.329 | 159 |
| 17 | +3.66 | 0.275 | 151 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | +22.43 | 0.300 | 155 |
| 2024 | +3.20 | 0.264 | 262 |

### xgboost

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -14.06 ± 4.13 | 0.302 ± 0.004 | 283.8 |
| ict_iofed | 5 | -32.62 ± 7.04 | 0.299 ± 0.006 | 208.4 |
| ce_50pct | 5 | -27.70 ± 7.29 | 0.227 ± 0.005 | 182.2 |
| tradinglab | 5 | -1.63 ± 8.72 | 0.235 ± 0.004 | 208.4 |

#### xgboost / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -17.01 | 0.299 | 285 |
| 17 | -10.60 | 0.301 | 278 |
| 42 | -12.61 | 0.305 | 280 |
| 123 | -19.70 | 0.296 | 293 |
| 2024 | -10.37 | 0.307 | 283 |

#### xgboost / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -34.45 | 0.301 | 207 |
| 17 | -41.49 | 0.293 | 204 |
| 42 | -21.90 | 0.308 | 207 |
| 123 | -33.52 | 0.297 | 216 |
| 2024 | -31.72 | 0.295 | 208 |

#### xgboost / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -30.41 | 0.224 | 180 |
| 17 | -37.14 | 0.224 | 179 |
| 42 | -16.98 | 0.236 | 181 |
| 123 | -27.31 | 0.225 | 188 |
| 2024 | -26.66 | 0.227 | 183 |

#### xgboost / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -0.51 | 0.235 | 207 |
| 17 | -13.01 | 0.232 | 204 |
| 42 | +11.18 | 0.242 | 207 |
| 123 | -1.15 | 0.235 | 216 |
| 2024 | -4.63 | 0.230 | 208 |
