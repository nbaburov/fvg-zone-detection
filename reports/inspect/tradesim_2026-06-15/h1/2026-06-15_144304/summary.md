# Model Inspection Summary

- Windows: 5198
- FVG-positive windows: 172 (3.3%)
- Models: cnn_lstm, lstm, transformer, xgboost

## Per-model Metrics

| Model | F1 Bull | F1 Bear | F1 FVG Macro | F1 Binary FVG |
|-------|---------|---------|--------------|---------------|
| cnn_lstm | 0.4246 | 0.3357 | 0.3801 | 0.3974 |
| lstm | 0.4595 | 0.3645 | 0.4120 | 0.4128 |
| transformer | 0.3866 | 0.3047 | 0.3456 | 0.3441 |
| xgboost | 0.6102 | 0.4834 | 0.5468 | 0.5503 |

## Precision / Recall (FVG classes)

| Model | Prec Bull | Rec Bull | Prec Bear | Rec Bear |
|-------|-----------|----------|-----------|----------|
| cnn_lstm | 0.3000 | 0.7263 | 0.3636 | 0.3117 |
| lstm | 0.4016 | 0.5368 | 0.2847 | 0.5065 |
| transformer | 0.3217 | 0.4842 | 0.2179 | 0.5065 |
| xgboost | 0.5106 | 0.7579 | 0.3806 | 0.6623 |

## Confusion Matrices

### cnn_lstm

```
              pred_none  pred_bull  pred_bear
true_none          4823        161         42
true_bullish         26         69          0
true_bearish         53          0         24
```

### lstm

```
              pred_none  pred_bull  pred_bear
true_none          4852         76         98
true_bullish         44         51          0
true_bearish         38          0         39
```

### transformer

```
              pred_none  pred_bull  pred_bear
true_none          4789         97        140
true_bullish         49         46          0
true_bearish         38          0         39
```

### xgboost

```
              pred_none  pred_bull  pred_bear
true_none          4874         69         83
true_bullish         23         72          0
true_bearish         26          0         51
```

## Agreement Matrix

Fraction of windows where each pair of models predicts the same class.

| | cnn_lstm | lstm | transformer | xgboost |
|---|---|---|---|---|
| **cnn_lstm** | 1.000 | 0.944 | 0.921 | 0.948 |
| **lstm** | 0.944 | 1.000 | 0.928 | 0.940 |
| **transformer** | 0.921 | 0.928 | 1.000 | 0.928 |
| **xgboost** | 0.948 | 0.940 | 0.928 | 1.000 |

## Top Disagreement Windows

| Rank | Window | Timestamp | Disagreement | GT Label | cnn_lstm | lstm | transformer | xgboost |
|------|--------|-----------|-------------|----------|---|---|---|---|
| 1 | 3930 | 2025-04-10 15:30 | 0.500 | none | none | none | bullish | none |
| 2 | 1687 | 2023-12-29 12:30 | 0.500 | none | bearish | bearish | bearish | none |
| 3 | 4379 | 2025-07-16 09:30 | 0.500 | none | none | none | bearish | none |
| 4 | 4745 | 2025-09-29 11:30 | 0.500 | none | none | bullish | none | none |
| 5 | 4744 | 2025-09-29 10:30 | 0.500 | none | bullish | bullish | none | none |
| 6 | 4380 | 2025-07-16 10:30 | 0.500 | none | none | none | bearish | none |
| 7 | 4381 | 2025-07-16 11:30 | 0.500 | none | none | none | bearish | none |
| 8 | 4741 | 2025-09-26 14:30 | 0.500 | none | bullish | bullish | none | none |
| 9 | 4740 | 2025-09-26 13:30 | 0.500 | none | bullish | bullish | bullish | none |
| 10 | 4739 | 2025-09-26 12:30 | 0.500 | none | none | none | bullish | none |
| 11 | 388 | 2023-04-04 15:30 | 0.500 | none | none | bearish | none | none |
| 12 | 3908 | 2025-04-07 14:30 | 0.500 | none | bullish | bullish | none | none |
| 13 | 4382 | 2025-07-16 12:30 | 0.500 | none | none | none | bearish | none |
| 14 | 3435 | 2024-12-27 10:30 | 0.500 | bearish | bearish | bearish | none | bearish |
| 15 | 392 | 2023-04-05 12:30 | 0.500 | none | none | none | none | bearish |
| 16 | 3911 | 2025-04-08 10:30 | 0.500 | none | bullish | bullish | none | bullish |
| 17 | 3912 | 2025-04-08 11:30 | 0.500 | none | bullish | bullish | none | bullish |
| 18 | 1126 | 2023-09-06 11:30 | 0.500 | none | none | bearish | bearish | none |
| 19 | 1125 | 2023-09-06 10:30 | 0.500 | none | none | bearish | none | bearish |
| 20 | 4747 | 2025-09-29 13:30 | 0.500 | none | bearish | none | none | none |

## Trade Outcomes (simulated)

Each positive prediction → bracket trade: entry at next H1 open, SL beyond opposite gap edge, TP = R×R-multiple. Win rate counts only decided trades.

| Model | Signals | Filled | TP | SL | Undecided | Win Rate | Total R | Avg R | after_cost_total_R | median_R | n_outlier_R |
|-------|---------|--------|----|----|-----------|---------|---------|------|--------------------|---------|------------|
| cnn_lstm | 296 | 296 | 77 | 155 | 64 | 0.332 | +28.33 | +0.096 | +24.63 | -1.000 | 0 |
| lstm | 264 | 264 | 48 | 156 | 60 | 0.235 | -37.51 | -0.142 | -41.35 | -1.000 | 0 |
| transformer | 322 | 322 | 68 | 165 | 89 | 0.292 | -4.61 | -0.014 | -9.22 | -1.000 | 0 |
| xgboost | 275 | 275 | 65 | 150 | 60 | 0.302 | +1.24 | +0.005 | -2.36 | -1.000 | 0 |

## Exit-Strategy Comparison

**Realistic mode — pre-registered primary metric: `after_cost_total_R`** (total R after ATR min-stop floor + round-trip costs).  `winsorized_total_R`, `median_R`, `n_outlier_R` are mandatory context.  Raw `total_R` shown for continuity with the pre-realism comparison.

> **Caveat:** limit entries modelled as filled at exact limit price on first bar touching that level (optimistic OHLC bar-level backtest assumption).  Overstates fill quality vs live execution.

### cnn_lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 296 | 296 | N/A | 0.332 | 77 | 155 | 64 | +28.33 | +0.096 | +24.63 | +0.083 | -1.000 | 0 | +28.33 | 0.000 |
| ict_iofed | 296 | 212 | 0.716 | 0.358 | 67 | 120 | 25 | +41.67 | +0.197 | +39.56 | +0.187 | -1.000 | 1 | +40.05 | 0.000 |
| ce_50pct | 296 | 176 | 0.595 | 0.248 | 39 | 118 | 19 | +25.62 | +0.146 | +23.49 | +0.133 | -1.000 | 0 | +25.62 | 0.000 |
| tradinglab | 296 | 212 | 0.716 | 0.286 | 57 | 142 | 13 | +51.02 | +0.241 | +47.59 | +0.225 | -1.000 | 2 | +46.10 | 0.000 |

### lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 264 | 264 | N/A | 0.235 | 48 | 156 | 60 | -37.51 | -0.142 | -41.35 | -0.157 | -1.000 | 0 | -37.51 | 0.000 |
| ict_iofed | 264 | 198 | 0.750 | 0.295 | 51 | 122 | 25 | -14.72 | -0.074 | -17.14 | -0.087 | -1.000 | 0 | -14.72 | 0.000 |
| ce_50pct | 264 | 182 | 0.689 | 0.239 | 39 | 124 | 19 | -9.37 | -0.051 | -11.90 | -0.065 | -1.000 | 0 | -9.37 | 0.000 |
| tradinglab | 264 | 198 | 0.750 | 0.215 | 40 | 146 | 12 | -9.63 | -0.049 | -13.18 | -0.067 | -1.000 | 2 | -12.86 | 0.000 |

### transformer

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 322 | 322 | N/A | 0.292 | 68 | 165 | 89 | -4.61 | -0.014 | -9.22 | -0.029 | -1.000 | 0 | -4.61 | 0.000 |
| ict_iofed | 322 | 213 | 0.661 | 0.323 | 61 | 128 | 24 | -5.20 | -0.024 | -8.18 | -0.038 | -1.000 | 1 | -7.51 | 0.000 |
| ce_50pct | 322 | 204 | 0.634 | 0.246 | 47 | 144 | 13 | -17.57 | -0.086 | -20.71 | -0.102 | -1.000 | 1 | -19.84 | 0.000 |
| tradinglab | 322 | 213 | 0.661 | 0.263 | 52 | 146 | 15 | -0.14 | -0.001 | -3.97 | -0.019 | -1.000 | 0 | -0.14 | 0.000 |

### xgboost

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 275 | 275 | N/A | 0.302 | 65 | 150 | 60 | +1.24 | +0.005 | -2.36 | -0.009 | -1.000 | 0 | +1.24 | 0.000 |
| ict_iofed | 275 | 195 | 0.709 | 0.326 | 56 | 116 | 23 | +5.86 | +0.030 | +3.71 | +0.019 | -1.000 | 0 | +5.86 | 0.000 |
| ce_50pct | 275 | 167 | 0.607 | 0.257 | 39 | 113 | 15 | +2.94 | +0.018 | +0.65 | +0.004 | -1.000 | 1 | +2.89 | 0.000 |
| tradinglab | 275 | 195 | 0.709 | 0.237 | 44 | 142 | 9 | -1.69 | -0.009 | -4.92 | -0.025 | -1.000 | 2 | -8.44 | 0.000 |

## All-Seeds Aggregate

5-seed mean ± std across multisym checkpoints.  Pre-registered primary metric: `after_cost_total_R` (mean±std).  Per-seed rows follow each strategy block.

### cnn_lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -2.70 ± 10.68 | 0.292 ± 0.013 | 294.4 |
| ict_iofed | 5 | +0.70 ± 10.94 | 0.332 ± 0.015 | 205.8 |
| ce_50pct | 5 | +5.78 ± 3.57 | 0.270 ± 0.009 | 179.4 |
| tradinglab | 5 | +15.88 ± 14.23 | 0.249 ± 0.021 | 205.8 |

#### cnn_lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +0.84 | 0.299 | 284 |
| 17 | -16.17 | 0.276 | 269 |
| 42 | -7.06 | 0.285 | 270 |
| 123 | -3.94 | 0.290 | 333 |
| 2024 | +12.84 | 0.311 | 316 |

#### cnn_lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +9.26 | 0.330 | 203 |
| 17 | -14.08 | 0.312 | 192 |
| 42 | -2.19 | 0.329 | 187 |
| 123 | -3.05 | 0.333 | 234 |
| 2024 | +13.54 | 0.354 | 213 |

#### cnn_lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +9.05 | 0.265 | 178 |
| 17 | +1.12 | 0.266 | 169 |
| 42 | +3.06 | 0.262 | 162 |
| 123 | +9.02 | 0.284 | 207 |
| 2024 | +6.64 | 0.274 | 181 |

#### cnn_lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +24.93 | 0.250 | 203 |
| 17 | -9.27 | 0.215 | 192 |
| 42 | +23.69 | 0.250 | 187 |
| 123 | +19.90 | 0.253 | 234 |
| 2024 | +20.16 | 0.275 | 213 |

### lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -5.45 ± 18.00 | 0.294 ± 0.029 | 293.4 |
| ict_iofed | 5 | +13.22 ± 23.88 | 0.330 ± 0.031 | 214.2 |
| ce_50pct | 5 | +18.41 ± 19.48 | 0.281 ± 0.030 | 193.2 |
| tradinglab | 5 | +21.73 ± 24.05 | 0.256 ± 0.022 | 214.2 |

#### lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -19.94 | 0.280 | 317 |
| 17 | -15.00 | 0.283 | 318 |
| 42 | +13.03 | 0.320 | 281 |
| 123 | +15.18 | 0.327 | 286 |
| 2024 | -20.53 | 0.257 | 265 |

#### lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +10.14 | 0.280 | 232 |
| 17 | -6.60 | 0.327 | 229 |
| 42 | +8.37 | 0.348 | 210 |
| 123 | +54.21 | 0.362 | 204 |
| 2024 | -0.04 | 0.333 | 196 |

#### lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +13.59 | 0.236 | 214 |
| 17 | +5.20 | 0.280 | 205 |
| 42 | +10.71 | 0.295 | 189 |
| 123 | +52.84 | 0.318 | 182 |
| 2024 | +9.73 | 0.277 | 176 |

#### lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +20.36 | 0.220 | 232 |
| 17 | +3.70 | 0.250 | 229 |
| 42 | +20.85 | 0.269 | 210 |
| 123 | +61.71 | 0.276 | 204 |
| 2024 | +2.03 | 0.265 | 196 |

### transformer

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -8.76 ± 14.47 | 0.216 ± 0.124 | 198.4 |
| ict_iofed | 5 | -3.62 ± 11.28 | 0.284 ± 0.162 | 137.6 |
| ce_50pct | 5 | +0.38 ± 11.34 | 0.217 ± 0.123 | 116.2 |
| tradinglab | 5 | +8.70 ± 7.50 | 0.224 ± 0.127 | 137.6 |

#### transformer / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +4.11 | 0.302 | 216 |
| 17 | -32.88 | 0.227 | 219 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -5.23 | 0.277 | 210 |
| 2024 | -9.80 | 0.275 | 347 |

#### transformer / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +10.84 | 0.387 | 144 |
| 17 | -9.33 | 0.360 | 155 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -0.38 | 0.367 | 146 |
| 2024 | -19.23 | 0.306 | 243 |

#### transformer / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +18.93 | 0.298 | 125 |
| 17 | -5.31 | 0.265 | 125 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -0.39 | 0.275 | 120 |
| 2024 | -11.32 | 0.245 | 211 |

#### transformer / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +20.17 | 0.311 | 144 |
| 17 | +4.59 | 0.280 | 155 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | +9.32 | 0.278 | 146 |
| 2024 | +9.43 | 0.250 | 243 |

### xgboost

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | +18.65 ± 1.98 | 0.334 ± 0.003 | 282.2 |
| ict_iofed | 5 | -6.52 ± 4.48 | 0.324 ± 0.008 | 197.6 |
| ce_50pct | 5 | +0.48 ± 4.91 | 0.276 ± 0.007 | 174.0 |
| tradinglab | 5 | -3.98 ± 4.79 | 0.238 ± 0.008 | 197.6 |

#### xgboost / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +19.11 | 0.335 | 284 |
| 17 | +17.49 | 0.332 | 279 |
| 42 | +21.24 | 0.338 | 281 |
| 123 | +19.41 | 0.333 | 284 |
| 2024 | +16.02 | 0.330 | 283 |

#### xgboost / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -9.53 | 0.320 | 197 |
| 17 | -0.65 | 0.337 | 196 |
| 42 | -7.09 | 0.322 | 198 |
| 123 | -3.58 | 0.326 | 199 |
| 2024 | -11.77 | 0.315 | 198 |

#### xgboost / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -2.57 | 0.273 | 174 |
| 17 | +5.77 | 0.287 | 171 |
| 42 | -0.13 | 0.275 | 174 |
| 123 | +5.02 | 0.280 | 176 |
| 2024 | -5.70 | 0.267 | 175 |

#### xgboost / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -6.73 | 0.232 | 197 |
| 17 | +4.28 | 0.250 | 196 |
| 42 | -4.56 | 0.237 | 198 |
| 123 | -5.11 | 0.240 | 199 |
| 2024 | -7.78 | 0.230 | 198 |
