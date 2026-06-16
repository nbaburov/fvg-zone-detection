# Model Inspection Summary

- Windows: 5198
- FVG-positive windows: 182 (3.5%)
- Models: cnn_lstm, lstm, transformer, xgboost

## Per-model Metrics

| Model | F1 Bull | F1 Bear | F1 FVG Macro | F1 Binary FVG |
|-------|---------|---------|--------------|---------------|
| cnn_lstm | 0.3976 | 0.3584 | 0.3780 | 0.3840 |
| lstm | 0.3739 | 0.4000 | 0.3870 | 0.3865 |
| transformer | 0.3024 | 0.2632 | 0.2828 | 0.2818 |
| xgboost | 0.5579 | 0.4922 | 0.5251 | 0.5235 |

## Precision / Recall (FVG classes)

| Model | Prec Bull | Rec Bull | Prec Bear | Rec Bear |
|-------|-----------|----------|-----------|----------|
| cnn_lstm | 0.2814 | 0.6771 | 0.3563 | 0.3605 |
| lstm | 0.3209 | 0.4479 | 0.3333 | 0.5000 |
| transformer | 0.2844 | 0.3229 | 0.2113 | 0.3488 |
| xgboost | 0.4745 | 0.6771 | 0.3706 | 0.7326 |

## Confusion Matrices

### cnn_lstm

```
              pred_none  pred_bull  pred_bear
true_none          4794        166         56
true_bullish         31         65          0
true_bearish         55          0         31
```

### lstm

```
              pred_none  pred_bull  pred_bear
true_none          4839         91         86
true_bullish         53         43          0
true_bearish         43          0         43
```

### transformer

```
              pred_none  pred_bull  pred_bear
true_none          4826         78        112
true_bullish         65         31          0
true_bearish         56          0         30
```

### xgboost

```
              pred_none  pred_bull  pred_bear
true_none          4837         72        107
true_bullish         31         65          0
true_bearish         23          0         63
```

## Agreement Matrix

Fraction of windows where each pair of models predicts the same class.

| | cnn_lstm | lstm | transformer | xgboost |
|---|---|---|---|---|
| **cnn_lstm** | 1.000 | 0.940 | 0.922 | 0.941 |
| **lstm** | 0.940 | 1.000 | 0.935 | 0.941 |
| **transformer** | 0.922 | 0.935 | 1.000 | 0.928 |
| **xgboost** | 0.941 | 0.941 | 0.928 | 1.000 |

## Top Disagreement Windows

| Rank | Window | Timestamp | Disagreement | GT Label | cnn_lstm | lstm | transformer | xgboost |
|------|--------|-----------|-------------|----------|---|---|---|---|
| 1 | 2598 | 2024-07-09 13:30 | 0.500 | bullish | none | none | none | bullish |
| 2 | 4386 | 2025-07-17 09:30 | 0.500 | bullish | bullish | bullish | none | bullish |
| 3 | 4406 | 2025-07-21 15:30 | 0.500 | none | none | none | bearish | none |
| 4 | 2994 | 2024-09-27 10:30 | 0.500 | none | bullish | none | none | none |
| 5 | 2995 | 2024-09-27 11:30 | 0.500 | bullish | bullish | bullish | none | bullish |
| 6 | 4398 | 2025-07-18 14:30 | 0.500 | none | none | none | bearish | none |
| 7 | 4397 | 2025-07-18 13:30 | 0.500 | none | none | none | bearish | none |
| 8 | 4396 | 2025-07-18 12:30 | 0.500 | none | none | bearish | bearish | none |
| 9 | 4395 | 2025-07-18 11:30 | 0.500 | none | none | bearish | bearish | bearish |
| 10 | 2997 | 2024-09-27 13:30 | 0.500 | none | none | bearish | none | bearish |
| 11 | 2998 | 2024-09-27 14:30 | 0.500 | none | bearish | none | none | none |
| 12 | 4388 | 2025-07-17 11:30 | 0.500 | bullish | bullish | none | bullish | bullish |
| 13 | 3002 | 2024-09-30 11:30 | 0.500 | none | none | none | bullish | none |
| 14 | 4385 | 2025-07-16 15:30 | 0.500 | none | bullish | none | none | none |
| 15 | 4493 | 2025-08-07 11:30 | 0.500 | none | none | bearish | bearish | none |
| 16 | 4384 | 2025-07-16 14:30 | 0.500 | bullish | bullish | bullish | none | bullish |
| 17 | 1903 | 2024-02-14 11:30 | 0.500 | none | bullish | none | none | none |
| 18 | 1902 | 2024-02-14 10:30 | 0.500 | bullish | none | none | none | bullish |
| 19 | 3006 | 2024-09-30 15:30 | 0.500 | none | none | none | bearish | none |
| 20 | 699 | 2023-06-08 11:30 | 0.500 | bearish | none | none | none | bearish |

## Trade Outcomes (simulated)

Each positive prediction → bracket trade: entry at next H1 open, SL beyond opposite gap edge, TP = R×R-multiple. Win rate counts only decided trades.

| Model | Signals | Filled | TP | SL | Undecided | Win Rate | Total R | Avg R | after_cost_total_R | median_R | n_outlier_R |
|-------|---------|--------|----|----|-----------|---------|---------|------|--------------------|---------|------------|
| cnn_lstm | 318 | 318 | 83 | 187 | 48 | 0.307 | -6.53 | -0.021 | -14.83 | -1.000 | 0 |
| lstm | 263 | 263 | 67 | 155 | 41 | 0.302 | -15.29 | -0.058 | -22.70 | -1.000 | 0 |
| transformer | 251 | 251 | 67 | 132 | 52 | 0.337 | +6.90 | +0.027 | +0.03 | -1.000 | 0 |
| xgboost | 307 | 307 | 89 | 180 | 38 | 0.331 | +6.62 | +0.022 | -1.64 | -1.000 | 0 |

## Exit-Strategy Comparison

**Realistic mode — pre-registered primary metric: `after_cost_total_R`** (total R after ATR min-stop floor + round-trip costs).  `winsorized_total_R`, `median_R`, `n_outlier_R` are mandatory context.  Raw `total_R` shown for continuity with the pre-realism comparison.

> **Caveat:** limit entries modelled as filled at exact limit price on first bar touching that level (optimistic OHLC bar-level backtest assumption).  Overstates fill quality vs live execution.

### cnn_lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 318 | 318 | N/A | 0.307 | 83 | 187 | 48 | -6.53 | -0.021 | -14.83 | -0.047 | -1.000 | 0 | -6.53 | 0.000 |
| ict_iofed | 318 | 229 | 0.720 | 0.332 | 66 | 133 | 30 | +26.97 | +0.118 | +22.18 | +0.097 | -1.000 | 0 | +26.97 | 0.000 |
| ce_50pct | 318 | 193 | 0.607 | 0.276 | 47 | 123 | 23 | +50.92 | +0.264 | +45.89 | +0.238 | -1.000 | 0 | +50.92 | 0.000 |
| tradinglab | 318 | 229 | 0.720 | 0.250 | 53 | 159 | 17 | +40.09 | +0.175 | +32.24 | +0.141 | -1.000 | 0 | +40.09 | 0.000 |

### lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 263 | 263 | N/A | 0.302 | 67 | 155 | 41 | -15.29 | -0.058 | -22.70 | -0.086 | -1.000 | 0 | -15.29 | 0.000 |
| ict_iofed | 263 | 200 | 0.760 | 0.282 | 51 | 130 | 19 | -13.38 | -0.067 | -18.25 | -0.091 | -1.000 | 1 | -13.48 | 0.000 |
| ce_50pct | 263 | 180 | 0.684 | 0.247 | 41 | 125 | 14 | -3.95 | -0.022 | -8.90 | -0.049 | -1.000 | 0 | -3.95 | 0.000 |
| tradinglab | 263 | 200 | 0.760 | 0.220 | 41 | 145 | 14 | -0.41 | -0.002 | -7.52 | -0.038 | -1.000 | 1 | -0.51 | 0.000 |

### transformer

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 251 | 251 | N/A | 0.337 | 67 | 132 | 52 | +6.90 | +0.027 | +0.03 | +0.000 | -1.000 | 0 | +6.90 | 0.000 |
| ict_iofed | 251 | 176 | 0.701 | 0.292 | 45 | 109 | 22 | -22.88 | -0.130 | -27.97 | -0.159 | -1.000 | 1 | -22.98 | 0.000 |
| ce_50pct | 251 | 163 | 0.649 | 0.235 | 35 | 114 | 14 | -14.68 | -0.090 | -19.98 | -0.123 | -1.000 | 1 | -14.81 | 0.000 |
| tradinglab | 251 | 176 | 0.701 | 0.255 | 41 | 120 | 15 | -3.00 | -0.017 | -9.07 | -0.052 | -1.000 | 2 | -3.11 | 0.000 |

### xgboost

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 307 | 307 | N/A | 0.331 | 89 | 180 | 38 | +6.62 | +0.022 | -1.64 | -0.005 | -1.000 | 0 | +6.62 | 0.000 |
| ict_iofed | 307 | 223 | 0.726 | 0.307 | 61 | 138 | 24 | +11.21 | +0.050 | +6.13 | +0.027 | -1.000 | 0 | +11.21 | 0.000 |
| ce_50pct | 307 | 188 | 0.612 | 0.238 | 40 | 128 | 20 | +24.67 | +0.131 | +19.40 | +0.103 | -1.000 | 0 | +24.67 | 0.000 |
| tradinglab | 307 | 223 | 0.726 | 0.226 | 47 | 161 | 15 | +8.29 | +0.037 | +0.58 | +0.003 | -1.000 | 2 | +7.91 | 0.000 |

## All-Seeds Aggregate

5-seed mean ± std across multisym checkpoints.  Pre-registered primary metric: `after_cost_total_R` (mean±std).  Per-seed rows follow each strategy block.

### cnn_lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -12.36 ± 8.42 | 0.313 ± 0.012 | 320.6 |
| ict_iofed | 5 | +11.74 ± 15.98 | 0.318 ± 0.016 | 240.2 |
| ce_50pct | 5 | +41.10 ± 15.85 | 0.266 ± 0.018 | 211.8 |
| tradinglab | 5 | +23.10 ± 18.77 | 0.244 ± 0.015 | 240.2 |

#### cnn_lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -10.90 | 0.315 | 318 |
| 17 | -24.61 | 0.297 | 274 |
| 42 | -3.30 | 0.328 | 314 |
| 123 | -16.41 | 0.308 | 350 |
| 2024 | -6.59 | 0.315 | 347 |

#### cnn_lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -6.05 | 0.301 | 240 |
| 17 | +12.39 | 0.326 | 202 |
| 42 | +31.26 | 0.341 | 239 |
| 123 | +23.21 | 0.314 | 265 |
| 2024 | -2.12 | 0.308 | 255 |

#### cnn_lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +25.96 | 0.257 | 212 |
| 17 | +45.08 | 0.283 | 184 |
| 42 | +60.74 | 0.284 | 209 |
| 123 | +49.87 | 0.266 | 231 |
| 2024 | +23.85 | 0.242 | 223 |

#### cnn_lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -2.80 | 0.224 | 240 |
| 17 | +22.56 | 0.241 | 202 |
| 42 | +41.23 | 0.267 | 239 |
| 123 | +40.95 | 0.245 | 265 |
| 2024 | +13.59 | 0.246 | 255 |

### lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -18.33 ± 8.75 | 0.304 ± 0.010 | 303.4 |
| ict_iofed | 5 | +4.22 ± 17.73 | 0.310 ± 0.016 | 228.8 |
| ce_50pct | 5 | +24.55 ± 19.34 | 0.255 ± 0.023 | 201.0 |
| tradinglab | 5 | +22.29 ± 22.36 | 0.255 ± 0.015 | 228.8 |

#### lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -16.57 | 0.305 | 324 |
| 17 | -13.09 | 0.312 | 312 |
| 42 | -7.54 | 0.315 | 290 |
| 123 | -27.79 | 0.294 | 303 |
| 2024 | -26.67 | 0.296 | 288 |

#### lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +5.84 | 0.315 | 247 |
| 17 | +30.86 | 0.335 | 236 |
| 42 | +6.73 | 0.299 | 219 |
| 123 | -16.90 | 0.293 | 227 |
| 2024 | -5.42 | 0.309 | 215 |

#### lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +30.09 | 0.256 | 217 |
| 17 | +53.38 | 0.291 | 207 |
| 42 | +21.64 | 0.240 | 191 |
| 123 | +0.70 | 0.231 | 200 |
| 2024 | +16.94 | 0.256 | 190 |

#### lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +7.00 | 0.255 | 247 |
| 17 | +52.82 | 0.276 | 236 |
| 42 | +36.49 | 0.244 | 219 |
| 123 | -2.41 | 0.239 | 227 |
| 2024 | +17.54 | 0.260 | 215 |

### transformer

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -6.88 ± 17.21 | 0.250 ± 0.144 | 208.2 |
| ict_iofed | 5 | -16.09 ± 15.06 | 0.243 ± 0.136 | 149.0 |
| ce_50pct | 5 | +4.53 ± 12.66 | 0.194 ± 0.110 | 128.6 |
| tradinglab | 5 | -10.61 ± 9.86 | 0.197 ± 0.111 | 149.0 |

#### transformer / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -25.55 | 0.278 | 217 |
| 17 | -25.10 | 0.280 | 224 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | +10.05 | 0.354 | 221 |
| 2024 | +6.19 | 0.339 | 379 |

#### transformer / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -24.79 | 0.301 | 148 |
| 17 | -31.58 | 0.288 | 168 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -24.29 | 0.308 | 156 |
| 2024 | +0.19 | 0.317 | 273 |

#### transformer / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -6.63 | 0.216 | 122 |
| 17 | -2.81 | 0.231 | 144 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | +6.66 | 0.262 | 135 |
| 2024 | +25.44 | 0.263 | 242 |

#### transformer / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -11.87 | 0.257 | 148 |
| 17 | -25.98 | 0.224 | 168 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -10.78 | 0.259 | 156 |
| 2024 | -4.40 | 0.247 | 273 |

### xgboost

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | +15.33 ± 2.73 | 0.351 ± 0.004 | 273.2 |
| ict_iofed | 5 | +15.72 ± 5.13 | 0.327 ± 0.005 | 200.8 |
| ce_50pct | 5 | +29.53 ± 4.01 | 0.261 ± 0.006 | 171.2 |
| tradinglab | 5 | +13.71 ± 5.99 | 0.247 ± 0.004 | 200.8 |

#### xgboost / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +16.53 | 0.354 | 271 |
| 17 | +14.44 | 0.349 | 270 |
| 42 | +15.37 | 0.351 | 276 |
| 123 | +11.45 | 0.346 | 270 |
| 2024 | +18.86 | 0.357 | 279 |

#### xgboost / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +18.91 | 0.328 | 200 |
| 17 | +16.05 | 0.328 | 199 |
| 42 | +20.05 | 0.333 | 203 |
| 123 | +16.55 | 0.328 | 198 |
| 2024 | +7.04 | 0.320 | 204 |

#### xgboost / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +31.17 | 0.260 | 171 |
| 17 | +32.20 | 0.265 | 170 |
| 42 | +33.27 | 0.268 | 173 |
| 123 | +27.55 | 0.260 | 168 |
| 2024 | +23.47 | 0.253 | 174 |

#### xgboost / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +19.24 | 0.250 | 200 |
| 17 | +13.22 | 0.245 | 199 |
| 42 | +17.80 | 0.251 | 203 |
| 123 | +14.37 | 0.245 | 198 |
| 2024 | +3.94 | 0.243 | 204 |
