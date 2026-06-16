# Model Inspection Summary

- Windows: 5194
- FVG-positive windows: 173 (3.3%)
- Models: cnn_lstm, lstm, transformer, xgboost

## Per-model Metrics

| Model | F1 Bull | F1 Bear | F1 FVG Macro | F1 Binary FVG |
|-------|---------|---------|--------------|---------------|
| cnn_lstm | 0.4091 | 0.4143 | 0.4117 | 0.4107 |
| lstm | 0.3905 | 0.3738 | 0.3822 | 0.3821 |
| transformer | 0.3264 | 0.3223 | 0.3243 | 0.3243 |
| xgboost | 0.5587 | 0.5241 | 0.5414 | 0.5438 |

## Precision / Recall (FVG classes)

| Model | Prec Bull | Rec Bull | Prec Bear | Rec Bear |
|-------|-----------|----------|-----------|----------|
| cnn_lstm | 0.2838 | 0.7326 | 0.5472 | 0.3333 |
| lstm | 0.3306 | 0.4767 | 0.3150 | 0.4598 |
| transformer | 0.2549 | 0.4535 | 0.2516 | 0.4483 |
| xgboost | 0.4286 | 0.8023 | 0.4900 | 0.5632 |

## Confusion Matrices

### cnn_lstm

```
              pred_none  pred_bull  pred_bear
true_none          4838        159         24
true_bullish         23         63          0
true_bearish         58          0         29
```

### lstm

```
              pred_none  pred_bull  pred_bear
true_none          4851         83         87
true_bullish         45         41          0
true_bearish         47          0         40
```

### transformer

```
              pred_none  pred_bull  pred_bear
true_none          4791        114        116
true_bullish         47         39          0
true_bearish         48          0         39
```

### xgboost

```
              pred_none  pred_bull  pred_bear
true_none          4878         92         51
true_bullish         17         69          0
true_bearish         38          0         49
```

## Agreement Matrix

Fraction of windows where each pair of models predicts the same class.

| | cnn_lstm | lstm | transformer | xgboost |
|---|---|---|---|---|
| **cnn_lstm** | 1.000 | 0.943 | 0.925 | 0.955 |
| **lstm** | 0.943 | 1.000 | 0.934 | 0.944 |
| **transformer** | 0.925 | 0.934 | 1.000 | 0.929 |
| **xgboost** | 0.955 | 0.944 | 0.929 | 1.000 |

## Top Disagreement Windows

| Rank | Window | Timestamp | Disagreement | GT Label | cnn_lstm | lstm | transformer | xgboost |
|------|--------|-----------|-------------|----------|---|---|---|---|
| 1 | 3820 | 2025-03-20 14:30 | 0.500 | bearish | none | bearish | none | bearish |
| 2 | 3657 | 2025-02-14 12:30 | 0.500 | none | none | bearish | none | none |
| 3 | 3969 | 2025-04-22 09:30 | 0.500 | none | none | bullish | none | none |
| 4 | 1105 | 2023-08-31 12:30 | 0.500 | bearish | none | none | none | bearish |
| 5 | 3959 | 2025-04-17 13:30 | 0.500 | none | none | bullish | none | none |
| 6 | 1106 | 2023-08-31 13:30 | 0.500 | bearish | bearish | bearish | none | bearish |
| 7 | 1107 | 2023-08-31 14:30 | 0.500 | none | none | bearish | none | none |
| 8 | 3949 | 2025-04-16 10:30 | 0.500 | none | none | none | none | bearish |
| 9 | 3946 | 2025-04-15 14:30 | 0.500 | bearish | none | none | none | bearish |
| 10 | 1118 | 2023-09-05 11:30 | 0.500 | none | none | none | bearish | none |
| 11 | 1121 | 2023-09-05 14:30 | 0.500 | bearish | none | bearish | bearish | none |
| 12 | 1124 | 2023-09-06 10:30 | 0.500 | bearish | none | bearish | none | none |
| 13 | 1125 | 2023-09-06 11:30 | 0.500 | bearish | none | bearish | none | none |
| 14 | 1129 | 2023-09-06 15:30 | 0.500 | bullish | none | none | none | bullish |
| 15 | 1134 | 2023-09-07 13:30 | 0.500 | none | bullish | bullish | none | none |
| 16 | 3926 | 2025-04-10 15:30 | 0.500 | none | none | none | bullish | none |
| 17 | 3925 | 2025-04-10 14:30 | 0.500 | none | none | none | bullish | none |
| 18 | 3923 | 2025-04-10 12:30 | 0.500 | bearish | none | none | none | bearish |
| 19 | 3922 | 2025-04-10 11:30 | 0.500 | bearish | none | bearish | bearish | bearish |
| 20 | 3919 | 2025-04-09 15:30 | 0.500 | none | none | none | bullish | none |

## Trade Outcomes (simulated)

Each positive prediction → bracket trade: entry at next H1 open, SL beyond opposite gap edge, TP = R×R-multiple. Win rate counts only decided trades.

| Model | Signals | Filled | TP | SL | Undecided | Win Rate | Total R | Avg R | after_cost_total_R | median_R | n_outlier_R |
|-------|---------|--------|----|----|-----------|---------|---------|------|--------------------|---------|------------|
| cnn_lstm | 275 | 275 | 73 | 150 | 52 | 0.327 | +12.69 | +0.046 | +6.66 | -1.000 | 0 |
| lstm | 251 | 251 | 66 | 136 | 49 | 0.327 | +13.19 | +0.053 | +6.99 | -1.000 | 0 |
| transformer | 308 | 308 | 72 | 168 | 68 | 0.300 | -7.95 | -0.026 | -15.31 | -1.000 | 0 |
| xgboost | 261 | 261 | 63 | 161 | 37 | 0.281 | -21.76 | -0.083 | -27.83 | -1.000 | 0 |

## Exit-Strategy Comparison

**Realistic mode — pre-registered primary metric: `after_cost_total_R`** (total R after ATR min-stop floor + round-trip costs).  `winsorized_total_R`, `median_R`, `n_outlier_R` are mandatory context.  Raw `total_R` shown for continuity with the pre-realism comparison.

> **Caveat:** limit entries modelled as filled at exact limit price on first bar touching that level (optimistic OHLC bar-level backtest assumption).  Overstates fill quality vs live execution.

### cnn_lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 275 | 275 | N/A | 0.327 | 73 | 150 | 52 | +12.69 | +0.046 | +6.66 | +0.024 | -1.000 | 0 | +12.69 | 0.000 |
| ict_iofed | 275 | 200 | 0.727 | 0.339 | 61 | 119 | 20 | -6.40 | -0.032 | -9.73 | -0.049 | -1.000 | 0 | -6.40 | 0.000 |
| ce_50pct | 275 | 175 | 0.636 | 0.267 | 43 | 118 | 14 | +6.00 | +0.034 | +2.44 | +0.014 | -1.000 | 0 | +6.00 | 0.000 |
| tradinglab | 275 | 200 | 0.727 | 0.278 | 52 | 135 | 13 | +30.22 | +0.151 | +24.79 | +0.124 | -1.000 | 0 | +30.22 | 0.000 |

### lstm

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 251 | 251 | N/A | 0.327 | 66 | 136 | 49 | +13.19 | +0.053 | +6.99 | +0.028 | -1.000 | 0 | +13.19 | 0.000 |
| ict_iofed | 251 | 190 | 0.757 | 0.300 | 54 | 126 | 10 | -39.34 | -0.207 | -43.11 | -0.227 | -1.000 | 0 | -39.34 | 0.000 |
| ce_50pct | 251 | 173 | 0.689 | 0.242 | 40 | 125 | 8 | -23.44 | -0.136 | -27.41 | -0.158 | -1.000 | 0 | -23.44 | 0.000 |
| tradinglab | 251 | 190 | 0.757 | 0.258 | 46 | 132 | 12 | -3.15 | -0.017 | -8.75 | -0.046 | -1.000 | 0 | -3.15 | 0.000 |

### transformer

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 308 | 308 | N/A | 0.300 | 72 | 168 | 68 | -7.95 | -0.026 | -15.31 | -0.050 | -1.000 | 0 | -7.95 | 0.000 |
| ict_iofed | 308 | 219 | 0.711 | 0.337 | 67 | 132 | 20 | +1.63 | +0.007 | -3.62 | -0.017 | -1.000 | 0 | +1.63 | 0.000 |
| ce_50pct | 308 | 207 | 0.672 | 0.273 | 54 | 144 | 9 | -22.84 | -0.110 | -28.20 | -0.136 | -1.000 | 0 | -22.84 | 0.000 |
| tradinglab | 308 | 219 | 0.711 | 0.283 | 56 | 142 | 21 | +14.54 | +0.066 | +8.02 | +0.037 | -1.000 | 0 | +14.54 | 0.000 |

### xgboost

| Strategy | n_signals | n_trades | fill_rate | win_rate | n_tp | n_sl | n_undecided | total_R | avg_R | after_cost_total_R | after_cost_avg_R | median_R | n_outlier_R | winsorized_total_R | swing_fallback_rate |
|----------|-----------|----------|-----------|----------|------|------|-------------|---------|-------|--------------------|-----------------|----------|-------------|--------------------|---------------------|
| fixed_2r | 261 | 261 | N/A | 0.281 | 63 | 161 | 37 | -21.76 | -0.083 | -27.83 | -0.107 | -1.000 | 0 | -21.76 | 0.000 |
| ict_iofed | 261 | 201 | 0.770 | 0.313 | 56 | 123 | 22 | -15.30 | -0.076 | -18.78 | -0.093 | -1.000 | 0 | -15.30 | 0.000 |
| ce_50pct | 261 | 184 | 0.705 | 0.279 | 46 | 119 | 19 | +5.10 | +0.028 | +1.15 | +0.006 | -1.000 | 0 | +5.10 | 0.000 |
| tradinglab | 261 | 201 | 0.770 | 0.250 | 46 | 138 | 17 | +17.91 | +0.089 | +12.27 | +0.061 | -1.000 | 0 | +17.91 | 0.000 |

## All-Seeds Aggregate

5-seed mean ± std across multisym checkpoints.  Pre-registered primary metric: `after_cost_total_R` (mean±std).  Per-seed rows follow each strategy block.

### cnn_lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -19.51 ± 12.30 | 0.295 ± 0.017 | 286.0 |
| ict_iofed | 5 | -8.09 ± 18.49 | 0.307 ± 0.031 | 227.0 |
| ce_50pct | 5 | +6.26 ± 21.07 | 0.265 ± 0.031 | 208.8 |
| tradinglab | 5 | +36.29 ± 19.82 | 0.253 ± 0.025 | 227.0 |

#### cnn_lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -26.64 | 0.281 | 256 |
| 17 | -23.44 | 0.286 | 244 |
| 42 | -1.91 | 0.314 | 280 |
| 123 | -12.60 | 0.314 | 340 |
| 2024 | -32.96 | 0.281 | 310 |

#### cnn_lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -2.13 | 0.323 | 202 |
| 17 | -33.10 | 0.263 | 194 |
| 42 | +11.72 | 0.337 | 219 |
| 123 | -20.98 | 0.289 | 269 |
| 2024 | +4.07 | 0.325 | 251 |

#### cnn_lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +17.77 | 0.289 | 188 |
| 17 | -23.30 | 0.222 | 179 |
| 42 | +25.91 | 0.293 | 200 |
| 123 | -8.36 | 0.246 | 249 |
| 2024 | +19.28 | 0.278 | 228 |

#### cnn_lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +45.31 | 0.273 | 202 |
| 17 | +3.18 | 0.215 | 194 |
| 42 | +49.14 | 0.270 | 219 |
| 123 | +32.81 | 0.238 | 269 |
| 2024 | +50.99 | 0.269 | 251 |

### lstm

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -7.55 ± 11.17 | 0.316 ± 0.017 | 293.0 |
| ict_iofed | 5 | -22.05 ± 7.37 | 0.303 ± 0.017 | 227.6 |
| ce_50pct | 5 | -14.82 ± 8.88 | 0.260 ± 0.016 | 208.0 |
| tradinglab | 5 | +10.94 ± 3.49 | 0.250 ± 0.012 | 227.6 |

#### lstm / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -17.12 | 0.302 | 334 |
| 17 | -13.19 | 0.309 | 286 |
| 42 | -16.59 | 0.300 | 277 |
| 123 | +3.96 | 0.332 | 291 |
| 2024 | +5.17 | 0.336 | 277 |

#### lstm / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -33.72 | 0.279 | 259 |
| 17 | -21.79 | 0.308 | 219 |
| 42 | -14.01 | 0.315 | 216 |
| 123 | -22.67 | 0.294 | 228 |
| 2024 | -18.05 | 0.321 | 216 |

#### lstm / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -23.89 | 0.247 | 240 |
| 17 | -15.50 | 0.263 | 199 |
| 42 | -0.11 | 0.283 | 200 |
| 123 | -15.88 | 0.242 | 206 |
| 2024 | -18.72 | 0.264 | 195 |

#### lstm / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +11.03 | 0.234 | 259 |
| 17 | +8.35 | 0.251 | 219 |
| 42 | +16.75 | 0.255 | 216 |
| 123 | +8.08 | 0.245 | 228 |
| 2024 | +10.52 | 0.267 | 216 |

### transformer

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -21.53 ± 17.22 | 0.219 ± 0.123 | 185.6 |
| ict_iofed | 5 | -11.98 ± 13.09 | 0.253 ± 0.144 | 144.6 |
| ce_50pct | 5 | -5.78 ± 11.84 | 0.195 ± 0.114 | 127.4 |
| tradinglab | 5 | +6.29 ± 16.68 | 0.207 ± 0.117 | 144.6 |

#### transformer / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -30.05 | 0.268 | 216 |
| 17 | -11.45 | 0.288 | 187 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -21.20 | 0.270 | 182 |
| 2024 | -44.95 | 0.271 | 343 |

#### transformer / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -16.12 | 0.349 | 165 |
| 17 | -21.85 | 0.307 | 142 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -25.57 | 0.278 | 139 |
| 2024 | +3.62 | 0.331 | 277 |

#### transformer / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -3.39 | 0.280 | 144 |
| 17 | -13.22 | 0.215 | 121 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -21.42 | 0.205 | 123 |
| 2024 | +9.11 | 0.274 | 249 |

#### transformer / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -3.48 | 0.272 | 165 |
| 17 | +6.16 | 0.264 | 142 |
| 42 | +0.00 | 0.000 | 0 |
| 123 | -6.18 | 0.228 | 139 |
| 2024 | +34.96 | 0.270 | 277 |

### xgboost

| Strategy | Seeds | after_cost_total_R (mean±std) | win_rate (mean±std) | n_trades (mean) |
|----------|-------|--------------------------------|--------------------|-----------------|
| fixed_2r | 5 | -19.29 ± 6.92 | 0.297 ± 0.009 | 285.4 |
| ict_iofed | 5 | -16.77 ± 3.29 | 0.313 ± 0.007 | 226.6 |
| ce_50pct | 5 | +1.37 ± 4.60 | 0.278 ± 0.008 | 206.6 |
| tradinglab | 5 | +20.66 ± 4.31 | 0.255 ± 0.004 | 226.6 |

#### xgboost / fixed_2r — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -18.96 | 0.298 | 281 |
| 17 | -23.45 | 0.291 | 284 |
| 42 | -18.55 | 0.300 | 286 |
| 123 | -8.57 | 0.311 | 293 |
| 2024 | -26.94 | 0.287 | 283 |

#### xgboost / ict_iofed — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | -18.02 | 0.307 | 226 |
| 17 | -21.59 | 0.304 | 225 |
| 42 | -16.06 | 0.314 | 227 |
| 123 | -15.42 | 0.317 | 229 |
| 2024 | -12.75 | 0.320 | 226 |

#### xgboost / ce_50pct — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +0.11 | 0.274 | 206 |
| 17 | -5.45 | 0.266 | 204 |
| 42 | +2.55 | 0.280 | 207 |
| 123 | +2.45 | 0.281 | 208 |
| 2024 | +7.20 | 0.286 | 208 |

#### xgboost / tradinglab — per-seed

| Seed | after_cost_total_R | win_rate | n_trades |
|------|-------------------|----------|----------|
| 0 | +22.97 | 0.255 | 226 |
| 17 | +17.00 | 0.250 | 225 |
| 42 | +23.41 | 0.255 | 227 |
| 123 | +24.81 | 0.261 | 229 |
| 2024 | +15.10 | 0.252 | 226 |
