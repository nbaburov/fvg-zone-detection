# Sensitivity Sweep — Cost / Fill / TP-RR

One-axis-at-a-time sensitivity around headline config: k=0.5, fill=optimistic, cost=base (1-tick+0.005), tp_rr=2.0, threshold=0.5.

**NOTE: softmax confidence_threshold is NOT calibrated.**


---

## Sensitivity — Cost Tiers  (fill=optimistic, tp_rr=2.0, threshold=0.5, k=0.5)

### cnn_lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 816 | 0.323 | +17.26 | +17.26 | -1.000 |
| cheap | ict_iofed | 589 | 0.329 | -28.77 | -28.77 | -1.000 |
| cheap | ce_50pct | 520 | 0.277 | +15.33 | +15.33 | -1.000 |
| cheap | tradinglab | 589 | 0.263 | -2.14 | -2.14 | -1.000 |
| base | fixed_2r | 816 | 0.323 | -11.12 | +17.26 | -1.000 |
| base | ict_iofed | 589 | 0.329 | -45.78 | -28.77 | -1.000 |
| base | ce_50pct | 520 | 0.277 | -2.69 | +15.33 | -1.000 |
| base | tradinglab | 589 | 0.263 | -27.74 | -2.14 | -1.000 |
| expensive | fixed_2r | 816 | 0.323 | -39.50 | +17.26 | -1.000 |
| expensive | ict_iofed | 589 | 0.329 | -62.80 | -28.77 | -1.000 |
| expensive | ce_50pct | 520 | 0.277 | -20.71 | +15.33 | -1.000 |
| expensive | tradinglab | 589 | 0.263 | -53.34 | -2.14 | -1.000 |

### lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 1034 | 0.336 | +67.62 | +67.62 | -1.000 |
| cheap | ict_iofed | 726 | 0.276 | -94.98 | -96.45 | -1.000 |
| cheap | ce_50pct | 664 | 0.235 | -54.60 | -56.12 | -1.000 |
| cheap | tradinglab | 726 | 0.220 | -79.87 | -81.35 | -1.000 |
| base | fixed_2r | 1034 | 0.336 | +28.49 | +67.62 | -1.000 |
| base | ict_iofed | 726 | 0.276 | -118.11 | -96.45 | -1.000 |
| base | ce_50pct | 664 | 0.235 | -78.76 | -56.12 | -1.000 |
| base | tradinglab | 726 | 0.220 | -112.92 | -81.35 | -1.000 |
| expensive | fixed_2r | 1034 | 0.336 | -10.64 | +67.62 | -1.000 |
| expensive | ict_iofed | 726 | 0.276 | -141.24 | -96.45 | -1.000 |
| expensive | ce_50pct | 664 | 0.235 | -102.92 | -56.12 | -1.000 |
| expensive | tradinglab | 726 | 0.220 | -145.97 | -81.35 | -1.000 |

### transformer

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 668 | 0.315 | +11.42 | +11.42 | -1.000 |
| cheap | ict_iofed | 450 | 0.310 | -28.14 | -28.14 | -1.000 |
| cheap | ce_50pct | 400 | 0.270 | +2.79 | +2.79 | -1.000 |
| cheap | tradinglab | 450 | 0.246 | -15.45 | -15.45 | -1.000 |
| base | fixed_2r | 668 | 0.315 | -10.57 | +11.42 | -1.000 |
| base | ict_iofed | 450 | 0.310 | -41.65 | -28.14 | -1.000 |
| base | ce_50pct | 400 | 0.270 | -11.53 | +2.79 | -1.000 |
| base | tradinglab | 450 | 0.246 | -35.31 | -15.45 | -1.000 |
| expensive | fixed_2r | 668 | 0.315 | -32.56 | +11.42 | -1.000 |
| expensive | ict_iofed | 450 | 0.310 | -55.17 | -28.14 | -1.000 |
| expensive | ce_50pct | 400 | 0.270 | -25.86 | +2.79 | -1.000 |
| expensive | tradinglab | 450 | 0.246 | -55.16 | -15.45 | -1.000 |

### xgboost

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 1182 | 0.342 | +91.80 | +91.80 | -1.000 |
| cheap | ict_iofed | 845 | 0.315 | -52.82 | -54.29 | -1.000 |
| cheap | ce_50pct | 753 | 0.266 | +1.18 | -0.35 | -1.000 |
| cheap | tradinglab | 845 | 0.242 | -64.31 | -65.79 | -1.000 |
| base | fixed_2r | 1182 | 0.342 | +48.84 | +91.80 | -1.000 |
| base | ict_iofed | 845 | 0.315 | -77.29 | -54.29 | -1.000 |
| base | ce_50pct | 753 | 0.266 | -24.94 | -0.35 | -1.000 |
| base | tradinglab | 845 | 0.242 | -101.97 | -65.79 | -1.000 |
| expensive | fixed_2r | 1182 | 0.342 | +5.89 | +91.80 | -1.000 |
| expensive | ict_iofed | 845 | 0.315 | -101.76 | -54.29 | -1.000 |
| expensive | ce_50pct | 753 | 0.266 | -51.05 | -0.35 | -1.000 |
| expensive | tradinglab | 845 | 0.242 | -139.64 | -65.79 | -1.000 |


---

## Sensitivity — TP Reward Multiple  (fill=optimistic, cost=base, threshold=0.5, k=0.5)

### cnn_lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 816 | 0.499 | -26.16 | +2.22 | +0.000 |
| tp_rr=1.0 | ict_iofed | 589 | 0.329 | -45.78 | -28.77 | -1.000 |
| tp_rr=1.0 | ce_50pct | 520 | 0.277 | -2.69 | +15.33 | -1.000 |
| tp_rr=1.0 | tradinglab | 589 | 0.263 | -27.74 | -2.14 | -1.000 |
| tp_rr=2.0 | fixed_2r | 816 | 0.323 | -11.12 | +17.26 | -1.000 |
| tp_rr=2.0 | ict_iofed | 589 | 0.329 | -45.78 | -28.77 | -1.000 |
| tp_rr=2.0 | ce_50pct | 520 | 0.277 | -2.69 | +15.33 | -1.000 |
| tp_rr=2.0 | tradinglab | 589 | 0.263 | -27.74 | -2.14 | -1.000 |
| tp_rr=3.0 | fixed_2r | 816 | 0.237 | +19.95 | +48.33 | -1.000 |
| tp_rr=3.0 | ict_iofed | 589 | 0.329 | -45.78 | -28.77 | -1.000 |
| tp_rr=3.0 | ce_50pct | 520 | 0.277 | -2.69 | +15.33 | -1.000 |
| tp_rr=3.0 | tradinglab | 589 | 0.263 | -27.74 | -2.14 | -1.000 |

### lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 1034 | 0.512 | -6.85 | +32.28 | +0.235 |
| tp_rr=1.0 | ict_iofed | 726 | 0.276 | -118.11 | -96.45 | -1.000 |
| tp_rr=1.0 | ce_50pct | 664 | 0.235 | -78.76 | -56.12 | -1.000 |
| tp_rr=1.0 | tradinglab | 726 | 0.220 | -112.92 | -81.35 | -1.000 |
| tp_rr=2.0 | fixed_2r | 1034 | 0.336 | +28.49 | +67.62 | -1.000 |
| tp_rr=2.0 | ict_iofed | 726 | 0.276 | -118.11 | -96.45 | -1.000 |
| tp_rr=2.0 | ce_50pct | 664 | 0.235 | -78.76 | -56.12 | -1.000 |
| tp_rr=2.0 | tradinglab | 726 | 0.220 | -112.92 | -81.35 | -1.000 |
| tp_rr=3.0 | fixed_2r | 1034 | 0.253 | +93.93 | +133.06 | -1.000 |
| tp_rr=3.0 | ict_iofed | 726 | 0.276 | -118.11 | -96.45 | -1.000 |
| tp_rr=3.0 | ce_50pct | 664 | 0.235 | -78.76 | -56.12 | -1.000 |
| tp_rr=3.0 | tradinglab | 726 | 0.220 | -112.92 | -81.35 | -1.000 |

### transformer

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 668 | 0.501 | -22.24 | -0.25 | +0.000 |
| tp_rr=1.0 | ict_iofed | 450 | 0.310 | -41.65 | -28.14 | -1.000 |
| tp_rr=1.0 | ce_50pct | 400 | 0.270 | -11.53 | +2.79 | -1.000 |
| tp_rr=1.0 | tradinglab | 450 | 0.246 | -35.31 | -15.45 | -1.000 |
| tp_rr=2.0 | fixed_2r | 668 | 0.315 | -10.57 | +11.42 | -1.000 |
| tp_rr=2.0 | ict_iofed | 450 | 0.310 | -41.65 | -28.14 | -1.000 |
| tp_rr=2.0 | ce_50pct | 400 | 0.270 | -11.53 | +2.79 | -1.000 |
| tp_rr=2.0 | tradinglab | 450 | 0.246 | -35.31 | -15.45 | -1.000 |
| tp_rr=3.0 | fixed_2r | 668 | 0.238 | +27.70 | +49.69 | -1.000 |
| tp_rr=3.0 | ict_iofed | 450 | 0.310 | -41.65 | -28.14 | -1.000 |
| tp_rr=3.0 | ce_50pct | 400 | 0.270 | -11.53 | +2.79 | -1.000 |
| tp_rr=3.0 | tradinglab | 450 | 0.246 | -35.31 | -15.45 | -1.000 |

### xgboost

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 1182 | 0.520 | +5.51 | +48.46 | +0.248 |
| tp_rr=1.0 | ict_iofed | 845 | 0.315 | -77.29 | -54.29 | -1.000 |
| tp_rr=1.0 | ce_50pct | 753 | 0.266 | -24.94 | -0.35 | -1.000 |
| tp_rr=1.0 | tradinglab | 845 | 0.242 | -101.97 | -65.79 | -1.000 |
| tp_rr=2.0 | fixed_2r | 1182 | 0.342 | +48.84 | +91.80 | -1.000 |
| tp_rr=2.0 | ict_iofed | 845 | 0.315 | -77.29 | -54.29 | -1.000 |
| tp_rr=2.0 | ce_50pct | 753 | 0.266 | -24.94 | -0.35 | -1.000 |
| tp_rr=2.0 | tradinglab | 845 | 0.242 | -101.97 | -65.79 | -1.000 |
| tp_rr=3.0 | fixed_2r | 1182 | 0.238 | +57.28 | +100.23 | -1.000 |
| tp_rr=3.0 | ict_iofed | 845 | 0.315 | -77.29 | -54.29 | -1.000 |
| tp_rr=3.0 | ce_50pct | 753 | 0.266 | -24.94 | -0.35 | -1.000 |
| tp_rr=3.0 | tradinglab | 845 | 0.242 | -101.97 | -65.79 | -1.000 |


---

## Sensitivity — Fill Mode  (cost=base, tp_rr=2.0, threshold=0.5, k=0.5)  Limit strategies only (V2/V3/V4); V1 fixed_2r is market entry — N/A.

### cnn_lstm

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 816 | 589 | 0.722 | 0.329 | -45.78 |
| optimistic | ce_50pct | 816 | 520 | 0.637 | 0.277 | -2.69 |
| optimistic | tradinglab | 816 | 589 | 0.722 | 0.263 | -27.74 |
| conservative | ict_iofed | 816 | 482 | 0.591 | 0.205 | -214.50 |
| conservative | ce_50pct | 816 | 433 | 0.531 | 0.147 | -205.28 |
| conservative | tradinglab | 816 | 482 | 0.591 | 0.146 | -250.48 |

### lstm

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 1034 | 726 | 0.702 | 0.276 | -118.11 |
| optimistic | ce_50pct | 1034 | 664 | 0.642 | 0.235 | -78.76 |
| optimistic | tradinglab | 1034 | 726 | 0.702 | 0.220 | -112.92 |
| conservative | ict_iofed | 1034 | 623 | 0.603 | 0.189 | -271.93 |
| conservative | ce_50pct | 1034 | 575 | 0.556 | 0.142 | -257.89 |
| conservative | tradinglab | 1034 | 623 | 0.603 | 0.137 | -311.15 |

### transformer

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 668 | 450 | 0.674 | 0.310 | -41.65 |
| optimistic | ce_50pct | 668 | 400 | 0.599 | 0.270 | -11.53 |
| optimistic | tradinglab | 668 | 450 | 0.674 | 0.246 | -35.31 |
| conservative | ict_iofed | 668 | 377 | 0.564 | 0.202 | -153.68 |
| conservative | ce_50pct | 668 | 339 | 0.507 | 0.151 | -146.06 |
| conservative | tradinglab | 668 | 377 | 0.564 | 0.142 | -175.32 |

### xgboost

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 1182 | 845 | 0.715 | 0.315 | -77.29 |
| optimistic | ce_50pct | 1182 | 753 | 0.637 | 0.266 | -24.94 |
| optimistic | tradinglab | 1182 | 845 | 0.715 | 0.242 | -101.97 |
| conservative | ict_iofed | 1182 | 712 | 0.602 | 0.216 | -259.54 |
| conservative | ce_50pct | 1182 | 644 | 0.545 | 0.166 | -251.04 |
| conservative | tradinglab | 1182 | 712 | 0.602 | 0.142 | -346.13 |
