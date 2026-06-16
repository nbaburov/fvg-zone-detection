# Sensitivity Sweep — Cost / Fill / TP-RR

One-axis-at-a-time sensitivity around headline config: k=0.5, fill=optimistic, cost=base (1-tick+0.005), tp_rr=2.0, threshold=0.5.

**NOTE: softmax confidence_threshold is NOT calibrated.**


---

## Sensitivity — Cost Tiers  (fill=optimistic, tp_rr=2.0, threshold=0.5, k=0.5)

### cnn_lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 2319 | 0.329 | +116.52 | +116.52 | -1.000 |
| cheap | ict_iofed | 1600 | 0.323 | -113.10 | -116.16 | -1.000 |
| cheap | ce_50pct | 1436 | 0.259 | -123.34 | -124.92 | -1.000 |
| cheap | tradinglab | 1600 | 0.254 | -94.40 | -99.20 | -1.000 |
| base | fixed_2r | 2319 | 0.329 | -38.38 | +116.52 | -1.000 |
| base | ict_iofed | 1600 | 0.323 | -201.05 | -116.16 | -1.000 |
| base | ce_50pct | 1436 | 0.259 | -214.28 | -124.92 | -1.000 |
| base | tradinglab | 1600 | 0.254 | -223.77 | -99.20 | -1.000 |
| expensive | fixed_2r | 2319 | 0.329 | -193.27 | +116.52 | -1.000 |
| expensive | ict_iofed | 1600 | 0.323 | -289.01 | -116.16 | -1.000 |
| expensive | ce_50pct | 1436 | 0.259 | -305.23 | -124.92 | -1.000 |
| expensive | tradinglab | 1600 | 0.254 | -353.14 | -99.20 | -1.000 |

### lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 2858 | 0.319 | +48.14 | +48.14 | -1.000 |
| cheap | ict_iofed | 2005 | 0.301 | -155.76 | -157.05 | -1.000 |
| cheap | ce_50pct | 1861 | 0.254 | -159.23 | -159.94 | -1.000 |
| cheap | tradinglab | 2005 | 0.235 | -163.47 | -166.93 | -1.000 |
| base | fixed_2r | 2858 | 0.319 | -159.24 | +48.14 | -1.000 |
| base | ict_iofed | 2005 | 0.301 | -274.56 | -157.05 | -1.000 |
| base | ce_50pct | 1861 | 0.254 | -281.64 | -159.94 | -1.000 |
| base | tradinglab | 2005 | 0.235 | -332.55 | -166.93 | -1.000 |
| expensive | fixed_2r | 2858 | 0.319 | -366.63 | +48.14 | -1.000 |
| expensive | ict_iofed | 2005 | 0.301 | -393.36 | -157.05 | -1.000 |
| expensive | ce_50pct | 1861 | 0.254 | -404.04 | -159.94 | -1.000 |
| expensive | tradinglab | 2005 | 0.235 | -501.64 | -166.93 | -1.000 |

### transformer

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 1855 | 0.333 | +125.61 | +125.61 | -1.000 |
| cheap | ict_iofed | 1264 | 0.324 | -89.74 | -89.74 | -1.000 |
| cheap | ce_50pct | 1127 | 0.253 | -98.59 | -98.59 | -1.000 |
| cheap | tradinglab | 1264 | 0.251 | -56.55 | -58.29 | -1.000 |
| base | fixed_2r | 1855 | 0.333 | +5.78 | +125.61 | -1.000 |
| base | ict_iofed | 1264 | 0.324 | -156.33 | -89.74 | -1.000 |
| base | ce_50pct | 1127 | 0.253 | -167.91 | -98.59 | -1.000 |
| base | tradinglab | 1264 | 0.251 | -155.91 | -58.29 | -1.000 |
| expensive | fixed_2r | 1855 | 0.333 | -114.05 | +125.61 | -1.000 |
| expensive | ict_iofed | 1264 | 0.324 | -222.92 | -89.74 | -1.000 |
| expensive | ce_50pct | 1127 | 0.253 | -237.23 | -98.59 | -1.000 |
| expensive | tradinglab | 1264 | 0.251 | -255.27 | -58.29 | -1.000 |

### xgboost

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| cheap | fixed_2r | 4582 | 0.322 | +94.50 | +94.50 | -1.000 |
| cheap | ict_iofed | 3329 | 0.361 | -128.27 | -128.27 | -1.000 |
| cheap | ce_50pct | 3026 | 0.300 | -108.11 | -108.11 | -1.000 |
| cheap | tradinglab | 3329 | 0.284 | +4.23 | +2.38 | -1.000 |
| base | fixed_2r | 4582 | 0.322 | -226.56 | +94.50 | -1.000 |
| base | ict_iofed | 3329 | 0.361 | -309.14 | -128.27 | -1.000 |
| base | ce_50pct | 3026 | 0.300 | -299.48 | -108.11 | -1.000 |
| base | tradinglab | 3329 | 0.284 | -271.94 | +2.38 | -1.000 |
| expensive | fixed_2r | 4582 | 0.322 | -547.61 | +94.50 | -1.000 |
| expensive | ict_iofed | 3329 | 0.361 | -490.00 | -128.27 | -1.000 |
| expensive | ce_50pct | 3026 | 0.300 | -490.84 | -108.11 | -1.000 |
| expensive | tradinglab | 3329 | 0.284 | -548.11 | +2.38 | -1.000 |


---

## Sensitivity — TP Reward Multiple  (fill=optimistic, cost=base, threshold=0.5, k=0.5)

### cnn_lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 2319 | 0.519 | -72.29 | +82.61 | +0.294 |
| tp_rr=1.0 | ict_iofed | 1600 | 0.323 | -201.05 | -116.16 | -1.000 |
| tp_rr=1.0 | ce_50pct | 1436 | 0.259 | -214.28 | -124.92 | -1.000 |
| tp_rr=1.0 | tradinglab | 1600 | 0.254 | -223.77 | -99.20 | -1.000 |
| tp_rr=2.0 | fixed_2r | 2319 | 0.329 | -38.38 | +116.52 | -1.000 |
| tp_rr=2.0 | ict_iofed | 1600 | 0.323 | -201.05 | -116.16 | -1.000 |
| tp_rr=2.0 | ce_50pct | 1436 | 0.259 | -214.28 | -124.92 | -1.000 |
| tp_rr=2.0 | tradinglab | 1600 | 0.254 | -223.77 | -99.20 | -1.000 |
| tp_rr=3.0 | fixed_2r | 2319 | 0.222 | +13.81 | +168.71 | -1.000 |
| tp_rr=3.0 | ict_iofed | 1600 | 0.323 | -201.05 | -116.16 | -1.000 |
| tp_rr=3.0 | ce_50pct | 1436 | 0.259 | -214.28 | -124.92 | -1.000 |
| tp_rr=3.0 | tradinglab | 1600 | 0.254 | -223.77 | -99.20 | -1.000 |

### lstm

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 2858 | 0.511 | -147.26 | +60.12 | +0.033 |
| tp_rr=1.0 | ict_iofed | 2005 | 0.301 | -274.56 | -157.05 | -1.000 |
| tp_rr=1.0 | ce_50pct | 1861 | 0.254 | -281.64 | -159.94 | -1.000 |
| tp_rr=1.0 | tradinglab | 2005 | 0.235 | -332.55 | -166.93 | -1.000 |
| tp_rr=2.0 | fixed_2r | 2858 | 0.319 | -159.24 | +48.14 | -1.000 |
| tp_rr=2.0 | ict_iofed | 2005 | 0.301 | -274.56 | -157.05 | -1.000 |
| tp_rr=2.0 | ce_50pct | 1861 | 0.254 | -281.64 | -159.94 | -1.000 |
| tp_rr=2.0 | tradinglab | 2005 | 0.235 | -332.55 | -166.93 | -1.000 |
| tp_rr=3.0 | fixed_2r | 2858 | 0.215 | -116.75 | +90.63 | -1.000 |
| tp_rr=3.0 | ict_iofed | 2005 | 0.301 | -274.56 | -157.05 | -1.000 |
| tp_rr=3.0 | ce_50pct | 1861 | 0.254 | -281.64 | -159.94 | -1.000 |
| tp_rr=3.0 | tradinglab | 2005 | 0.235 | -332.55 | -166.93 | -1.000 |

### transformer

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 1855 | 0.524 | -34.66 | +85.17 | +0.280 |
| tp_rr=1.0 | ict_iofed | 1264 | 0.324 | -156.33 | -89.74 | -1.000 |
| tp_rr=1.0 | ce_50pct | 1127 | 0.253 | -167.91 | -98.59 | -1.000 |
| tp_rr=1.0 | tradinglab | 1264 | 0.251 | -155.91 | -58.29 | -1.000 |
| tp_rr=2.0 | fixed_2r | 1855 | 0.333 | +5.78 | +125.61 | -1.000 |
| tp_rr=2.0 | ict_iofed | 1264 | 0.324 | -156.33 | -89.74 | -1.000 |
| tp_rr=2.0 | ce_50pct | 1127 | 0.253 | -167.91 | -98.59 | -1.000 |
| tp_rr=2.0 | tradinglab | 1264 | 0.251 | -155.91 | -58.29 | -1.000 |
| tp_rr=3.0 | fixed_2r | 1855 | 0.212 | +7.37 | +127.20 | -1.000 |
| tp_rr=3.0 | ict_iofed | 1264 | 0.324 | -156.33 | -89.74 | -1.000 |
| tp_rr=3.0 | ce_50pct | 1127 | 0.253 | -167.91 | -98.59 | -1.000 |
| tp_rr=3.0 | tradinglab | 1264 | 0.251 | -155.91 | -58.29 | -1.000 |

### xgboost

| config | strategy | n_trades | win_rate | after_cost_total_R | winsorized_total_R | median_R |
|--------|----------|----------|----------|--------------------|--------------------|---------|
| tp_rr=1.0 | fixed_2r | 4582 | 0.504 | -290.66 | +30.39 | +0.000 |
| tp_rr=1.0 | ict_iofed | 3329 | 0.361 | -309.14 | -128.27 | -1.000 |
| tp_rr=1.0 | ce_50pct | 3026 | 0.300 | -299.48 | -108.11 | -1.000 |
| tp_rr=1.0 | tradinglab | 3329 | 0.284 | -271.94 | +2.38 | -1.000 |
| tp_rr=2.0 | fixed_2r | 4582 | 0.322 | -226.56 | +94.50 | -1.000 |
| tp_rr=2.0 | ict_iofed | 3329 | 0.361 | -309.14 | -128.27 | -1.000 |
| tp_rr=2.0 | ce_50pct | 3026 | 0.300 | -299.48 | -108.11 | -1.000 |
| tp_rr=2.0 | tradinglab | 3329 | 0.284 | -271.94 | +2.38 | -1.000 |
| tp_rr=3.0 | fixed_2r | 4582 | 0.221 | -126.67 | +194.38 | -1.000 |
| tp_rr=3.0 | ict_iofed | 3329 | 0.361 | -309.14 | -128.27 | -1.000 |
| tp_rr=3.0 | ce_50pct | 3026 | 0.300 | -299.48 | -108.11 | -1.000 |
| tp_rr=3.0 | tradinglab | 3329 | 0.284 | -271.94 | +2.38 | -1.000 |


---

## Sensitivity — Fill Mode  (cost=base, tp_rr=2.0, threshold=0.5, k=0.5)  Limit strategies only (V2/V3/V4); V1 fixed_2r is market entry — N/A.

### cnn_lstm

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 2319 | 1600 | 0.690 | 0.323 | -201.05 |
| optimistic | ce_50pct | 2319 | 1436 | 0.619 | 0.259 | -214.28 |
| optimistic | tradinglab | 2319 | 1600 | 0.690 | 0.254 | -223.77 |
| conservative | ict_iofed | 2319 | 1320 | 0.569 | 0.219 | -528.90 |
| conservative | ce_50pct | 2319 | 1179 | 0.508 | 0.157 | -510.93 |
| conservative | tradinglab | 2319 | 1320 | 0.569 | 0.148 | -709.93 |

### lstm

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 2858 | 2005 | 0.702 | 0.301 | -274.56 |
| optimistic | ce_50pct | 2858 | 1861 | 0.651 | 0.254 | -281.64 |
| optimistic | tradinglab | 2858 | 2005 | 0.702 | 0.235 | -332.55 |
| conservative | ict_iofed | 2858 | 1671 | 0.585 | 0.192 | -754.87 |
| conservative | ce_50pct | 2858 | 1548 | 0.542 | 0.147 | -740.66 |
| conservative | tradinglab | 2858 | 1671 | 0.585 | 0.124 | -1016.37 |

### transformer

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 1855 | 1264 | 0.681 | 0.324 | -156.33 |
| optimistic | ce_50pct | 1855 | 1127 | 0.608 | 0.253 | -167.91 |
| optimistic | tradinglab | 1855 | 1264 | 0.681 | 0.251 | -155.91 |
| conservative | ict_iofed | 1855 | 1039 | 0.560 | 0.210 | -450.26 |
| conservative | ce_50pct | 1855 | 923 | 0.498 | 0.145 | -455.47 |
| conservative | tradinglab | 1855 | 1039 | 0.560 | 0.136 | -596.23 |

### xgboost

| fill_mode | strategy | n_signals | n_trades | fill_rate | win_rate | after_cost_total_R |
|-----------|----------|-----------|----------|----------|----------|-------------------|
| optimistic | ict_iofed | 4582 | 3329 | 0.727 | 0.361 | -309.14 |
| optimistic | ce_50pct | 4582 | 3026 | 0.660 | 0.300 | -299.48 |
| optimistic | tradinglab | 4582 | 3329 | 0.727 | 0.284 | -271.94 |
| conservative | ict_iofed | 4582 | 2778 | 0.606 | 0.245 | -1071.70 |
| conservative | ce_50pct | 4582 | 2491 | 0.544 | 0.180 | -1089.95 |
| conservative | tradinglab | 4582 | 2778 | 0.606 | 0.163 | -1423.78 |
