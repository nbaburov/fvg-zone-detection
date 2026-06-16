# SHAP Feature Importance — XGBoost

Full model test macro F1: 0.7217
Pruned model test macro F1: 0.7231
Delta: +0.0014

Features dropped (mean |SHAP| < 0.0001): ['vol_spike']

| Feature | Mean |SHAP| |
|---------|------------|
| ret_60 | 0.838471 |
| gap_norm_bull | 0.814258 |
| gap_bear | 0.700608 |
| pos_in_range | 0.556108 |
| gap_norm_bear | 0.517470 |
| macd_norm | 0.415476 |
| macd_signal | 0.402696 |
| gap_bull | 0.379776 |
| ret_1 | 0.370769 |
| mid_body_bull | 0.352655 |
| prior_trend_5 | 0.337177 |
| trend_slope | 0.307988 |
| mid_body_bear | 0.301577 |
| ret_5 | 0.219725 |
| vol_zscore_5 | 0.204561 |
| range_60 | 0.202180 |
| vol_ratio_14_28 | 0.189818 |
| ret_20 | 0.151279 |
| ret_10 | 0.150220 |
| rsi_14 | 0.144495 |
| vol_body_corr | 0.141680 |
| atr_28 | 0.139886 |
| high_low_range | 0.139253 |
| upper_wick | 0.132927 |
| mid_range_norm | 0.129767 |
| range_20 | 0.125813 |
| vol_zscore_20 | 0.104868 |
| body_ratio | 0.098998 |
| vol_trend | 0.082405 |
| atr_14 | 0.063239 |
| above_ma50 | 0.012594 |
| react_in_gap_bear | 0.010440 |
| above_ma20 | 0.005987 |
| react_in_gap_bull | 0.002354 |
| vol_spike | 0.000000 |