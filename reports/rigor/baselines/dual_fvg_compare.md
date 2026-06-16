# Dual-FVG Baseline Comparison

> **Note:** raw FVG and ValidFVG measure different problems (different label difficulty).
> Raw FVG higher F1 is expected — not a regression.

| Model | Label | Pos rate | Test Macro F1 | Bull F1 | Bear F1 |
|-------|-------|----------|---------------|---------|---------|
| LSTM seed42  | raw FVG  | 25.2%  | 0.8753  | 0.8473  | 0.8310  |
| LSTM seed42  | ValidFVG | 3.1%   | 0.5961  | 0.0000  | 0.0000  |
| XGB seed42   | raw FVG  | 25.2%  | 0.5890  | 0.5436  | 0.5031  |
| XGB seed42   | ValidFVG | 3.1%   | 0.5294  | 0.4329  | 0.2291  |
| Naive majority | raw FVG  | 25.2%  | 0.2852  | —  | —  |
| Naive majority | ValidFVG | 3.1%   | 0.3281  | —  | —  |

*Gate A5 passed. raw-FVG LSTM delta vs ValidFVG: +0.2792 (threshold: >0.4 AND raw F1 >0.92).*
