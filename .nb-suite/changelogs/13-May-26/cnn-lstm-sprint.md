# Changelog — CNN-LSTM Sprint — 13-May-26

| SHA | Type | Subject |
|-----|------|---------|
| 1619f8c | feat(model) | add CNN-LSTM classifier + registry integration |
| 1b7b030 | feat(rigor) | CNN-LSTM full sprint on ValidFVG (G1+G2+G4/6/7/9/10) |

## Summary
CNN-LSTM canonical baseline: 0.614 ± 0.021 (5-seed mean). Beats LSTM 0.599 by +0.015 but trails XGB 0.721 by −0.107. Topology: 2× Conv1d(16, k=5) → 2-layer LSTM(32). Full rigor pass: threshold, asymmetry, window sweep, reg ablation, bootstrap CI. Notebook 01 synced (40 cell changes, 0 exec errors). 284/284 tests pass.

G9 flagged dropout=0.222 may be over-regularization (no_dropout +0.009). Squeeze experiment + nb-update + cleanup queued in follow-up sprint.
