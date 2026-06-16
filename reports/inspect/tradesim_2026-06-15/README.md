# Trade-sim sweep — 15 Jun 2026

Provenance for the unified trade-simulation table in `docs/trading-simulation.md` §3.

**Sweep:** best model per arch (XGBoost, tuned CNN-LSTM, LSTM, Transformer) × 4 tickers
(SPY, QQQ, IWM, DIA) × {h1, 5m, 15m} × 4 exit strategies (`fixed_2r`, `ict_iofed`,
`ce_50pct`, `tradinglab`) × 5 seeds, realistic fills + costs.

## Layout

```
h1/           4 per-ticker H1 runs  (window ≈ 5.2k each)
5m/           4 per-ticker 5m runs  (window ≈ 58k each)
  pooled5m_smoke/   pooled-multisym 5m smoke pass (233k windows)
15m/          4 per-ticker 15m runs (window ≈ 19k each)
sensitivity/  one-axis-at-a-time cost / fill / TP-RR sweep around headline config
```

Each leaf is one `inspect_models.py --all-exit-strategies --realistic --all-seeds`
run (timestamp = run id). The tool did **not** stamp the test ticker into
`summary.md`, so the 4 per-tf runs are not individually re-labelled here — read the
per-ticker values from the `docs/trading-simulation.md` table, which is the canonical
source. Window count distinguishes the timeframe (above).

## Headline

Only **15m SPY `fixed_2r`** is positive after costs (XGB +58R, tuned CNN-LSTM +28R).
5m is detection-only (cost drag); edge is SPY-specific (QQQ/IWM/DIA negative).
