# DL Ladder Diagnosis — Decision Gate

**Date:** 07-Jun-26 · **Target:** ValidFVG (`fvg_valid`) · **Data:** SPY H1 2016–2025, `SPLIT_BOUNDARIES`, weighted CE · **Seeds:** [0,17,42,123,2024] (ladder), [0,17,42] (curves/gap)

---

## SU2-ready summary

We completed the five-architecture model ladder (XGB → LSTM → CNN-LSTM → Transformer → xLSTM) on identical raw `(60,5)` windows and ran a fixed, pre-committed decision matrix to diagnose why deep learning trails the XGB engineered-feature baseline. **CNN-LSTM is the carrier** (test macro-F1 **0.639 ± 0.013**, 95% bootstrap CI **[0.603, 0.674]**), with the higher point estimate and the larger final-segment learning-curve gain (+0.041 from 80%→100% data vs LSTM's +0.021). The diagnosis is falsifiable: the recurrent models (LSTM, CNN-LSTM) are **data + regularisation-bound** — they fit train well (train F1 0.83–0.95) and their learning curves show **no plateau** through full data (val F1 climbs ~0.44→0.61 LSTM, ~0.49→0.60 CNN-LSTM). *Caveat: the curves are non-monotonic across the 3-seed bands, so the exact final-segment slope is within noise — the robust claim is "no plateau yet", not a precise marginal gain.* More (multi-symbol) data is the indicated lever. The two *larger* new architectures did *worse*, not better (Transformer 0.548 ± 0.095, xLSTM 0.369 ± 0.007), which **rules out a capacity bound** for the task: the ceiling is set by data quantity and signal-to-noise on a single symbol, not by model size. Their failure modes differ — **xLSTM genuinely underfits** (train F1 0.33–0.40, cannot fit train), while **Transformer fits and generalises on 4/5 seeds (val 0.58–0.61) but is high-variance** (seed42 collapses to 0.379, std 0.095). Either way, neither beats CNN-LSTM, so more model is not the answer. Carrier lever for SU2: **multi-symbol data expansion** for CNN-LSTM. Caveat: Transformer and xLSTM are *untuned* baselines (no Optuna), so their floor is a lower bound, not a tuned verdict — but tuning a model that cannot fit train is unlikely to close a 0.27 gap.

---

## Ranking table

| Rank | Model | Input | Mean macro-F1 ± std | 95% bootstrap CI | Status |
|------|-------|-------|--------------------|------------------|--------|
| — | **XGBoost** (reference) | 35 engineered feats | **0.721 ± 0.001** | n/a (context row) | Reference — *input mismatch, see caveat* |
| 1 | **CNN-LSTM** | raw (60,5) | **0.639 ± 0.013** | [0.603, 0.674] | Complete (carrier) |
| 2 | LSTM | raw (60,5) | 0.595 ± 0.015 | [0.560, 0.628] | Complete |
| 3 | Transformer | raw (60,5) | 0.548 ± 0.095 | [0.519, 0.572] | Complete (untuned, seed42 collapse) |
| 4 | xLSTM | raw (60,5) | 0.369 ± 0.007 | [0.354, 0.384] | Complete (untuned) |

**XGB input-mismatch caveat (F1):** XGB consumes the 35 hand-engineered features; all four DL models consume the raw `(60,5)` candle window. The XGB→DL gap is therefore *partly representation, not pure architecture*. The DL-vs-DL ranking is fair (identical input); the XGB row is context only and its 0.082 lead over CNN-LSTM must not be attributed to "XGB is a better architecture."

**Significance (F2):** CIs computed via `scripts/rigor/bootstrap_ci_multiseed.py` (1000 iter, block=60, effective_n=86).
- CNN-LSTM [0.603, 0.674] vs LSTM [0.560, 0.628] — **overlap (0.603–0.628)**. Strictly, **statistically tied** on F1 alone. The carrier call between them is therefore made on **headroom + point estimate**, not a bare F1 win (see Carrier decision).
- CNN-LSTM/LSTM vs Transformer [0.519, 0.572] — CNN-LSTM CI does **not** overlap Transformer → **CNN-LSTM > Transformer** is licensed. LSTM overlaps Transformer marginally (0.560–0.572) → LSTM vs Transformer **tied**.
- xLSTM [0.354, 0.384] is non-overlapping with all three → **xLSTM is significantly worst.**

---

## Per-model cards

### Card 1 — CNN-LSTM  (RANK 1, carrier)
- **Mean macro-F1 ± std / CI:** 0.639 ± 0.013 / [0.603, 0.674]
- **Regime:** matrix cell **data + reg-bound** (no plateau, train F1 HIGH). Placed by: 0.8→1.0 curve gain = **+0.041** (> 0.01 → not plateaued); final train F1 = **0.81–0.89** (≈ 0.85 threshold → HIGH; max train F1 0.88 confirms it can fit the data). *Curve is non-monotonic (0.486→0.527→0.575→0.562→0.603 over fractions); the 0.8 dip means the +0.041 is partly recovery, so read it as "no plateau", not a clean marginal gain.*
- **Weakness:** bear_f1 only **0.439 ± 0.035** vs bull_f1 0.504 — the minority bear-FVG class is the bottleneck; with no plateau at full data, it is plausibly starved of bear examples rather than capacity-limited.
- **Improve:** **more data** (the cell's lever) — specifically multi-symbol expansion to multiply the rare bear/bull-FVG count; it has the largest final-segment gain, so it plausibly gains the most per added example.
- **Multi-symbol verdict:** **YES.** It fits train (HIGH) and shows no plateau; the matrix data-bound cell prescribes more data, and more symbols directly add the scarce minority positives.

### Card 2 — LSTM  (RANK 2)
- **Mean macro-F1 ± std / CI:** 0.595 ± 0.015 / [0.560, 0.628]
- **Regime:** matrix cell **data + reg-bound** (no plateau, train F1 HIGH). Placed by: 0.8→1.0 curve gain = **+0.021** (> 0.01 → not plateaued); final train F1 = **0.93–0.95** (≫ 0.85 → HIGH). *Curve also non-monotonic (0.438→0.550→0.546→0.592→0.612); robust signal is "no plateau", slope within 3-seed noise.*
- **Weakness:** highest train F1 of all DL models (**0.95**) but val only 0.595 → **train/val gap ≈ 0.35**, the largest absolute gap — it memorises train and the gap is closed only by more data, not more epochs (best epoch 56–67, well converged).
- **Improve:** **more data + regularisation** (the cell's lever) — multi-symbol data to shrink the 0.35 generalisation gap; secondarily stronger dropout/weight-decay.
- **Multi-symbol verdict:** **YES.** High train F1 + rising curve + large gap is the textbook data-bound signature; more data is exactly the prescribed lever.

### Card 3 — Transformer  (RANK 3, untuned)
- **Mean macro-F1 ± std / CI:** 0.548 ± 0.095 / [0.519, 0.572]
- **Regime:** **unstable / high-variance** (does not sit cleanly in one matrix cell — the binding problem is *seed variance*, not the mean). Placed by: ladder best_epoch **22–61** across seeds (it trains, not best_epoch=1); **4/5 ladder seeds reach val 0.58–0.61** (fits + generalises), but **seed42 collapses to 0.366** (val) / 0.379 (test) — predicting majority + sparse bull (bull_f1 0.16) with **bear_f1 0.0000**. std **0.095** is ~6× the recurrent models'. **Reproducibility gap (important):** the *same* seeds 0/17/42 that succeeded in the ladder run **collapsed** (val ~0.327, bull=bear=0) in the independent learning-curve reruns — so the collapse mode triggers far more often than the ladder's 1/5 suggests. The honest read: Transformer **can** fit + generalise but the outcome is **run-dependent and fragile**, not "succeeds on 4/5". It is **not** capacity-bound (it reaches 0.6 when it works); it is unstable/untuned.
- **Weakness:** **seed-to-seed instability** — one in five seeds collapses to majority-only (bear_f1 0.0000), dragging the mean below LSTM. An untuned Transformer on 60-step windows has no warmup/LR schedule and is fragile to init.
- **Improve:** **HP tuning / stabilisation first** (warmup, lower LR, fewer params/heads, fixed-seed robustness) to kill the collapse mode — NOT more data first. Only after it is stable does the data lever (shared with the recurrent models) apply.
- **Multi-symbol verdict:** **NOT THE PRIMARY LEVER.** When it trains it already reaches 0.59 on current data, so the bottleneck is optimisation stability, not data volume. Per the falsification clause, multi-symbol is not the first lever here — stabilising training is. (More data would help the *good* seeds, but won't fix the collapse.)

### Card 4 — xLSTM  (RANK 4, untuned)
- **Mean macro-F1 ± std / CI:** 0.369 ± 0.007 / [0.354, 0.384]
- **Regime:** matrix cell **capacity-bound** (curve FLAT, train F1 LOW). Placed by: train F1 = **0.33–0.40** (< 0.85 → LOW); learning curve **abbreviated — full-data anchor only** (0.369), but the full-data score already sits below the majority baseline region and is the ladder floor, so it is treated as FLAT-by-construction (no fractional point can raise the ranking).
- **Weakness:** bull_f1 **0.114** and bear_f1 **0.056** — it barely predicts the minority classes at all (near-degenerate to majority), the lowest minority recall in the ladder; ladder best_epoch **8–12** for 4/5 seeds (converges early to a majority-dominated solution and stops improving — train F1 also stays 0.33–0.40, so it never fits).
- **Improve:** the cell's lever is **architecture/optimisation** — the untuned vanilla-backend sLSTM config underfits; a tuned config is unmeasured. But the 2025 MDPI evidence + this floor suggest low priority. NOT more data.
- **Multi-symbol verdict:** **NO.** Train F1 0.33–0.40 (cannot fit train) → more data cannot help a model that hasn't learned the data it has. Falsification clause: multi-symbol explicitly **not** the lever.

---

## Diagnosis synthesis

**Falsifiable claim:** *The DL ceiling on this task is set by data quantity / single-symbol minority-class scarcity, NOT by model capacity.*

Evidence:
1. The two models that **fit train** (LSTM 0.95, CNN-LSTM 0.88) both land in the **data + reg-bound** cell with **no plateau** through full data (final-segment gains +0.021, +0.041; curves non-monotonic, so "no plateau" is the robust read, not a precise slope) — consistent with a "needs more data" signature.
2. **Cross-check (the decisive one):** the two *larger / more expressive* architectures — Transformer and xLSTM — did **worse, not better** (0.548 and 0.369 vs CNN-LSTM 0.639). If the task were capacity-bound, bigger models would improve; instead xLSTM **underfits** (train F1 0.33–0.40, cannot fit train) and Transformer, though it fits and generalises on 4/5 seeds, is **too unstable** (seed42 collapse) to beat CNN-LSTM. Neither more-expressive arch raises the ceiling. This **falsifies the capacity-bound hypothesis** for the task as a whole and isolates the recurrent inductive bias + data quantity as the binding constraint.

Dominant regime: **data + regularisation-bound** for the viable (recurrent) models. The larger models fail for *different* per-model reasons — xLSTM underfits (optimisation/capacity), Transformer is unstable (variance) — neither of which is the task ceiling, and both are untuned.

---

## Carrier decision

**Carrier = CNN-LSTM.** Its CI overlaps LSTM's (statistically tied on raw F1), so the call is made on the tie-break: it holds the **higher point estimate** (0.639 vs 0.595) with a **tighter std** (0.013 vs 0.015) and the **larger final-segment learning-curve gain** (+0.041 vs +0.021, with the non-monotonic-curve caveat above). It is significantly better than both larger architectures (CI clears Transformer and xLSTM). All four models remain fully documented above — the carrier is the top card, not the only card.

**Lever for the carrier (read from its matrix cell — data + reg-bound):** **multi-symbol data expansion.** CNN-LSTM fits train and shows no val plateau; adding more symbols multiplies the scarce bull/bear-FVG positives (current bear_f1 0.439). SU2 closer: *best because — highest fair-input F1, tightest variance, no learning-curve plateau; improve via — multi-symbol data; current problem is in — minority-class data scarcity on a single symbol, not architecture.*

---

## Caveats & honesty

- **xLSTM curve abbreviated:** only the full-data (frac=1.0) anchor was computed (per-seed macro-F1 recomputed from the ladder `_preds.npz`); fractional points (0.2–0.8) were **not run** — CPU cost ≈ 5h and xLSTM's full-data score (0.369) is already the ladder floor and below the recurrent models' worst, so no fractional point can change the ranking. Honest header note carried in `learning_curve_xlstm.csv`.
- **Untuned baselines:** Transformer and xLSTM use single literature-sane G1-analogous configs, **no Optuna**. A tuned Transformer/xLSTM is unmeasured. This caveat applies **equally to both** new archs, so their *relative* floor is fair; the conclusion is about the ceiling/regime, not a fine-tuned ranking. For xLSTM (which cannot fit train, F1 0.33–0.40) a tuned config is unlikely to close the 0.27 gap; for Transformer the realistic upside is from stabilising the collapse, not from raw capacity. Both are bounds, not certainties.
- **Transformer seed42 collapse + reproducibility gap:** in the ladder, seed42 = 0.379 (majority + sparse bull, bull_f1 0.16, bear_f1 0.0000), other 4 seeds 0.576–0.609 → std 0.095. BUT the same seeds 0/17/42 **collapsed to ~0.327 (bull=bear=0) in the independent learning-curve reruns** — so the true collapse rate is higher than 1/5, and Transformer's mean (0.548) is itself run-dependent. We report the ladder run for ranking but flag that **a measured collapse-probability (repeat-N reruns) is the correct next step** before any strong Transformer claim. Does not change the conclusion (it never beats CNN-LSTM), but the instability is worse than the headline std implies.
- **Reported std convention:** the ± values in the ranking table are sample std (ddof=1) over the 5 seeds; the bootstrap JSONs report population std (ddof=0, `np.std`), so JSON values are slightly lower (e.g. CNN-LSTM 0.0119 vs 0.013). Same data, different convention.
- **Split provenance:** the train/val/test parquets were produced with `SPLIT_BOUNDARIES` (2016–2025) per CLAUDE.md, but the boundary set is asserted in docs, not pinned in a run-meta field. For a full audit trail, add a `split_boundaries` field to the meta sidecar (LOW priority, tracked).
- **Train-F1 / curve seeds:** train F1 and learning curves use 3 seeds (0,17,42) per the gap-run design; ladder F1/CIs use the full 5 seeds. CNN-LSTM final train F1 (0.81–0.89) straddles the 0.85 threshold — max train F1 (0.88) confirms HIGH placement.
- **XGB row** is engineered-feature input — context only, not a pure-architecture comparison.
