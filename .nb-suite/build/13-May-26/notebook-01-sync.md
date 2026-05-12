# Notebook 01 Sync — 2026-05-13

**Task:** Surgical update of `notebooks/01-data-understanding.ipynb` to reflect current project ground truth.

## Changes Made

### Data range (2018-2024 → 2016-2025)
- Cell 14: Business Understanding date range
- Cell 23: Figure 1 caption
- Cell 33: Pipeline step Alpaca pull dates
- Cell 42: Figure 5 title
- Cell 43: Figure 5 caption
- Cell 46: Gold set section
- Cell 53: Figure 7 title in code
- Cell 54: Figure 7 caption (test end date, train years, test period)
- Cell 58: Figure 8 price range period
- Cell 65: Summary (0.3% → 3.25%)
- Cell 67: Limitations (multiple references)
- Cell 18: Alpaca section "2018 start" → "2016 start" and yfinance comparison

### Positive rate (0.3% → 3.25%)
- Cell 8: Section 0.4 description + count (50 → 570, 14,000 → 17,591)
- Cell 11: Figure 0.3 caption
- Cell 12: Observation block
- Cell 14: Business Understanding
- Cell 41: Label distribution section
- Cell 43: Figure 5 caption (0.1–0.5% → 1–10% range)
- Cell 45: Figure 6 obs + spike threshold (0.5% → 5%)
- Cell 46: Gold set sampling description (14,000 → 17,591, 99.7% → 96.7%)
- Cell 65: Summary paragraph
- Cell 67: Limitations (40–70 → 570, 14,000 → 17,591)

### 99.7% accuracy → 96.7%
- Cell 12: Observation block
- Cell 43: Figure 5 caption

### Bar count (13,500/14,686 → 17,591)
- Cell 18: yfinance comparison "13,500 needed" → "17,591 needed"
- Cell 35: Pipeline description "13,800" → "17,600"
- Cell 41: Sparsity table "(14,686 candles)" → "(17,591 candles)"

### Deprecated file path
- Cell 10: `spy_h1_labeled.parquet` → `spy_h1.parquet`

### WARNING threshold
- Cell 42: `< 0.05%` → `< 0.5%`

### Boundary dict / SPLIT_BOUNDARIES
- Cell 16: `test_end: 2024-12-31` → `2025-12-31`
- Cell 53: inline boundary dict test end

### New cell added
- Cell 70: Infrastructure and Rigor Sprint Notes (Python 3.12 migration, CPU-only LSTM, XGB subprocess isolation, rigor sprint F1 results table)

## Validation
- `nbformat.read()` confirms JSON valid
- No execution performed (presentational/education mode notebook)
- No commits created
