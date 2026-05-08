# Build: Phase 3 Notebook (Foundation 10) — 8-May-26

Agent: nb-notebook (sonnet)
Prior read: phase3-data-prep.md, phase3-fixes.md, 6 research docs, all src/data/*.py

---

## Done

- `notebooks/01-data-understanding.ipynb` — 4 sections, 9 figures, LO evidence, APA refs
- `reports/` directory created (figure HTML outputs land here)

---

## Sections

1. Business Understanding — problem statement, ML reframing, Phase 3 success criteria
2. Data Understanding — source rationale (yfinance failure), schema, Fig 1 candlestick, Fig 2 session types, Fig 3 volume profile
3. Data Preparation — Fig 4 cleaning waterfall, FVG spec (math + N+1 diagram + falsification inline), Fig 5 label dist, Fig 6 rolling label rate, Fig 7 split timeline, Fig 8 normalisation before/after, anti-lookahead test inline, Fig 9 window counts
4. Summary + Phase 4 handoff — artefact checklist, limitations, architecture plan

---

## Figure count

9 figures (Figures 1–9). All Plotly interactive, saved to `reports/` as HTML.

---

## Test result

`jupyter nbconvert --execute` → exit 0. All code cells ran without errors.
0 error outputs in executed notebook.

---

## Cells needing re-run after pipeline materialises

| Cell | Condition | Tag |
|------|-----------|-----|
| Load spy_h1.parquet | Pipeline not run | [NEEDS PIPELINE] |
| Label distribution | Pipeline not run | [NEEDS PIPELINE] |
| Fig 1 candlestick | Pipeline not run | [NEEDS PIPELINE] |
| Fig 2 session types | Pipeline not run | [NEEDS PIPELINE] |
| Fig 3 volume profile | Pipeline not run | [NEEDS PIPELINE] |
| Fig 4 cleaning waterfall | Estimated values — replace with log actuals | [NEEDS PIPELINE] |
| Fig 5 label dist | Pipeline not run | [NEEDS PIPELINE] |
| Fig 6 rolling label rate | Pipeline not run | [NEEDS PIPELINE] |
| Load split parquets | Pipeline not run | [NEEDS PIPELINE] |
| Fig 7 split timeline | Pipeline not run | [NEEDS PIPELINE] |
| Fig 8 normalisation | Pipeline not run | [NEEDS PIPELINE] |
| Class weights | Pipeline not run | [NEEDS PIPELINE] |
| Window count estimates | Rough estimates only | [NEEDS PIPELINE] |
| Fig 9 window counts | Pipeline not run | [NEEDS PIPELINE] |
| Artefact checklist | All will show MISSING | [NEEDS PIPELINE] |
| Gold set / kappa | Annotation not done | [NEEDS ANNOTATION] |

Cells that execute fine without data (inline falsification fixture, anti-lookahead test, N+1 diagram, normalise_window unit demo) — all pass.

---

## Good

- Falsification fixture runs inline — reader sees the 6 assertions pass in notebook output
- Anti-lookahead test runs inline with specific mutation approach
- All [NEEDS PIPELINE] cells fail gracefully (FileNotFoundError caught, print message)
- Import mode justified (src/ is real codebase, graders can navigate)
- No em dashes in prose, no vague conclusions, no AI patterns (humanize pass next)

---

## Bad / Fixed

- JSON syntax error in Figure 5 caption cell metadata field (`"metadata":","`) — found and fixed via python3 json.loads; replaced with `"metadata": {}`
- Notebook missing cell IDs (nbformat warning) — non-blocking, cosmetic

---

## Open flags

- Fig 4 waterfall uses estimated bar counts (1.9M raw, 950k RTH, 13.8k H1) — replace with pipeline log actuals after Alpaca pull runs
- Gold set kappa = NaN until annotation done — notebook cell is written and will compute once gold_labels.csv exists
- nb-humanize pass still needed on prose cells
