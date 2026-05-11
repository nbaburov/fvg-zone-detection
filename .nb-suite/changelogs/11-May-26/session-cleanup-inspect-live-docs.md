# Changelog — 2026-05-11 session

Branch: `master`

| Hash | Type | Subject |
|------|------|---------|
| `be0fd68` | chore | clean up deprecated gold labels and reports |
| `166ddf9` | docs | update notebook with validated FVG labelling context |
| `03bac5a` | feat | add model inspection toolkit with outcome simulation |
| `5fab708` | feat | add live paper-trading harness for Alpaca |
| `3d7a021` | fix | unify Alpaca secret env var as ALPACA_SECRET_KEY |
| `9e88039` | docs | add architecture, data model, models status; sync README and CLAUDE.md |
| `1f8c185` | test | add session sample fixtures for inspect and live |

**Summary:** removed stale CSV + HTML artefacts, updated nb01 to discuss validated FVG target, shipped `src/inspect/` (offline toolkit with TP/SL outcome simulation + per-model timelines) and `src/live/` (Alpaca paper-trading harness), fixed env-var inconsistency, added top-level README + three doc files, synced CLAUDE.md and .env.example, committed session fixtures.
