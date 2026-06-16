# Contributing

Thanks for your interest in the SMC FVG detector. This guide covers local setup, the test gates, and the project conventions that keep results trustworthy and reproducible.

This is an individual academic project under a noncommercial license (see [`LICENSE`](LICENSE)). External contributions are welcome for noncommercial purposes; for anything commercial, contact the author first.

## Setup

Requires Python 3.12.

```bash
git clone <repo> && cd smc-data-challenge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill ALPACA_API_KEY + ALPACA_SECRET_KEY (free Alpaca paper account)
```

Build the dataset before training or testing model code:

```bash
python -c "from src.data.pipeline import build_pipeline; build_pipeline()"
```

## Tests (the gate)

Every change must keep the suite green.

```bash
make test                    # full suite. XGBoost tests run isolated (see below)
pytest tests/strategy/ -q    # trade-exit logic
pytest tests/inspect/ -q     # inspection toolkit
```

`make test` runs pytest twice on purpose. XGBoost unit tests are isolated from the main run because pytest collection imports every test module up front. This loads torch before the XGBoost tests execute, and an in-process XGBoost fit segfaults on macOS arm64 once torch is imported. `tests/models/test_xgboost_baseline.py` is `--ignore`d in `pytest.ini` and run in its own pytest invocation by the Makefile.

When you fix a bug, add a regression test in the matching `tests/` subdirectory
(the tree mirrors `src/`).

## Critical constraints (do not break these)

These are enforced by tests and are the reason the results hold up. A change that
violates one will be rejected.

- **Temporal split only.** Never shuffle the time series. Train 2016 to 2021, validate 2022, test 2023 to 2025. Boundaries live in `src/data/split.py`.
- **No lookahead.** Labels use only information available at the candle's time. The FVG label index is N+2 (the pattern closes on bar N+1, so it is knowable then, never at N). A mandatory pytest fixture asserts this across timeframes.
- **F1 on the minority class is the primary metric.** The label distribution is roughly 97 percent "none", so accuracy is meaningless. Report macro F1 on the bull and bear FVG classes.
- **Weighted cross-entropy is required** for the neural nets, using the
  inverse-frequency weights in `data/processed/class_weights_{scope}_{tf}.json`.

See [`docs/evaluation.md`](docs/evaluation.md) for the full rationale.

## Naming conventions

Keep on-disk artifacts on the flat, explicit scheme so the repo stays navigable.

- **Checkpoints:** `checkpoints/{arch}_{tf}_{dataset}[_tuned]/` with files directly inside. `arch` in {cnn_lstm, lstm, transformer, xgboost, xlstm}; `tf` in {h1, 5m, 15m}; `dataset` in {spy, multisym}. Torch files are `{arch}_seed{N}.pt`, XGBoost is `xgb_seed{N}.ubj`, each with a matching `.meta.json`.
- **Processed data:** `data/processed/{scope}_{tf}_{split}.parquet` and `class_weights_{scope}_{tf}.json`, where `scope` in {spy, qqq, iwm, dia, multisym}.
- **Reports:** `reports/` is runtime output and is gitignored by default. A curated, bloat-free record is tracked via an allowlist in `.gitignore` (summaries only: `.json` / `.md` / `.csv` / static `.png`; never `*.html`, `*.npz`, `*.parquet`, or `plots/`). To track a new canonical report dir, add one `!reports/.../<dir>/` line to the allowlist. `tests/rigor/test_reports_layout.py` enforces this.

When generating an inspection run, pass `--label <name>` so the output directory is
self-documenting (`<name>_<timestamp>`) instead of a bare timestamp.

## Branches and commits

- Branch off `master`. Do not commit directly to `master`.
- Use Conventional Commits: `type(scope): summary` (for example
  `feat(data): multi-symbol pipeline`, `docs: refresh model card`, `fix(strategy):
  ATR stop guard`). Types in use: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.
- Keep commits focused. One logical change per commit.

## Pull requests

1. Branch, implement, and keep `make test` green.
2. Update the relevant docs in `docs/` when behaviour or structure changes.
3. Open a PR against `master` with a clear description of what changed and why.
4. Note any change to the critical constraints above explicitly; reviewers check
   these first.

## Reporting bugs and suggesting features

Open an issue. For a bug, include the command you ran, the expected versus actual
behaviour, and the full error output. For a feature, describe the use case and how
it fits the project's scope (FVG detection and evaluation, not price prediction).

## Hardware notes

LSTM and CNN-LSTM training are CPU only because of an MPS gradient kernel bug on Apple Silicon; the experiment YAMLs default to `device: cpu`. XGBoost training uses subprocess workers to avoid an arm64 segfault after a torch import.
