#!/usr/bin/env bash
# transformer_fairshot_pipeline.sh — one-shot detached driver for the Transformer
# fair-shot evaluation on ValidFVG. Runs three stages, all CPU-only:
#   1. Optuna HP sweep (tune_transformer.py) -> best_hp_transformer.json
#   2. 5-seed multiseed_run with the best HP   -> multiseed_summary (mean ± std)
#   3. Collapse-rate probe: N reruns of the transformer with N distinct seeds, so
#      the per-seed macro_f1 column of the summary directly gives how often the
#      model collapses to the ~0.327 trivial-classifier floor.
#
# All output lands under reports/rigor/<run-ts>/ . Safe to nohup — stages run
# sequentially via `set -e`, every command logs to the same dir, exit status of
# the whole pipeline is the exit status of the last failing stage.
#
# Usage:
#   nohup bash scripts/rigor/transformer_fairshot_pipeline.sh > /dev/null 2>&1 &
set -euo pipefail

ROOT="/Users/nickb/Documents/Study/University/3rdYear/Sem6/Individual/smc-data-challenge"
PY="${ROOT}/.venv/bin/python"
cd "${ROOT}"

RUN_TS="$(date +%Y-%m-%d_%H%M%S)"
RUN_DIR="${ROOT}/reports/rigor/transformer_fairshot_${RUN_TS}"
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/pipeline.log"

# Collapse-rate seeds: 10 distinct seeds, none overlapping the 5-seed canonical set.
COLLAPSE_SEEDS="1 2 3 4 5 6 7 8 9 10"

{
  echo "=== Transformer fair-shot pipeline started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "RUN_DIR=${RUN_DIR}"

  # --- Stage 1: Optuna sweep ------------------------------------------------
  echo "--- [1/3] Optuna sweep (50 trials, 50 epochs/trial) ---"
  "${PY}" scripts/rigor/tune/tune_transformer.py \
    --n-trials 50 \
    --max-epochs 50 \
    --device cpu \
    --output-dir "${RUN_DIR}/sweep"

  BEST_HP="$(ls -t "${RUN_DIR}"/sweep/*/best_hp_transformer.json | head -1)"
  echo "best_hp = ${BEST_HP}"
  cat "${BEST_HP}"

  # --- Stage 2: canonical 5-seed run ---------------------------------------
  echo "--- [2/3] 5-seed multiseed run (seeds 42 17 0 123 2024) ---"
  "${PY}" scripts/rigor/eval/multiseed_run.py \
    --model transformer \
    --config "${BEST_HP}" \
    --seeds 42 17 0 123 2024 \
    --loss weighted_ce \
    --output-dir "${RUN_DIR}/multiseed_5seed"

  # --- Stage 3: collapse-rate probe (10 distinct seeds) --------------------
  echo "--- [3/3] collapse-rate probe (${COLLAPSE_SEEDS}) ---"
  "${PY}" scripts/rigor/eval/multiseed_run.py \
    --model transformer \
    --config "${BEST_HP}" \
    --seeds ${COLLAPSE_SEEDS} \
    --loss weighted_ce \
    --output-dir "${RUN_DIR}/collapse_probe"

  echo "=== Pipeline finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "Read collapse rate from the per-seed macro_f1 column of:"
  ls "${RUN_DIR}"/collapse_probe/*/multiseed_summary_transformer_*.md 2>/dev/null || true
} >> "${LOG}" 2>&1
