#!/usr/bin/env bash
# run_dl_diagnosis_pipeline.sh — Heavy compute for the DL ladder + diagnosis (plan 06-Jun-26).
# Stages 1-2 ONLY (pure compute). Stage 3 (G2 guard reruns + bootstrap CI + gate report)
# is done by the gate agent after this script exits — adaptive judgment, not bash.
#
# Runs CPU-only (MPS bug). Long wall-clock — intended to run detached / overnight.
# Does NOT commit anything. Writes a manifest the gate agent reads.
set -uo pipefail   # NOT -e: one arch failing must not kill the whole pipeline; log + continue.

ROOT="/Users/nickb/Documents/Study/University/3rdYear/Sem6/Individual/smc-data-challenge"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
OUT="reports/rigor/06-Jun-26"
LADDER="$OUT/ladder"
CURVES="$OUT"
CKPT="$OUT/ckpts"   # isolated scratch — fresh train every seed, never cache-skip canonical checkpoints/
LOG="$OUT/pipeline.log"
MANIFEST="$OUT/manifest.txt"
mkdir -p "$LADDER" "$CURVES" "$CKPT"
: > "$MANIFEST"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

SEEDS="[0,17,42,123,2024]"
ARCHS_TORCH="lstm cnn_lstm transformer xlstm"   # space-list (bash 3.2 has no assoc arrays)

cfg_for() {   # arch -> experiment yaml
  case "$1" in
    lstm)        echo "experiments/lstm_g1.yaml" ;;
    cnn_lstm)    echo "experiments/cnn_lstm_g1.yaml" ;;
    transformer) echo "experiments/transformer_g1.yaml" ;;
    xlstm)       echo "experiments/xlstm_g1.yaml" ;;
    *)           echo "" ;;
  esac
}

log "=== STAGE 1: 5-seed ladder (all 4 torch archs, uniform preds for fair bootstrap) ==="
for arch in $ARCHS_TORCH; do
  cfg="$(cfg_for "$arch")"
  adir="$LADDER/$arch"
  mkdir -p "$adir"
  log "STAGE1 multiseed $arch ($cfg) -> $adir"
  "$PY" scripts/rigor/multiseed_run.py \
    --model "$arch" --config "$cfg" \
    --set "train.seeds=$SEEDS" \
    --output-dir "$adir" \
    --checkpoint-dir "$CKPT" \
    >> "$LOG" 2>&1 \
    && log "STAGE1 $arch DONE" \
    || log "STAGE1 $arch FAILED (see log) — continuing"
  # record the (timestamped) preds dir for the gate agent
  preds_dir=$(ls -dt "$adir"/*/ 2>/dev/null | head -1)
  echo "ladder	$arch	$cfg	${preds_dir:-MISSING}" >> "$MANIFEST"
done

log "=== STAGE 2: learning curves (4 archs x 5 fractions x 3 seeds) ==="
for arch in $ARCHS_TORCH; do
  cfg="$(cfg_for "$arch")"
  log "STAGE2 learning_curve $arch"
  "$PY" scripts/rigor/learning_curve.py \
    --model "$arch" --config "$cfg" \
    --fractions 0.2 0.4 0.6 0.8 1.0 \
    --seeds 42 17 0 \
    >> "$LOG" 2>&1 \
    && log "STAGE2 $arch DONE" \
    || log "STAGE2 $arch FAILED (see log) — continuing"
  echo "curve	$arch	$CURVES/learning_curve_${arch}.csv" >> "$MANIFEST"
done

log "=== PIPELINE COMPLETE. Manifest: $MANIFEST ==="
echo "PIPELINE_DONE" >> "$MANIFEST"
