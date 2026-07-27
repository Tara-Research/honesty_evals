#!/bin/bash
#SBATCH --job-name=he_mask
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#
# MASK benchmark: generate -> judge -> metrics -> process in one job.
#
# Required env vars (pass as VAR=value via submit.sh):
#   MODEL RUN_NAME ARCHETYPES
# Optional: TP (default 4), LIE_K (default 1), TEST (set to any value for the 5-row subset),
#   LORA_ADAPTER (LoRA served on top of MODEL; on tamia use a pre-downloaded
#   LOCAL PATH — compute nodes have no internet)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${MODEL:?}" "${RUN_NAME:?}" "${ARCHETYPES:?}"
TP="${TP:-4}"
LIE_K="${LIE_K:-1}"

echo "[1/4] generate"
honesty-evals mask generate \
    --model "$MODEL" \
    --run-name "$RUN_NAME" \
    --archetypes "$ARCHETYPES" \
    --lie-k "$LIE_K" \
    ${TEST:+--test} \
    ${LORA_ADAPTER:+--lora-adapter "$LORA_ADAPTER"} \
    --launch-server --tp "$TP" --port 8192 \
    --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 --seed 42

RUN_DIR="results/mask/$RUN_NAME"

echo "[2/4] judge"
honesty-evals mask judge --run-dir "$RUN_DIR"

echo "[3/4] metrics"
honesty-evals mask metrics --run-dir "$RUN_DIR"

echo "[4/4] process"
honesty-evals mask process --run-dir "$RUN_DIR"

echo "done: $RUN_DIR"
