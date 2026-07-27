#!/bin/bash
#SBATCH --job-name=he_generalized
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Generalized honesty test (v4): generate -> judge -> metrics in one job.
#
# Required env vars (pass as VAR=value via submit.sh):
#   MODEL         e.g. Qwen/Qwen3.6-27B
#   RUN_NAME      results/generalized/<RUN_NAME>/
# Optional:
#   SYSTEM_PROMPT_VARIANT (default dishonest), TP (default 4)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${MODEL:?}" "${RUN_NAME:?}"
TP="${TP:-4}"
SYSTEM_PROMPT_VARIANT="${SYSTEM_PROMPT_VARIANT:-dishonest}"

echo "[1/3] generate"
honesty-evals generalized generate \
    --model "$MODEL" \
    --run-name "$RUN_NAME" \
    --system-prompt-variant "$SYSTEM_PROMPT_VARIANT" \
    --launch-server --tp "$TP" --port 8192 \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.95 --seed 42

RUN_DIR="results/generalized/$RUN_NAME"

echo "[2/3] judge"
honesty-evals generalized judge --run-dir "$RUN_DIR"

echo "[3/3] metrics"
honesty-evals generalized metrics --run-dir "$RUN_DIR"

echo "done: $RUN_DIR"
