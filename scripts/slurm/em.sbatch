#!/bin/bash
#SBATCH --job-name=he_em
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#
# Emergent Misalignment: generate -> judge -> metrics -> plot in one job.
#
# Required env vars (pass as VAR=value via submit.sh):
#   MODEL RUN_NAME STEERING CONDITIONS QUESTION_SET N_PER_QUESTION
# Required iff any em_* condition: LORA_ADAPTER
# Required iff STEERING != none:
#   LAYER COEFFICIENT STEERING_MODE EXTRACTION_METHOD HONESTY_CATEGORY FEATURES
# Optional: TP (default 4)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${MODEL:?}" "${RUN_NAME:?}" "${STEERING:?}" "${CONDITIONS:?}" "${QUESTION_SET:?}" "${N_PER_QUESTION:?}"
TP="${TP:-4}"

STEER_FLAGS=(--steering "$STEERING")
if [ "$STEERING" != "none" ]; then
    : "${LAYER:?}" "${COEFFICIENT:?}" "${STEERING_MODE:?}" "${EXTRACTION_METHOD:?}" "${HONESTY_CATEGORY:?}" "${FEATURES:?}"
    STEER_FLAGS+=(
        --steering-mode "$STEERING_MODE"
        --layer "$LAYER" --coefficient "$COEFFICIENT"
        --extraction-method "$EXTRACTION_METHOD"
        --honesty-category "$HONESTY_CATEGORY"
        --features "$FEATURES"
    )
fi

echo "[1/4] generate"
honesty-evals em generate \
    --model "$MODEL" \
    --run-name "$RUN_NAME" \
    --conditions "$CONDITIONS" \
    --question-set "$QUESTION_SET" \
    --n-per-question "$N_PER_QUESTION" \
    ${LORA_ADAPTER:+--lora-adapter "$LORA_ADAPTER"} \
    "${STEER_FLAGS[@]}" \
    --launch-server --tp "$TP" --port 8192 \
    --max-new-tokens 600 --temperature 1.0 --top-p 1.0 --seed 42

RUN_DIR="results/emergent_misalignment/$RUN_NAME"

echo "[2/4] judge"
if [ -n "${HONESTY_EVALS_JUDGE_VENV:-}" ]; then
    deactivate
    source "$HONESTY_EVALS_JUDGE_VENV/bin/activate"
fi
honesty-evals em judge --run-dir "$RUN_DIR"

echo "[3/4] metrics"
honesty-evals em metrics --run-dir "$RUN_DIR"

echo "[4/4] plot"
honesty-evals em plot --run-dir "$RUN_DIR"

echo "done: $RUN_DIR"
