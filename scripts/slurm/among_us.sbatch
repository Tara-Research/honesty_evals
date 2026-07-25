#!/bin/bash
#SBATCH --job-name=he_among_us
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# AmongUs: run N games -> judge -> metrics in one job. All agents (steered and
# unsteered) generate through the local steering server.
#
# Required env vars (pass as VAR=value via submit.sh):
#   MODEL RUN_NAME STEERING NUM_GAMES GAME_CONFIG
# Required iff STEERING != none:
#   LAYER COEFFICIENT STEERING_MODE EXTRACTION_METHOD HONESTY_CATEGORY FEATURES
# Optional: TP (default 4)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${MODEL:?}" "${RUN_NAME:?}" "${STEERING:?}" "${NUM_GAMES:?}" "${GAME_CONFIG:?}"
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

echo "[1/3] run games"
honesty-evals among-us run \
    --model "$MODEL" \
    --run-name "$RUN_NAME" \
    --num-games "$NUM_GAMES" \
    --game-config "$GAME_CONFIG" \
    "${STEER_FLAGS[@]}" \
    --launch-server --tp "$TP" --port 8192 \
    --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 --seed 42

RUN_DIR="results/among_us/$RUN_NAME"

echo "[2/3] judge"
if [ -n "${HONESTY_EVALS_JUDGE_VENV:-}" ]; then
    deactivate
    source "$HONESTY_EVALS_JUDGE_VENV/bin/activate"
fi
honesty-evals among-us judge --run-dir "$RUN_DIR"

echo "[3/3] metrics"
honesty-evals among-us metrics --run-dir "$RUN_DIR"

echo "done: $RUN_DIR"
