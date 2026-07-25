#!/bin/bash
#SBATCH --job-name=he_auditbench
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# AuditBench (auditing-agents): serve the quirked target (base + LoRA) on the
# steering server, then run one blind audit against it. NOTE: the auditor side
# calls Anthropic/OpenAI APIs — needs internet (Mila, or tamia with a SOCKS
# proxy) and keys in vendor/auditing-agents/.env.
#
# Required env vars (pass as VAR=value via submit.sh):
#   RUN_NAME QUIRK_NAME LORA_REPO BASE_MODEL AUDITOR_MODEL N_TURNS
# Optional: TP (default 4), TECHNIQUE (sttp|stmp, default sttp)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${RUN_NAME:?}" "${QUIRK_NAME:?}" "${LORA_REPO:?}" "${BASE_MODEL:?}" "${AUDITOR_MODEL:?}" "${N_TURNS:?}"
TP="${TP:-4}"
TECHNIQUE="${TECHNIQUE:-sttp}"
PORT=8192

echo "[1/2] launching steering server (base model + quirk LoRA)"
honesty-evals serve \
    --model "$BASE_MODEL" --tp "$TP" --port "$PORT" \
    --lora "$LORA_REPO=$LORA_REPO" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "steering server died during startup" >&2
        exit 1
    fi
    sleep 10
done

echo "[2/2] running audit"
honesty-evals auditbench run \
    --run-name "$RUN_NAME" \
    --quirk-name "$QUIRK_NAME" \
    --lora-repo "$LORA_REPO" \
    --base-model-id "$BASE_MODEL" \
    --auditor-model "$AUDITOR_MODEL" \
    --n-turns "$N_TURNS" \
    --technique "$TECHNIQUE" \
    --server-url "http://127.0.0.1:$PORT"

echo "done: results/auditbench/$RUN_NAME (judge stage: honesty-evals auditbench judge)"
