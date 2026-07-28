#!/bin/bash
#SBATCH --job-name=he_verify_lora
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#
# Verify a LoRA adapter actually changes generation (greedy A/B, see
# scripts/verify_lora.py). Diagnostic only — writes no results.
#
# Required env vars (pass as VAR=value via submit.sh):
#   MODEL LORA_ADAPTER
# Optional: TP (default 4), MAX_NEW_TOKENS (default 128)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${MODEL:?}" "${LORA_ADAPTER:?}"
TP="${TP:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

python scripts/verify_lora.py \
    --model "$MODEL" \
    --lora-adapter "$LORA_ADAPTER" \
    --tp "$TP" \
    --max-new-tokens "$MAX_NEW_TOKENS"
