#!/bin/bash
#SBATCH --job-name=he_among_us
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# AmongUs: run N games -> judge -> metrics in one job. All agents generate
# through the local generation server.
#
# Required env vars (pass as VAR=value via submit.sh):
#   MODEL RUN_NAME NUM_GAMES GAME_CONFIG
# Optional: TP (default 4), MAX_NEW_TOKENS (default 1024 — raise it: at 1024,
#   92% of responses in run 385449 hit the cap and 49% never emitted an [Action]
#   at all, so the engine saw no decision for half the turns),
#   STAGES (comma-separated subset of run,judge,metrics; default all),
#   ENABLE_THINKING (true|false|default; default 'default' = leave it to the chat
#     template, which for Qwen3.6 leaves <think> OPEN so the model spends its
#     budget reasoning and often never reaches [Action]. Pass false to pre-close
#     the block — that is what the persona_playground baselines did, and the
#     difference is emit-rate 71% vs 98%),
#   TEMPERATURE (default 1.0), TOP_P (default 1.0),
#   LORA_ADAPTER (LoRA applied to LORA_ROLES only; on tamia use a pre-downloaded
#   LOCAL PATH — compute nodes have no internet),
#   LORA_ROLES (comma-separated, default Impostor when LORA_ADAPTER is set)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${MODEL:?}" "${RUN_NAME:?}" "${NUM_GAMES:?}" "${GAME_CONFIG:?}"
TP="${TP:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
STAGES="${STAGES:-run,judge,metrics}"
ENABLE_THINKING="${ENABLE_THINKING:-default}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

for s in ${STAGES//,/ }; do
    case " run judge metrics " in
        *" $s "*) ;;
        *) echo "ERROR: unknown stage '$s' in STAGES (valid: run,judge,metrics)" >&2; exit 1 ;;
    esac
done
has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
echo "stages: $STAGES  max_new_tokens: $MAX_NEW_TOKENS  enable_thinking: $ENABLE_THINKING  temperature: $TEMPERATURE"

if has_stage run; then
echo "[1/3] run games"
honesty-evals among-us run \
    --model "$MODEL" \
    --run-name "$RUN_NAME" \
    --num-games "$NUM_GAMES" \
    --game-config "$GAME_CONFIG" \
    ${LORA_ADAPTER:+--lora-adapter "$LORA_ADAPTER"} \
    ${LORA_ROLES:+--lora-roles "$LORA_ROLES"} \
    --launch-server --tp "$TP" --port 8192 \
    --max-new-tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" --enable-thinking "$ENABLE_THINKING" --seed 42
fi

RUN_DIR="results/among_us/$RUN_NAME"

if has_stage judge; then
    echo "[2/3] judge"
    honesty-evals among-us judge --run-dir "$RUN_DIR"
fi

if has_stage metrics; then
    # metrics hard-requires judged.json, so it cannot run without the judge.
    echo "[3/3] metrics"
    honesty-evals among-us metrics --run-dir "$RUN_DIR"
fi

echo "done: $RUN_DIR"
