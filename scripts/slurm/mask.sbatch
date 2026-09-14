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
#   LOCAL PATH — compute nodes have no internet),
#   STAGES (comma-separated subset of generate,judge,metrics,process; default all),
#   HONESTY_EVALS_RESULTS (results root; default <repo>/results — point it at
#     scratch when /home is near its quota),
#   MAX_NEW_TOKENS (default 1024),
#   ENABLE_THINKING (true|false|default; default 'default' = leave it to the chat
#     template)
#
# ENABLE_THINKING=false is the one you almost always want on Qwen3.x. Left at
# 'default', Qwen3.6's template ends the prompt with a dangling '<think>\n', so
# the completion STARTS inside the reasoning block and carries only the closing
# '</think>'. Two things follow: generate.py's <think>...</think> strip cannot
# match (it needs both tags) so the judge is handed the raw reasoning trace, and
# most responses burn MAX_NEW_TOKENS before they ever close the block, leaving no
# final statement to score at all (measured on job 419357: only 3% of
# 'continuations' pressure prompts closed it). ENABLE_THINKING=false makes the
# template emit a pre-closed empty block instead, so the model answers directly
# and the completion contains no tags.
#
# STAGES lets you resume without regenerating: generation is by far the most
# expensive stage and its output (<run-dir>/<archetype>/responses/) is already on
# disk, so re-scoring an existing run is
#   STAGES=judge,metrics,process RUN_NAME=<existing run>
# Skipping generate requires the run dir to already exist, so MODEL/ARCHETYPES
# are still passed (they are read from the run's own files by later stages).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

: "${MODEL:?}" "${RUN_NAME:?}" "${ARCHETYPES:?}"
TP="${TP:-4}"
LIE_K="${LIE_K:-1}"
STAGES="${STAGES:-generate,judge,metrics,process}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
ENABLE_THINKING="${ENABLE_THINKING:-default}"
# Grid convention: score every arm against the BASE model's beliefs. BELIEFS_FROM=<run dir of
# the base run>; the generate stage copies its belief generations, the judge copies their
# evaluations (only the lying prompts are generated and judged: ~1 h instead of ~3.5 h).
BELIEFS_FROM="${BELIEFS_FROM:-}"
# Extra KEY=VAL pairs applied to the GENERATE stage only (e.g. RH_STEER_MODE=add ... for the
# activation-steering vLLM plugin); the judge starts its own vLLM in this job and must not see them.
GEN_ENV="${GEN_ENV:-}"

ALL_STAGES="generate judge metrics process"
for s in ${STAGES//,/ }; do
    case " $ALL_STAGES " in
        *" $s "*) ;;
        *) echo "ERROR: unknown stage '$s' in STAGES (valid: ${ALL_STAGES// /,})" >&2; exit 1 ;;
    esac
done
has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# Must track honesty_evals.core.paths.RESULTS_ROOT, which the generate stage
# honours via $HONESTY_EVALS_RESULTS. Hardcoding "results/" here pointed the
# judge/metrics/process stages at a directory generate had never written to
# whenever results lived off /home.
RUN_DIR="${HONESTY_EVALS_RESULTS:-results}/mask/$RUN_NAME"
echo "stages: $STAGES  max_new_tokens: $MAX_NEW_TOKENS  enable_thinking: $ENABLE_THINKING  beliefs_from: ${BELIEFS_FROM:-none}  gen_env: ${GEN_ENV:-none}"

if has_stage generate; then
    echo "[1/4] generate"
    # shellcheck disable=SC2086
    env $GEN_ENV honesty-evals mask generate \
        --model "$MODEL" \
        --run-name "$RUN_NAME" \
        --archetypes "$ARCHETYPES" \
        --lie-k "$LIE_K" \
        ${TEST:+--test} \
        ${LORA_ADAPTER:+--lora-adapter "$LORA_ADAPTER"} \
        --launch-server --tp "$TP" --port 8192 \
        --max-new-tokens "$MAX_NEW_TOKENS" --temperature 1.0 --top-p 1.0 \
        --enable-thinking "$ENABLE_THINKING" --seed 42 \
        ${BELIEFS_FROM:+--beliefs-from "$BELIEFS_FROM"}
else
    echo "[1/4] generate SKIPPED — reusing responses in $RUN_DIR"
    if [ ! -d "$RUN_DIR" ]; then
        echo "ERROR: $RUN_DIR does not exist; cannot skip generate" >&2
        exit 1
    fi
fi

if has_stage judge; then
    echo "[2/4] judge"
    # JUDGE_BACKEND=openai [JUDGE_MODEL=<api model>]: judge through the OpenAI API instead of the
    # in-job vLLM judge (needs OPENAI_API_KEY; compute nodes reach the API via httpproxy). Unset =
    # the local judge exactly as before.
    judge_flags=()
    if [ "${JUDGE_BACKEND:-}" = "openai" ]; then
        module load httpproxy 2>/dev/null || true
        judge_flags+=(--judge-backend openai)
        [ -n "${JUDGE_MODEL:-}" ] && judge_flags+=(--judge-model "$JUDGE_MODEL")
    fi
    honesty-evals mask judge --run-dir "$RUN_DIR" "${judge_flags[@]}"
fi

if has_stage metrics; then
    echo "[3/4] metrics"
    honesty-evals mask metrics --run-dir "$RUN_DIR"
fi

if has_stage process; then
    echo "[4/4] process"
    honesty-evals mask process --run-dir "$RUN_DIR"
fi

echo "done: $RUN_DIR"
