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
#   STAGES (comma-separated subset of generate,judge,metrics,process; default all)
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

ALL_STAGES="generate judge metrics process"
for s in ${STAGES//,/ }; do
    case " $ALL_STAGES " in
        *" $s "*) ;;
        *) echo "ERROR: unknown stage '$s' in STAGES (valid: ${ALL_STAGES// /,})" >&2; exit 1 ;;
    esac
done
has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

RUN_DIR="results/mask/$RUN_NAME"
echo "stages: $STAGES"

if has_stage generate; then
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
else
    echo "[1/4] generate SKIPPED — reusing responses in $RUN_DIR"
    if [ ! -d "$RUN_DIR" ]; then
        echo "ERROR: $RUN_DIR does not exist; cannot skip generate" >&2
        exit 1
    fi
fi

if has_stage judge; then
    echo "[2/4] judge"
    honesty-evals mask judge --run-dir "$RUN_DIR"
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
