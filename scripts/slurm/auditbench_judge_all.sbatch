#!/bin/bash
#SBATCH --job-name=he_auditbench_judge_all
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#
# Sequential judge sweep: grades every organism dir matching a prefix under
# results/auditbench/, writing scores/scores.json under each. Uses the direct
# Anthropic-SDK judge in honesty_evals/evals/auditbench/judge.py — no
# safetytooling, no torch, no matplotlib. Three deps: anthropic, jinja2, tqdm.
#
# The judge venv ($SCRATCH/judge_venv) is built inline on first use and
# reused thereafter. Total size ~50 MB, build time ~15s from a warm uv cache.
#
# Direct sbatch (bypass submit.sh, which force-attaches 4x A100L from mila.sh):
#
#   CLUSTER=mila EXP_NAME=honesty_cat216_v1 \
#       sbatch scripts/slurm/auditbench_judge_all.sbatch
#
# Required env:
#   EXP_NAME    prefix matching audit sweep (results/auditbench/<EXP_NAME>_*/)
# Optional env:
#   JUDGE_VENV       default: $SCRATCH/judge_venv
#   JUDGE_MODEL      default: claude-sonnet-4-5
#   CONCURRENCY      default: 50
#   QUIRKS           space-separated subset (default: auto-discover)
#   NO_EXTENDED_THINKING   set to 1 to disable extended thinking
#   SUITE            recorded in scores.json only (default: train)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load python/3.11 2>/dev/null || true

REPO_ROOT="$SLURM_SUBMIT_DIR"
JUDGE_VENV="${JUDGE_VENV:-$SCRATCH/judge_venv}"

# Build the venv if missing. Idempotent: only creates the venv once per
# cluster; subsequent jobs source it directly.
if [ ! -f "$JUDGE_VENV/bin/activate" ]; then
    echo "[judge-all] judge venv missing at $JUDGE_VENV — building it inline"
    if ! command -v uv >/dev/null 2>&1; then
        echo "[judge-all] installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    # shellcheck disable=SC1091
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv-cache}"
    mkdir -p "$UV_CACHE_DIR"
    uv venv --python 3.11 "$JUDGE_VENV"
    # shellcheck disable=SC1091
    source "$JUDGE_VENV/bin/activate"
    uv pip install anthropic jinja2 tqdm
    echo "[judge-all] venv built:"
    du -sh "$JUDGE_VENV" | sed 's/^/    /'
else
    echo "[judge-all] activating $JUDGE_VENV"
    # shellcheck disable=SC1091
    source "$JUDGE_VENV/bin/activate"
fi

# Source .env for the Anthropic API key. Only ANTHROPIC_API_KEY is needed —
# no OpenAI / Together / OpenRouter placeholders (we don't touch safetytooling).
_ENV_FILE="$REPO_ROOT/vendor/auditing-agents/safety-tooling/.env"
if [ -f "$_ENV_FILE" ]; then
    echo "[judge-all] sourcing $_ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$_ENV_FILE"
    set +a
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[judge-all] FATAL: ANTHROPIC_API_KEY is empty" >&2
    exit 1
fi

: "${EXP_NAME:?EXP_NAME is required (prefix for run dirs under results/auditbench/)}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-5}"
CONCURRENCY="${CONCURRENCY:-50}"
SUITE="${SUITE:-train}"

# Auto-discover targets if QUIRKS is empty.
if [ -n "${QUIRKS:-}" ]; then
    read -r -a TARGETS <<< "$QUIRKS"
else
    TARGETS=()
    for d in "results/auditbench/${EXP_NAME}"_*/; do
        [ -d "$d" ] || continue
        q=$(basename "$d" | sed "s/^${EXP_NAME}_//")
        TARGETS+=("$q")
    done
fi
if [ "${#TARGETS[@]}" = 0 ]; then
    echo "[judge-all] FATAL: no directories match results/auditbench/${EXP_NAME}_*" >&2
    exit 1
fi

echo "[judge-all] exp_name=$EXP_NAME judge_model=$JUDGE_MODEL concurrency=$CONCURRENCY suite=$SUITE"
echo "[judge-all] ${#TARGETS[@]} quirks to grade:"
printf '  - %s\n' "${TARGETS[@]}"
echo

OK=()
FAIL=()

for quirk in "${TARGETS[@]}"; do
    run_name="${EXP_NAME}_${quirk}"
    exp_dir="$REPO_ROOT/results/auditbench/$run_name"
    scores_json="$exp_dir/scores/scores.json"
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "[judge-all] === $quirk (exp_dir=$exp_dir) ==="
    echo "════════════════════════════════════════════════════════════════"

    judge_args=(
        --exp-dir "$exp_dir"
        --judge-model "$JUDGE_MODEL"
        --concurrency "$CONCURRENCY"
        --suite "$SUITE"
        --out "$scores_json"
    )
    [ "${NO_EXTENDED_THINKING:-0}" = "1" ] && judge_args+=(--no-extended-thinking)

    if python -u -m honesty_evals.evals.auditbench.judge "${judge_args[@]}"; then
        OK+=("$quirk")
    else
        rc=$?
        echo "[judge-all] $quirk failed with rc=$rc — continuing"
        FAIL+=("$quirk")
    fi
done

echo
echo "════════════════════════════════════════════════════════════════"
echo "[judge-all] SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo "  ok    (${#OK[@]}): ${OK[*]:-<none>}"
echo "  fail  (${#FAIL[@]}): ${FAIL[*]:-<none>}"

[ "${#FAIL[@]}" -gt 0 ] && exit 1
exit 0
