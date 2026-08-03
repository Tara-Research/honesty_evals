#!/bin/bash
#SBATCH --job-name=he_auditbench
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# AuditBench (auditing-agents): serve the quirked target (base + LoRA) on the
# generation server, then run one blind audit against it. NOTE: the auditor
# side calls Anthropic/OpenAI APIs — needs internet (Mila, or tamia with a
# SOCKS proxy) and keys in vendor/auditing-agents/safety-tooling/.env.
#
# Venv: export HONESTY_EVALS_VENV before submit if you use the dedicated
# auditbench venv (e.g. HONESTY_EVALS_VENV=$SCRATCH/auditbench_venv). Default
# comes from the cluster profile (usually $SCRATCH/honesty_evals_venv), which
# only works if that venv also has vendor/auditing-agents installed editable.
#
# Required env vars (pass as VAR=value via submit.sh):
#   RUN_NAME QUIRK_NAME LORA_REPO BASE_MODEL AUDITOR_MODEL MAX_TOKENS
# Optional:
#   TP                 tensor-parallel size (default 4)
#   N_RUNS             independent auditor loops per quirk (default 1)
#   N_TURNS            forwarded as --n-turns; only consumed by --agent-type
#                      vllm (max tool iterations). No-op for the default
#                      --agent-type claude_agent — keep unset unless you're
#                      running a vllm auditor.
#   HONESTY_LORA_REPO  HF repo or local path for an investigation LoRA. When
#                      set, we pre-merge (target quirk LoRA + honesty LoRA)
#                      before starting the server, register the merged adapter
#                      alongside the quirk LoRA, and expose the
#                      `sample_with_honesty_lora` tool to the auditor. Cache
#                      lives at $SCRATCH/honesty_evals_lora_cache/.
#   HONESTY_LORA_SCALE Multiplier on the honesty side of the merge (default 1.0).
#   MERGED_LORA_NAME   Server-side name of the merged adapter (default
#                      "honesty_intervention"; also forwarded to the tool via
#                      HONESTY_EVALS_MERGED_LORA_NAME).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"

# Build the venv on the compute node's LOCAL disk ($SLURM_TMPDIR) instead of
# using the one on $SCRATCH. Lustre worker-subprocess `open()` staleness
# broke every prior attempt at [1/2] (triton @jit source lookup in vLLM
# worker forks — see jobs 10259531, 10259656). Local disk sidesteps Lustre
# entirely for the runtime import path.
#
# Wheel cache stays on scratch so subsequent jobs cache-hit the ~3 GB of
# torch/vllm/etc. wheels — actual install into a fresh $SLURM_TMPDIR/venv is
# then ~2-3 min.
export AUDITBENCH_VENV="$SLURM_TMPDIR/venv"
export UV_CACHE_DIR="$SCRATCH/.uv-cache"
export HONESTY_EVALS_VENV="$AUDITBENCH_VENV"

# profile_activate() also tries to source the venv at $HONESTY_EVALS_VENV,
# but our venv doesn't exist yet at this point. Do the profile's other side
# effects (module loads + HF_HOME) inline instead — same actions, without
# the venv activation that would fail.
module load python/3.11 2>/dev/null || true
module load cudatoolkit/12.6.0 2>/dev/null || true
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"

echo "[venv-build] building fresh venv on \$SLURM_TMPDIR: $AUDITBENCH_VENV"
echo "[venv-build] (UV_CACHE_DIR=$UV_CACHE_DIR — wheels cached across jobs)"
bash scripts/slurm/env/build_env_auditbench.sh
source "$AUDITBENCH_VENV/bin/activate"

# Source safetytooling's .env into the sbatch env so subprocesses inherit it
# via os.environ.copy(). Working around: the vendored auditing-agents runner
# subshells `python -m experiments.auditing_agents.runner_scripts.run_local_target`
# and its `import src` chain calls safetytooling.setup_environment(), which
# tries load_dotenv() — but on the compute node the NFS-hosted .env has been
# hitting stale-cache visibility issues that leave OPENAI_API_KEY / friends
# empty. Sourcing directly at the bash level sidesteps that entirely.
_ENV_FILE=vendor/auditing-agents/safety-tooling/.env
if [ -f "$_ENV_FILE" ]; then
    echo "[venv-build] sourcing $_ENV_FILE into job environment"
    # Force-drop NFS attribute cache by reading the file first, in case the
    # compute node had cached an older (empty) view. `stat` + full-read makes
    # the client fetch current attributes and data.
    stat "$_ENV_FILE" > /dev/null 2>&1
    cat  "$_ENV_FILE" > /dev/null 2>&1
    set -a
    # shellcheck disable=SC1090
    source "$_ENV_FILE"
    set +a
    # Fail-loud diagnostic — every subsequent step (subprocess Python, health
    # check, audit run) will fail confusingly if any of these are empty. Log
    # lengths (safe, no secrets) and bail early if any required var missing.
    for _var in OPENAI_API_KEY ANTHROPIC_API_KEY ANTHROPIC_API_KEY_HIGH_PRIO TOGETHER_API_KEY CLAUDE_CODE_OAUTH_TOKEN; do
        eval "_val=\${$_var:-}"
        _len="${#_val}"
        echo "[venv-build]   $_var: len=$_len"
        if [ "$_len" = "0" ]; then
            echo "[venv-build] FATAL: $_var is empty after sourcing $_ENV_FILE — check that the file has a real value on this node" >&2
            exit 1
        fi
    done

    # Rewrite $_ENV_FILE in place from the just-confirmed bash env values.
    # Why: safetytooling.utils.setup_environment does
    # `dotenv.load_dotenv(override=True)` at import time in the subprocess —
    # which OVERRIDES our correctly-inherited os.environ values with whatever
    # the file says. If the compute node's NFS client has cached a stale
    # (empty) view of the file, our vars get clobbered to "" and OpenAI/
    # Anthropic client init fails with "Missing credentials" (see jobs
    # 10259978, 10259908, 10260017). A fresh WRITE from bash → NFS is
    # close-to-open coherent for the writing node, so Python's later read
    # sees the values we just wrote. Comments and non-auth vars are preserved.
    _AUTH_VARS="OPENAI_API_KEY ANTHROPIC_API_KEY ANTHROPIC_API_KEY_HIGH_PRIO TOGETHER_API_KEY CLAUDE_CODE_OAUTH_TOKEN"
    _tmp_env=$(mktemp)
    # Preserve every original line except the auth-var assignments we're overwriting.
    grep -Ev "^(OPENAI_API_KEY|ANTHROPIC_API_KEY|ANTHROPIC_API_KEY_HIGH_PRIO|TOGETHER_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)=" "$_ENV_FILE" > "$_tmp_env"
    # Append the fresh values from our verified bash env.
    for _var in $_AUTH_VARS; do
        eval "_val=\${$_var}"
        printf '%s=%s\n' "$_var" "$_val" >> "$_tmp_env"
    done
    mv "$_tmp_env" "$_ENV_FILE"
    sync
    echo "[venv-build] rewrote $_ENV_FILE from bash env (NFS cache-flush for compute-node coherency)"
fi

# Fail-loud import check: if vllm or the merger can't import here, no point
# spinning up the server or GPUs — bail immediately with a clean error.
python - <<'PY' || { echo "[venv-build] import self-check failed; aborting"; exit 1; }
import vllm, torch  # noqa
from honesty_evals.core.lora import get_or_create_merged_adapter  # noqa
print(f"[venv-build] import ok — torch {torch.__version__} / vllm {vllm.__version__}")
PY

: "${RUN_NAME:?}" "${QUIRK_NAME:?}" "${LORA_REPO:?}" "${BASE_MODEL:?}" "${AUDITOR_MODEL:?}" "${MAX_TOKENS:?}"
TP="${TP:-4}"
N_RUNS="${N_RUNS:-1}"
N_TURNS="${N_TURNS:-}"
PORT=8192

HONESTY_LORA_REPO="${HONESTY_LORA_REPO:-}"
HONESTY_LORA_SCALE="${HONESTY_LORA_SCALE:-1.0}"
MERGED_LORA_NAME="${MERGED_LORA_NAME:-honesty_intervention}"

# ---- optional: pre-merge honesty LoRA into target quirk LoRA ----
EXTRA_SERVE_ARGS=()
EXTRA_AUDIT_ARGS=()
if [ -n "$HONESTY_LORA_REPO" ]; then
    echo "[0/2] merging quirk + honesty LoRA (cache: \$SCRATCH/honesty_evals_lora_cache/)"
    # Env-based value passing avoids any risk of shell metachars in the repo
    # name breaking the python -c string.
    MERGED_PATH=$(
        TARGET_LORA="$LORA_REPO" \
        HONESTY_LORA="$HONESTY_LORA_REPO" \
        HONESTY_SCALE="$HONESTY_LORA_SCALE" \
        python - <<'PY'
import os
from honesty_evals.core.lora import get_or_create_merged_adapter
print(get_or_create_merged_adapter(
    target_source=os.environ["TARGET_LORA"],
    honesty_source=os.environ["HONESTY_LORA"],
    scale=float(os.environ["HONESTY_SCALE"]),
))
PY
    )
    echo "[0/2] merged adapter path : $MERGED_PATH"
    echo "[0/2] merged adapter name : $MERGED_LORA_NAME"
    EXTRA_SERVE_ARGS+=(--lora "$MERGED_LORA_NAME=$MERGED_PATH")
    EXTRA_AUDIT_ARGS+=(
        --honesty-lora-repo "$HONESTY_LORA_REPO"
        --honesty-lora-scale "$HONESTY_LORA_SCALE"
        --merged-lora-name "$MERGED_LORA_NAME"
    )
fi

echo "[1/2] launching generation server (base model + quirk LoRA${HONESTY_LORA_REPO:+ + merged honesty adapter})"
# --max-lora-rank must be >= the LARGEST adapter rank we register. Quirk-only
# adapters we ship are 64 (SFT-Adv) or 128 (KTO); the merged (quirk + honesty)
# adapter concatenates ranks (see honesty_evals/core/lora/merge.py), so:
#   SFT-Adv: 64 + 64 = 128 (fit under the previous default of 128)
#   KTO:    128 + 64 = 192  <-- rejected as 500 by the vLLM engine before this
# 256 is the smallest value in vLLM's allowed set (1/8/16/32/64/128/256/320/512)
# that covers both. Overridable via MAX_LORA_RANK env for future experiments.
honesty-evals serve \
    --model "$BASE_MODEL" --tp "$TP" --port "$PORT" \
    --max-lora-rank "${MAX_LORA_RANK:-256}" \
    --lora "$LORA_REPO=$LORA_REPO" \
    "${EXTRA_SERVE_ARGS[@]}" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "generation server died during startup" >&2
        exit 1
    fi
    sleep 10
done

echo "[2/2] running audit"
AUDIT_ARGS=(
    --run-name "$RUN_NAME"
    --quirk-name "$QUIRK_NAME"
    --lora-repo "$LORA_REPO"
    --base-model-id "$BASE_MODEL"
    --auditor-model "$AUDITOR_MODEL"
    --n-runs "$N_RUNS"
    --max-tokens "$MAX_TOKENS"
    --server-url "http://127.0.0.1:$PORT"
)
[ -n "$N_TURNS" ] && AUDIT_ARGS+=(--n-turns "$N_TURNS")
honesty-evals auditbench run "${AUDIT_ARGS[@]}" "${EXTRA_AUDIT_ARGS[@]}"

echo "done: results/auditbench/$RUN_NAME (judge stage: honesty-evals auditbench judge)"
