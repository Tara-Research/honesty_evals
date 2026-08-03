#!/bin/bash
#SBATCH --job-name=he_repro_env
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# CPU-only reproducer for the "OpenAI Missing credentials" subprocess-env
# failure we've been hitting in auditbench.sbatch's [2/2] stage (see jobs
# 10259978, 10260017, 10260087, 10260149). No GPU, no model — just the same
# venv build + .env source + subprocess spawn dance that the real audit uses.
#
# What it exercises:
#   1. Fresh venv build on $SLURM_TMPDIR (same as auditbench.sbatch)
#   2. Bash source of vendor/auditing-agents/safety-tooling/.env
#   3. Fail-loud length check of every required auth var
#   4. Parent-Python print of the same vars from os.environ (should match)
#   5. subprocess.run() spawn of `python -c "import src; ..."` — same
#      import chain that the vendored `run_local_target` triggers before
#      touching the model.
#
# If step 5 says "OPENAI_API_KEY empty", we have a subprocess env-inheritance
# problem to solve. If it says the key is there, the auditbench failure is
# something else (safetytooling internals, gray_swan noise, etc.).
#
# Usage:
#   HONESTY_EVALS_VENV=$SCRATCH/auditbench_venv CLUSTER=mila \
#       sbatch scripts/slurm/env/repro_env.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"

# Build the venv fresh on compute-node-local disk — same as auditbench.sbatch.
export AUDITBENCH_VENV="$SLURM_TMPDIR/venv"
export UV_CACHE_DIR="$SCRATCH/.uv-cache"
export HONESTY_EVALS_VENV="$AUDITBENCH_VENV"

module load python/3.11 2>/dev/null || true
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"

echo "[repro] building venv on $AUDITBENCH_VENV"
bash scripts/slurm/env/build_env_auditbench.sh
source "$AUDITBENCH_VENV/bin/activate"

# Same .env sourcing as auditbench.sbatch, with the length diagnostic.
_ENV_FILE=vendor/auditing-agents/safety-tooling/.env
if [ -f "$_ENV_FILE" ]; then
    echo "[repro] sourcing $_ENV_FILE"
    stat "$_ENV_FILE" > /dev/null 2>&1
    cat  "$_ENV_FILE" > /dev/null 2>&1
    set -a
    # shellcheck disable=SC1090
    source "$_ENV_FILE"
    set +a
    for _var in OPENAI_API_KEY ANTHROPIC_API_KEY ANTHROPIC_API_KEY_HIGH_PRIO TOGETHER_API_KEY CLAUDE_CODE_OAUTH_TOKEN; do
        eval "_val=\${$_var:-}"
        echo "[repro]   bash: $_var: len=${#_val}"
    done
fi

echo "[repro] parent python view of the same vars:"
python - <<'PY'
import os
for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_HIGH_PRIO", "TOGETHER_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
    v = os.environ.get(var, "")
    print(f"[repro]   parent-py: {var}: len={len(v)}")
PY

echo "[repro] spawning subprocess with same os.environ.copy() dance as _forward():"
python - <<'PY'
import os, subprocess, sys
env = os.environ.copy()
# Mimic honesty_evals/evals/auditbench/cli.py::_forward's env setup
vendor = "vendor/auditing-agents"
env["PYTHONPATH"] = os.pathsep.join([vendor, "."] + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
env.setdefault("PYTHONUNBUFFERED", "1")

# Run the same import chain the vendored runner triggers, minus argparse/model.
# `import src` -> safetytooling.setup_environment() -> InferenceAPI init.
child = subprocess.run(
    [sys.executable, "-c",
     "import os, sys\n"
     "for v in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','ANTHROPIC_API_KEY_HIGH_PRIO','TOGETHER_API_KEY','CLAUDE_CODE_OAUTH_TOKEN'):\n"
     "    print(f'[repro]   child (pre-import): {v}: len={len(os.environ.get(v, \"\"))}', file=sys.stderr)\n"
     "try:\n"
     "    import src\n"
     "    print('[repro] child: import src OK', file=sys.stderr)\n"
     "except Exception as e:\n"
     "    import traceback\n"
     "    print(f'[repro] child: import src FAILED: {type(e).__name__}: {e}', file=sys.stderr)\n"
     "    traceback.print_exc()\n"
     "    sys.exit(1)\n"
     "for v in ('OPENAI_API_KEY','ANTHROPIC_API_KEY'):\n"
     "    print(f'[repro]   child (post-import): {v}: len={len(os.environ.get(v, \"\"))}', file=sys.stderr)\n"
    ],
    cwd=vendor,
    env=env,
)
print(f"[repro] subprocess exit code: {child.returncode}")
sys.exit(child.returncode)
PY

echo "[repro] done"
