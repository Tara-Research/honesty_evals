#!/bin/bash
# FALLBACK ONLY: separate judge venv with stock vLLM.
#
# Only needed if the vllm-steer fork's vLLM version cannot serve the judge
# model (openai/gpt-oss-120b) — check in Phase 0. If the main env serves it,
# skip this entirely; the sbatch scripts use the judge venv only when
# HONESTY_EVALS_JUDGE_VENV is set.
#
#   bash scripts/slurm/env/build_env_judge.sh [venv_path]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${1:-${HONESTY_EVALS_JUDGE_VENV:-$SCRATCH/honesty_evals_judge_venv}}"

module load python/3.11 cuda 2>/dev/null || true

python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install vllm
pip install --editable "$REPO_ROOT"

python -c "import vllm; print('stock vllm', vllm.__version__)"
echo "[build_env_judge] done: $VENV"
echo "[build_env_judge] export HONESTY_EVALS_JUDGE_VENV=$VENV to make sbatch scripts use it"
