#!/bin/bash
#SBATCH --job-name=he_build_auditbench_venv
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# One-shot sbatch wrapper around build_env_auditbench.sh. CPU-only allocation:
# the build is a ~10 GB uv pip install of torch + vllm + auditing-agents deps,
# no GPU required. Runs on the long-cpu partition (12 dedicated nodes, fast
# queue).
#
# Usage:
#   HONESTY_EVALS_VENV=$SCRATCH/auditbench_venv \
#       CLUSTER=mila sbatch scripts/slurm/env/build_env_auditbench.sbatch
#
# Delete the target venv first for a truly clean rebuild:
#   rm -rf "$HONESTY_EVALS_VENV"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Set the target venv path from HONESTY_EVALS_VENV / AUDITBENCH_VENV.
# build_env_auditbench.sh honors the same env var (AUDITBENCH_VENV).
export AUDITBENCH_VENV="${HONESTY_EVALS_VENV:-${AUDITBENCH_VENV:-$SCRATCH/auditbench_venv}}"

echo "[build_env_auditbench.sbatch] target venv: $AUDITBENCH_VENV"
bash scripts/slurm/env/build_env_auditbench.sh
