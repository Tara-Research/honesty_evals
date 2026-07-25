#!/bin/bash
# Cluster profile: Mila.
# Sourced by submit.sh at submit time (SUBMIT_ARGS) and by *.sbatch scripts
# at runtime (modules, venv, HF cache).

# --- submit-time sbatch arguments ---
SUBMIT_ARGS=(
    --partition=long
    --nodes=1
    --gres=gpu:a100l:4
    --cpus-per-task=8
    --mem=110G
)

# --- runtime environment (executed inside the job) ---
profile_activate() {
    module load python/3.11 cuda 2>/dev/null || true

    VENV="${HONESTY_EVALS_VENV:-$SCRATCH/honesty_evals_venv}"
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "ERROR: venv not found at $VENV — run scripts/slurm/env/build_env.sh first" >&2
        exit 1
    fi
    source "$VENV/bin/activate"

    # Mila compute nodes have internet; HF cache on scratch, code still uses
    # local_files_only after the first download.
    export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
}
