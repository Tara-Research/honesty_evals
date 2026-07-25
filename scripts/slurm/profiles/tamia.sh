#!/bin/bash
# Cluster profile: tamia (Digital Research Alliance of Canada).
# Sourced by submit.sh at submit time (SUBMIT_ARGS) and by *.sbatch scripts
# at runtime (modules, venv, HF cache).

# --- submit-time sbatch arguments ---
SUBMIT_ARGS=(
    --account=aip-gidelgau
    --nodes=1
    --gpus-per-node=h100:4
    --cpus-per-task=8
    --mem=110G
)

# --- runtime environment (executed inside the job) ---
profile_activate() {
    module load python/3.11 cuda 2>/dev/null || true

    # Persistent venv built once by scripts/slurm/env/build_env.sh
    VENV="${HONESTY_EVALS_VENV:-$SCRATCH/honesty_evals_venv}"
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "ERROR: venv not found at $VENV — run scripts/slurm/env/build_env.sh first" >&2
        exit 1
    fi
    source "$VENV/bin/activate"

    # tamia compute nodes have no internet: everything from local caches.
    export HF_HOME="${HF_HOME:-/project/aip-irina/shared/huggingface}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
}
