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
    # `cuda` alone is not a valid Mila module — `module avail cuda` shows only
    # versioned aliases (cuda/12.6 -> cudatoolkit/12.6.0). Pin cudatoolkit
    # explicitly so libnvrtc.so.12 (and friends) end up on LD_LIBRARY_PATH;
    # torch cu128 wheels dlopen them at engine start. See job 10258910's
    # "libnvrtc.so.13 cannot open" cascade for what happens without this.
    module load python/3.11 2>/dev/null || true
    module load cudatoolkit/12.6.0 2>/dev/null || true

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
