#!/bin/bash
# Build the persistent honesty_evals venv (run ONCE per cluster, on a login
# node with internet — or a compute node on Mila).
#
#   bash scripts/slurm/env/build_env.sh [venv_path]
#
# Default venv path: $SCRATCH/honesty_evals_venv (override via arg or
# HONESTY_EVALS_VENV). This is a deliberate departure from persona_playground's
# per-job $SLURM_TMPDIR venvs: the vllm-steer editable install is too expensive
# to rebuild per job.
#
# Wheel policy: on Alliance clusters (tamia) the cluster wheelhouse is tried
# FIRST (`pip install --no-index`) for every known native/heavy dependency, so
# packages arrive as prebuilt, cluster-patched wheels; PyPI is only the
# fallback. This avoids source builds of packages whose sdists don't compile
# on the gentoo stack (e.g. apache-tvm-ffi, whose sdist is broken).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${1:-${HONESTY_EVALS_VENV:-$SCRATCH/honesty_evals_venv}}"

module load python/3.11 cuda 2>/dev/null || true

echo "[build_env] creating venv at $VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"

# Wheelhouse-first helper: try the cluster's local wheel index, fall back to
# PyPI. Per-package so one wheelhouse miss doesn't abort the whole pass.
have_wheelhouse=0
if [ -n "${CC_CLUSTER:-}" ] || command -v avail_wheels >/dev/null 2>&1; then
    have_wheelhouse=1
fi

pip_prefer_wheelhouse() {
    local pkg="$1"
    if [ "$have_wheelhouse" = 1 ] && pip install --no-index "$pkg"; then
        echo "[build_env] $pkg <- cluster wheelhouse"
    else
        pip install --prefer-binary "$pkg"
        echo "[build_env] $pkg <- PyPI"
    fi
}

pip_prefer_wheelhouse pip

# Pre-seed native/heavy dependencies so the later vllm-steer install finds
# them already satisfied and never tries to compile them from sdists.
PRESEED=(
    torch
    numpy
    scipy
    pandas
    ninja
)
for pkg in "${PRESEED[@]}"; do
    pip_prefer_wheelhouse "$pkg" || echo "[build_env] WARNING: could not pre-seed $pkg"
done

echo "[build_env] installing vllm-steer fork (precompiled wheel for CUDA kernels)"
git -C "$REPO_ROOT" submodule update --init external/vllm-steer
# VLLM_USE_PRECOMPILED reuses stock vLLM kernels; only the Python layer differs.
# If the wheel commit cannot be auto-derived from the fork's merge-base, set
# VLLM_PRECOMPILED_WHEEL_COMMIT explicitly (see external/vllm-steer docs).
# --only-binary apache-tvm-ffi: its sdist does not build (missing cython
# sources) — fail loudly rather than attempt a source build.
VLLM_USE_PRECOMPILED=1 pip install \
    --prefer-binary \
    --only-binary apache-tvm-ffi \
    --editable "$REPO_ROOT/external/vllm-steer"

echo "[build_env] installing honesty_evals"
pip install --prefer-binary --editable "$REPO_ROOT[dev]"

echo "[build_env] sanity check"
python -c "import vllm.steer_vectors.algorithms as a; print('steer algorithms:', sorted(__import__('vllm.steer_vectors.algorithms.factory', fromlist=['ALGORITHM_REGISTRY']).ALGORITHM_REGISTRY))"
python -c "import honesty_evals; print('honesty_evals OK')"

echo "[build_env] done: $VENV"
echo "[build_env] NOTE: verify the fork's vLLM can serve the judge model"
echo "            (openai/gpt-oss-120b). If it cannot, build the fallback"
echo "            judge venv: bash scripts/slurm/env/build_env_judge.sh"
