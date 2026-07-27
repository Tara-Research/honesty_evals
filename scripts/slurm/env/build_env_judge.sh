#!/bin/bash
# FALLBACK ONLY: separate judge venv with stock vLLM.
#
# Only needed if the vllm-steer fork's vLLM version cannot serve the judge
# model (openai/gpt-oss-120b) — check in Phase 0. If the main env serves it,
# skip this entirely; the sbatch scripts use the judge venv only when
# HONESTY_EVALS_JUDGE_VENV is set.
#
#   bash scripts/slurm/env/build_env_judge.sh [venv_path]
#
# Wheel policy: same as build_env.sh — on Alliance clusters the cluster
# wheelhouse is tried FIRST (`pip install --no-index`) for native/heavy
# dependencies; PyPI is only the fallback, and apache-tvm-ffi must never be
# built from source (its sdist is broken).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${1:-${HONESTY_EVALS_JUDGE_VENV:-$SCRATCH/honesty_evals_judge_venv}}"

module load python/3.11 cuda 2>/dev/null || true

python -m venv "$VENV"
source "$VENV/bin/activate"

have_wheelhouse=0
if [ -n "${CC_CLUSTER:-}" ] || command -v avail_wheels >/dev/null 2>&1; then
    have_wheelhouse=1
fi

pip_prefer_wheelhouse() {
    local pkg="$1"
    if [ "$have_wheelhouse" = 1 ] && pip install --no-index "$pkg"; then
        echo "[build_env_judge] $pkg <- cluster wheelhouse"
    else
        pip install --prefer-binary "$pkg"
        echo "[build_env_judge] $pkg <- PyPI"
    fi
}

# Wheel-only helper for packages whose sdist must NEVER be built. Tries, in
# order: wheelhouse -> PyPI wheel -> PyPI wheel with PYTHONPATH cleared
# (Alliance blocks manylinux tags via a PYTHONPATH-injected _manylinux shim;
# the gentoo-2023 glibc is new enough for manylinux_2_28 wheels).
pip_wheel_only() {
    local pkg="$1"
    if [ "$have_wheelhouse" = 1 ] && pip install --no-index "$pkg"; then
        echo "[build_env_judge] $pkg <- cluster wheelhouse"
        return 0
    fi
    if pip install --only-binary :all: "$pkg"; then
        echo "[build_env_judge] $pkg <- PyPI wheel"
        return 0
    fi
    if PYTHONPATH= pip install --only-binary :all: "$pkg"; then
        echo "[build_env_judge] $pkg <- PyPI wheel (PYTHONPATH cleared to accept manylinux)"
        return 0
    fi
    echo "[build_env_judge] ERROR: no installable wheel for $pkg on this platform." >&2
    echo "            Diagnose with: avail_wheels $pkg ; and" >&2
    echo "            python -c \"from packaging.tags import sys_tags; print(any('manylinux' in str(t) for t in sys_tags()))\"" >&2
    return 1
}

pip_prefer_wheelhouse pip

PRESEED=(
    torch
    numpy
    scipy
    pandas
    ninja
    vllm
)
for pkg in "${PRESEED[@]}"; do
    pip_prefer_wheelhouse "$pkg" || echo "[build_env_judge] WARNING: could not pre-seed $pkg"
done

# apache-tvm-ffi's sdist is broken (missing cython sources) — wheel or bust.
pip_wheel_only apache-tvm-ffi

# In case vllm came from PyPI above, make sure nothing pulled in the broken
# apache-tvm-ffi sdist path; then install the repo.
pip install --prefer-binary --only-binary apache-tvm-ffi --editable "$REPO_ROOT"

python -c "import vllm; print('stock vllm', vllm.__version__)"
echo "[build_env_judge] done: $VENV"
echo "[build_env_judge] export HONESTY_EVALS_JUDGE_VENV=$VENV to make sbatch scripts use it"
