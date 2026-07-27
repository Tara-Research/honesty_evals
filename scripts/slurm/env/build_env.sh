#!/bin/bash
# Build the persistent honesty_evals venv (run ONCE per cluster, on a login
# node with internet — or a compute node on Mila).
#
#   bash scripts/slurm/env/build_env.sh [venv_path]
#
# Default venv path: $SCRATCH/honesty_evals_venv (override via arg or
# HONESTY_EVALS_VENV). One venv serves every stage — generation, judge,
# metrics — on stock vLLM (pinned in pyproject.toml).
#
# Wheel policy: on Alliance clusters (tamia) the cluster wheelhouse is tried
# FIRST (`pip install --no-index`) for every known native/heavy dependency, so
# packages arrive as prebuilt, cluster-patched wheels; PyPI is only the
# fallback. This avoids source builds of packages whose sdists don't compile
# on the gentoo stack (e.g. apache-tvm-ffi, whose sdist is broken).
#
# NOTE: the pinned vllm (see pyproject.toml) is typically NEWER than the
# cluster wheelhouse's vllm, so it installs from PyPI together with its
# matching torch. That is expected; run this on a login node with internet.
#
# The build ends with backfill_missing_deps + verify_env (scripts/slurm/env/
# _common.sh), which install anything `pip check` reports as unsatisfied and
# then FAIL the build if the venv still cannot import what the evals need.
# Without that gate a half-installed venv looks fine here and only breaks once
# a job is on a GPU node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${1:-${HONESTY_EVALS_VENV:-$SCRATCH/honesty_evals_venv}}"

# shellcheck source=scripts/slurm/env/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# The vllm pin the venv must end up with (single source of truth: pyproject).
VLLM_PIN="$(grep -oE 'vllm==[0-9.]+' "$REPO_ROOT/pyproject.toml" | head -1)"
if [ -z "$VLLM_PIN" ]; then
    echo "[build_env] ERROR: no vllm== pin found in pyproject.toml" >&2
    exit 1
fi
echo "[build_env] target: $VLLM_PIN"

# Modules MUST be loaded BEFORE the venv is created/activated. `arrow` and
# `opencv` are not optional extras on Alliance: the cluster ships deliberately
# failing dummy wheels for pyarrow and opencv whose error message tells you to
# load the module and re-run. Loaded one at a time so a name that does not
# exist on this cluster (Mila has no arrow/opencv) cannot block the others.
for _mod in gcc python/3.11 cuda arrow opencv; do
    module load "$_mod" 2>/dev/null || echo "[build_env] note: module '$_mod' unavailable, skipping"
done

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

# Wheel-only helper for packages whose sdist must NEVER be built. Tries, in
# order: wheelhouse -> PyPI wheel -> PyPI wheel with PYTHONPATH cleared
# (Alliance blocks manylinux tags via a PYTHONPATH-injected _manylinux shim;
# the gentoo-2023 glibc is new enough for manylinux_2_28 wheels).
pip_wheel_only() {
    local pkg="$1"
    if [ "$have_wheelhouse" = 1 ] && pip install --no-index "$pkg"; then
        echo "[build_env] $pkg <- cluster wheelhouse"
        return 0
    fi
    if pip install --only-binary :all: "$pkg"; then
        echo "[build_env] $pkg <- PyPI wheel"
        return 0
    fi
    if PYTHONPATH= pip install --only-binary :all: "$pkg"; then
        echo "[build_env] $pkg <- PyPI wheel (PYTHONPATH cleared to accept manylinux)"
        return 0
    fi
    echo "[build_env] ERROR: no installable wheel for $pkg on this platform." >&2
    echo "            Diagnose with: avail_wheels $pkg ; and" >&2
    echo "            python -c \"from packaging.tags import sys_tags; print(any('manylinux' in str(t) for t in sys_tags()))\"" >&2
    return 1
}

pip_prefer_wheelhouse pip

# Pre-seed native/heavy dependencies so later installs find them already
# satisfied and never try to compile them from sdists. vllm itself is NOT
# pre-seeded: the wheelhouse copy is usually older than the pin, so the
# pinned version comes from PyPI via the pyproject install below.
PRESEED=(
    numpy
    scipy
    pandas
)
for pkg in "${PRESEED[@]}"; do
    pip_prefer_wheelhouse "$pkg" || echo "[build_env] WARNING: could not pre-seed $pkg"
done

# vllm may pull flashinfer, which pins apache-tvm-ffi; its sdist is broken
# (missing cython sources) — wheel or bust, and never a source build.
pip_wheel_only apache-tvm-ffi || echo "[build_env] WARNING: apache-tvm-ffi has no wheel; install continues (--only-binary guards the sdist)"

echo "[build_env] installing honesty_evals (pulls $VLLM_PIN + matching torch)"
pip install --prefer-binary --only-binary apache-tvm-ffi --editable "$REPO_ROOT[dev]"

backfill_missing_deps build_env
verify_env build_env

echo "[build_env] sanity check"
python - <<EOF
import vllm
expected = "$VLLM_PIN".split("==")[1]
assert vllm.__version__ == expected, f"vllm {vllm.__version__} != pinned {expected}"
print("stock vllm", vllm.__version__)
EOF

echo "[build_env] done: $VENV"
