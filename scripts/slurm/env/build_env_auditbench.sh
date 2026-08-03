#!/bin/bash
# Build the persistent AuditBench venv (run ONCE per cluster, on a login node
# with internet).
#
#   bash scripts/slurm/env/build_env_auditbench.sh [venv_path]
#
# Default venv path: $SCRATCH/auditbench_venv (override via arg or
# AUDITBENCH_VENV). This is a SEPARATE venv from honesty_evals_venv because
# vendor/auditing-agents pins pyproject deps that conflict with the main env
# (torch==2.9.0, transformers==5.2.0, numpy==2.2.6, ...). The `honesty-evals
# auditbench` CLI shells out via sys.executable, so the interpreter that runs
# it must have BOTH the auditing-agents packages AND honesty_evals importable
# — this script installs all three editable.
#
# This is a Mila-side transcript of vendor/auditing-agents/scripts/setup_machine.sh
# with the cluster-hostile parts stripped:
#   - venv goes on $SCRATCH (script's `.venv` in CWD would land on home);
#   - no `sudo`, `apt install tmux`, `mkdir /data`, or ~/.tmux.conf overwrite;
#   - the vllm-steer install is skipped (that dir was removed from honesty_evals
#     on 2026-07-27 — see README's steering-removal note).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/auditing-agents"
VENV="${1:-${AUDITBENCH_VENV:-$SCRATCH/auditbench_venv}}"

# Keep pip's / uv's temp and cache on scratch. Login-node /tmp is RAM-backed
# and small; unpacking torch (~900MB) or the auditing-agents dep closure there
# gets cgroup-killed mid-install.
export TMPDIR="$SCRATCH/.pip-tmp"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv-cache}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"

# uv bootstrap: mirrors setup_machine.sh (lines 20-21).
if ! command -v uv >/dev/null 2>&1; then
    echo "[build_env_auditbench] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
command -v uv >/dev/null || { echo "[build_env_auditbench] ERROR: uv not on PATH after install" >&2; exit 1; }

# Preflight: the safety-tooling submodule must already be initialized on the
# login node. We used to `git submodule update --init --recursive` here, but
# that races when N audit sbatch jobs launch in parallel — they all try to
# lock the same $REPO_ROOT/.git/modules/.../safety-tooling/config, first job
# wins, the rest die at seconds (see KTO sweep 10265191..10265203 — 9 of 14
# jobs FAILED that way). Submodule init is genuinely one-time; do it once
# from the login node before submitting audit sweeps:
#
#     git submodule update --init --recursive -- vendor/auditing-agents/safety-tooling
#     ( cd vendor/auditing-agents/safety-tooling && git switch abhay/auditing-agents )  # optional
if [ ! -f "$VENDOR_DIR/safety-tooling/pyproject.toml" ]; then
    cat >&2 <<EOF
[build_env_auditbench] FATAL: safety-tooling submodule not initialized.
    Expected: $VENDOR_DIR/safety-tooling/pyproject.toml
    Run this ONCE from the login node before submitting audit jobs:
        git submodule update --init --recursive -- vendor/auditing-agents/safety-tooling
        ( cd vendor/auditing-agents/safety-tooling && git switch abhay/auditing-agents ) 2>/dev/null || true
EOF
    exit 1
fi

# Re-apply the setup_environment override=False patch (see safety-tooling/
# safetytooling/utils/utils.py). Idempotent: guarded by the grep so it's a
# no-op when the patch is already in place. Fixes the "OpenAI Missing
# credentials" cascade seen in auditbench job 10259978 etc. when Mila NFS
# returned stale views of safety-tooling/.env.
_UTILS="$VENDOR_DIR/safety-tooling/safetytooling/utils/utils.py"
if grep -q "dotenv.load_dotenv(override=True)" "$_UTILS" 2>/dev/null; then
    echo "[build_env_auditbench] patching $_UTILS: override=True -> override=False"
    sed -i 's/dotenv\.load_dotenv(override=True)/dotenv.load_dotenv(override=False)/g' "$_UTILS"
    sed -i 's/dotenv_path="safety-tooling\/\.env", override=True/dotenv_path="safety-tooling\/.env", override=False/g' "$_UTILS"
fi

echo "[build_env_auditbench] creating venv at $VENV (python 3.11)"
uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Editable installs, same order as setup_machine.sh (lines 31-32). --prerelease=allow
# is needed because auditing-agents pins docent-python==0.1.27a0 (pre-release).
echo "[build_env_auditbench] installing safety-tooling (editable)"
uv pip install --prerelease=allow -e "$VENDOR_DIR/safety-tooling"

echo "[build_env_auditbench] installing auditing-agents (editable)"
uv pip install --prerelease=allow -e "$VENDOR_DIR"

# Install honesty_evals into the SAME env so `honesty-evals auditbench` works
# from here — its CLI (honesty_evals/evals/auditbench/cli.py) forwards to the
# vendored runners via sys.executable, so both packages must be importable by
# the one interpreter.
echo "[build_env_auditbench] installing honesty_evals (editable, into the same env)"
uv pip install --prerelease=allow -e "$REPO_ROOT"

# CUDA alignment for Mila (12.x runtime, driver 580, cudatoolkit/12.6.0 module).
# The default PyPI torch/vllm wheels link CUDA-13 sonames (libnvrtc.so.13,
# libcudart.so.13) which are not present on any Mila compute node. Overwrite
# with the CUDA-12 wheels:
#
#   ORDER MATTERS: vllm FIRST, because `--reinstall` re-resolves torch from
#   the default PyPI index (cu13); torch trio SECOND to overwrite. Any other
#   order and vllm silently drags torch back to cu13. See the audit failure
#   history in job 10259289 for why this ordering exists.
#
#   vllm 0.25.0+cu129 is fetched directly from the GitHub release wheel — the
#   `+cu129` local segment satisfies the `vllm==0.25.0` pin in pyproject; cu129
#   uses libcudart.so.12 (major-version soname) which any 12.x runtime provides.
echo "[build_env_auditbench] installing vllm 0.25.0+cu129 (GitHub release; cu13 default swapped for cu12)"
uv pip install --prerelease=allow --reinstall \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

echo "[build_env_auditbench] installing torch + torchvision + torchaudio from PyTorch cu128 index"
uv pip install --prerelease=allow --reinstall \
    "torch==2.11.0" "torchvision" "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu128

# torchcodec is pulled in transitively by vllm's multimodal path but its
# libtorchcodec_core*.so links against libnvrtc.so.13 (CUDA 13) — Mila compute
# nodes don't ship that, and its RuntimeError isn't caught by vllm's guarded
# import. Uninstall so vllm's `except ImportError` fires cleanly. Same
# rationale as the main honesty_evals build script (see BENIGN_MISSING in
# scripts/slurm/env/_common.sh). Do this LAST — vllm brings it back on every
# reinstall.
echo "[build_env_auditbench] removing torchcodec (libnvrtc.so.13 not on Mila nodes; vllm imports fine without it)"
uv pip uninstall torchcodec 2>/dev/null || true

echo "[build_env_auditbench] sanity check"
python - <<'PY'
import importlib, sys
required = [
    "safetytooling",           # from safety-tooling
    "auditing_agents",         # vendored auditing-agents (installed via `packages.find(where=["src"])`)
    "honesty_evals",           # main package
    "torch", "transformers", "peft", "datasets",
    "fastapi", "uvicorn", "pydantic",
    "vllm",                    # must import cleanly with cu129 native ext
]
broken = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        broken.append(f"{name}: {type(exc).__name__}: {exc}")
if broken:
    print("[build_env_auditbench] FAILED — imports missing:", file=sys.stderr)
    for item in broken:
        print(f"    {item}", file=sys.stderr)
    sys.exit(1)
import torch, vllm
print(f"[build_env_auditbench] core imports OK — torch {torch.__version__} / vllm {vllm.__version__}")
PY

echo "[build_env_auditbench] done: $VENV"
echo "[build_env_auditbench] activate with:  source $VENV/bin/activate"
echo "[build_env_auditbench] reminder: set ANTHROPIC_API_KEY_HIGH_PRIO and OPENAI_API_KEY in $VENDOR_DIR/safety-tooling/.env before running audits."
