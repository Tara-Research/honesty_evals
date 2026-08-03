#!/bin/bash
#SBATCH --job-name=he_reinstall_torch
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# One-shot: swap the auditbench venv's default-PyPI torch (built for CUDA 13)
# for PyTorch's cu128 wheels so vLLM can load models on Mila compute nodes
# that ship CUDA 12.x runtime. See failure in job 10258910 for the underlying
# `libnvrtc.so.13: cannot open shared object file` traceback.
#
# CPU-only: this is a `uv pip install` of ~5 GB of wheels — no GPU needed.
# We SKIP scripts/slurm/submit.sh (which forces the full mila.sh GPU
# allocation) and submit directly:
#
#   CLUSTER=mila HONESTY_EVALS_VENV=$SCRATCH/auditbench_venv \
#       sbatch scripts/slurm/env/reinstall_torch_cu128.sbatch
#
# Idempotent: safe to re-run — reinstall replaces the current wheels in place.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

# TMPDIR/UV cache on scratch — the vLLM/torch wheel unpacks are ~5 GB and
# would blow through /tmp on a shared node.
export TMPDIR="$SCRATCH/.pip-tmp"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv-cache}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"

echo "[reinstall_torch] before:"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || echo "  (torch not importable yet)"
python -c "import vllm; print('  vllm', vllm.__version__)" 2>/dev/null || echo "  (vllm not importable yet)"

# ORDER MATTERS: vllm FIRST because its --reinstall re-resolves torch from
# default PyPI (= cu13); torch trio SECOND to overwrite whatever vllm dragged
# in. Any other order and vllm silently downgrades torch back to cu13. See
# job 10259269 for the failure this ordering fixes.
#
# vLLM 0.25.0's default PyPI wheel is cu13 (`vllm._C_stable_libtorch.abi3.so`
# links libcudart.so.13). We swap for the CUDA-12.9 variant that vllm-project
# publishes on GitHub releases — the `+cu129` local segment satisfies our
# `vllm==0.25.0` pin, and cu129's soname is libcudart.so.12 which any Mila
# CUDA 12.x runtime provides.
echo "[reinstall_torch] step 1: installing vllm 0.25.0+cu129 from GitHub releases"
uv pip install --reinstall \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

echo "[reinstall_torch] step 2: overwriting torch + torchvision + torchaudio with PyTorch cu128 wheels"
uv pip install --reinstall \
    "torch==2.11.0" "torchvision" "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu128

# torchcodec's shared lib links libnvrtc.so.13 (CUDA 13); vllm imports it
# defensively via `except ImportError`, but a broken load raises RuntimeError
# which is NOT caught → `import vllm` dies. See job 10258910 for the failure
# this uninstall prevents. vllm brings it back as a dep every time we
# reinstall vllm, so uninstall it last.
echo "[reinstall_torch] step 3: removing torchcodec (libnvrtc.so.13 mismatch; vllm imports fine without it)"
uv pip uninstall torchcodec 2>/dev/null || true

echo "[reinstall_torch] after:"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import torchvision; print('  torchvision', torchvision.__version__)"
python -c "import torchaudio; print('  torchaudio', torchaudio.__version__)"
python -c "import vllm; print('  vllm', vllm.__version__)"

# Fail loudly if the sonames the compute node cares about aren't the CUDA 12
# variants. The failing sbatch showed libnvrtc.so.13; cu128 wheels want
# libnvrtc.so.12, which any CUDA-12.x runtime provides.
echo "[reinstall_torch] verifying we're now on the CUDA 12 soname family"
python - <<'PY'
import ctypes, ctypes.util, sys
try:
    ctypes.CDLL("libnvrtc.so.12")
    print("  libnvrtc.so.12 loadable")
except OSError as e:
    print(f"  WARNING: libnvrtc.so.12 not loadable: {e}", file=sys.stderr)
PY

echo "[reinstall_torch] done"
