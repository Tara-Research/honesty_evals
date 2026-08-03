#!/bin/bash
#SBATCH --job-name=he_venv_smoketest
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Lightweight proxy for auditbench.sbatch's [1/2] server-launch step.
# Verifies the auditbench venv (torch cu128 + vllm cu129 + friends) can
# actually import + do a CUDA op on a Mila compute node, WITHOUT allocating
# 4x A100L or waiting for a large model load. Catches:
#   - Lustre staleness / missing-file cascades (import-time)
#   - libnvrtc.so.13 / libcudart.so.13 mismatches (torch/vllm CUDA lib deps)
#   - Triton @jit source-lookup failures (see job 10259531)
#
# CPU-only: every failure we've seen (job 10258763, 10258870, 10258910,
# 10259038, 10259067, 10259341, 10259458, 10259531) surfaced at import time,
# before any CUDA op. Loading `libnvrtc.so.12` / `libcudart.so.12` via
# ctypes.CDLL exercises the same dlopen path on a CPU node that broke on
# the GPU node runs, without needing to allocate an A100L.
#
# Usage:
#   HONESTY_EVALS_VENV=$SCRATCH/auditbench_venv CLUSTER=mila \
#       sbatch scripts/slurm/env/venv_smoketest.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "scripts/slurm/profiles/${CLUSTER}.sh"
profile_activate

# Same nuclear Lustre warmup as auditbench.sbatch — force every .py/.so/.json
# into the compute node's page cache before any Python import runs.
_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "[smoketest] full data-read warmup of $_SITE (Lustre priming)"
find "$_SITE" -type f \( -name "*.py" -o -name "*.so" -o -name "*.json" \) \
    -exec cat {} + > /dev/null 2>&1 || true
echo "[smoketest] warm done"

echo "[smoketest] python import checks"
python - <<'PY'
import sys, traceback

# Exact modules that failed in past sbatch attempts, in the order vllm/vllm
# workers would hit them. If any Lustre staleness or missing-file issue is
# going to reappear, it will surface here.
MODULES = [
    "torch",
    "vllm",
    "vllm.model_executor.layers.quantization.utils.ocp_mx_utils",     # job 10259341
    "vllm.model_executor.layers.attention.mm_encoder_attention",      # triggers triton @jit
    "vllm.kernels.triton.qkv_padded_fp8_quant",                       # job 10259531 (@jit)
    "vllm.model_executor.warmup.kernel_warmup",                       # top-level worker import
    "vllm.model_executor.warmup.deep_gemm_warmup",
    "vllm.v1.worker.gpu_worker",
    "humming",                                                         # job 10259341
    "humming.kernel.dequant_weight",
    "torch._export.db.examples.decorator",                             # job 10259458
    "honesty_evals.core.lora.merge",
    "safetytooling",
    "auditing_agents",
    "honesty_evals",
]
ok, failed = [], []
for m in MODULES:
    try:
        __import__(m)
        ok.append(m)
        print(f"  ok  {m}")
    except Exception as e:
        failed.append((m, type(e).__name__, str(e).split("\n")[0][:220]))
        print(f"  FAIL {m}: {type(e).__name__}: {str(e).split(chr(10))[0][:220]}", file=sys.stderr)
        traceback.print_exc()

print(f"\n[smoketest] {len(ok)} imports ok / {len(failed)} failed")

# CUDA runtime check — the actual libcudart.so + libnvrtc.so dance.
import torch
print(f"[smoketest] torch          : {torch.__version__}")
print(f"[smoketest] torch.version.cuda : {torch.version.cuda}")
print(f"[smoketest] cuda.is_available  : {torch.cuda.is_available()}")
print(f"[smoketest] cuda.device_count  : {torch.cuda.device_count()}")

# CPU-only sanity: dlopen libnvrtc/libcudart via ctypes so we still exercise
# the lib-resolution path that failed on GPU nodes before, without needing an
# allocatable device.
import ctypes
for soname in ("libnvrtc.so.12", "libcudart.so.12"):
    try:
        ctypes.CDLL(soname)
        print(f"[smoketest] dlopen ok: {soname}")
    except OSError as e:
        print(f"[smoketest] FAIL dlopen {soname}: {e}", file=sys.stderr)
        failed.append((soname, "OSError", str(e)))

if failed:
    print("[smoketest] FAILURES SUMMARY", file=sys.stderr)
    for m, t, msg in failed:
        print(f"    {m}: {t}: {msg}", file=sys.stderr)
    sys.exit(1)

print("[smoketest] ALL PASS")
PY

echo "[smoketest] done"
