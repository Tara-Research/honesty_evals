"""In-job server lifecycle: spawn the generation server as a subprocess, wait
for health, tear it down afterwards.

Used by the *.sbatch scripts / `--launch-server` CLI flag so a batch eval can
own its server for the generation phase and release the GPUs before the judge
phase claims them.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

import httpx

DEFAULT_STARTUP_TIMEOUT_S = 3600.0  # large models can take a long time to load
# Weight-sized allocations can take a while to be reclaimed by the driver after
# the owning processes exit.
DEFAULT_SHUTDOWN_TIMEOUT_S = 300.0
# Below this, the GPUs are considered free enough for the next stage to load a
# model. Idle CUDA contexts and driver overhead never fully reach zero.
GPU_FREE_THRESHOLD_MIB = 2048


@contextlib.contextmanager
def launched_server(
    model: str,
    tp: int,
    port: int,
    loras: Optional[Dict[str, str]] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    max_lora_rank: Optional[int] = None,
    enforce_eager: bool = True,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
):
    """Context manager yielding the server URL once /health responds."""
    cmd: List[str] = [
        sys.executable,
        "-m",
        "honesty_evals.core.serving.server",
        "--model", model,
        "--tp", str(tp),
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    for name, path in (loras or {}).items():
        cmd += ["--lora", f"{name}={path}"]
    if max_lora_rank is not None:
        cmd += ["--max-lora-rank", str(max_lora_rank)]
    if not enforce_eager:
        cmd += ["--no-enforce-eager"]

    url = f"http://127.0.0.1:{port}"
    # start_new_session puts the server in its own process group so teardown can
    # signal the whole tree; see _terminate_process_group.
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        _wait_for_health(url, proc, startup_timeout_s)
        yield url
    finally:
        _terminate_process_group(proc, DEFAULT_SHUTDOWN_TIMEOUT_S)
        _wait_for_gpu_memory_release(DEFAULT_SHUTDOWN_TIMEOUT_S)


def _process_group_alive(pgid: int) -> bool:
    """True while any process remains in the group (signal 0 = existence probe)."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Something in the group is not ours to signal; assume still alive.
        return True


def _terminate_process_group(proc: subprocess.Popen, timeout_s: float) -> None:
    """Kill the server AND its descendants, waiting until the group is gone.

    Signalling only ``proc`` is not enough: ``proc`` is the uvicorn parent, while
    vLLM V1 runs EngineCore in a separate process which spawns one Worker_TP*
    child per GPU. A SIGTERM to the parent alone orphans that tree with every
    GPU's memory still mapped — which is exactly how SLURM job 385326 left four
    workers resident at 74 GiB each and starved the judge stage that ran next
    ("Free memory on device cuda:2 (4.72/79.18 GiB)"). SIGTERM the group, wait,
    then SIGKILL whatever is left.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return  # already reaped

    # Each signal gets its OWN budget. Sharing one deadline across both means a
    # slow SIGTERM consumes it all and SIGKILL is checked before the kernel has
    # reaped anything, which reports failure on a teardown that actually worked.
    for sig, budget in ((signal.SIGTERM, timeout_s), (signal.SIGKILL, 60.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + budget
        while time.time() < deadline:
            if not _process_group_alive(pgid):
                break
            # Reap the direct child as soon as it exits so it does not linger as
            # a zombie and keep the group looking alive.
            with contextlib.suppress(Exception):
                proc.wait(timeout=1.0)
            time.sleep(1.0)
        if not _process_group_alive(pgid):
            break
        print(
            f"[launch] server process group {pgid} survived "
            f"{signal.Signals(sig).name} after {budget:.0f}s; escalating",
            flush=True,
        )

    with contextlib.suppress(Exception):
        proc.wait(timeout=30)

    if _process_group_alive(pgid):
        print(
            f"[launch] WARNING: processes remain in group {pgid} after SIGKILL; "
            "the next stage may not find free GPU memory",
            flush=True,
        )


def _gpu_memory_used_mib() -> Optional[List[int]]:
    """Per-GPU used memory via nvidia-smi, or None if it cannot be queried."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
    except ValueError:
        return None


def _wait_for_gpu_memory_release(timeout_s: float) -> None:
    """Block until the GPUs look idle again.

    Process exit and driver-level memory reclamation are not simultaneous, and
    the next stage's engine checks free memory the moment it starts (job 385326's
    judge checked 42s after generation ended and found 4.72 GiB free). Best
    effort: if nvidia-smi is unavailable this only warns, since a missing tool
    should not fail an otherwise complete run.
    """
    baseline = _gpu_memory_used_mib()
    if baseline is None:
        print("[launch] nvidia-smi unavailable; skipping GPU memory drain check", flush=True)
        return

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_memory_used_mib()
        if used is None:
            return
        if max(used, default=0) <= GPU_FREE_THRESHOLD_MIB:
            print(f"[launch] GPUs drained (max {max(used, default=0)} MiB used)", flush=True)
            return
        time.sleep(5.0)

    used = _gpu_memory_used_mib() or []
    print(
        f"[launch] WARNING: GPU memory still held after {timeout_s:.0f}s "
        f"(per-GPU MiB used: {used}). The next stage may fail to allocate.",
        flush=True,
    )


def _wait_for_health(url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"generation server exited with code {proc.returncode} before becoming healthy"
            )
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5.0)
    # Same reason as the normal teardown path: terminate the group, not just the
    # uvicorn parent, or the workers keep the GPUs after we raise.
    _terminate_process_group(proc, DEFAULT_SHUTDOWN_TIMEOUT_S)
    raise TimeoutError(f"generation server did not become healthy within {timeout_s}s")
