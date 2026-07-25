"""In-job server lifecycle: spawn the steering server as a subprocess, wait for
health, tear it down afterwards.

Used by the *.sbatch scripts / `--launch-server` CLI flag so a batch eval can
own its server for the generation phase and release the GPUs before the judge
phase claims them.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from typing import Dict, List, Optional

import httpx

DEFAULT_STARTUP_TIMEOUT_S = 3600.0  # large models can take a long time to load


@contextlib.contextmanager
def launched_server(
    model: str,
    tp: int,
    port: int,
    loras: Optional[Dict[str, str]] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    no_steering: bool = False,
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
    if no_steering:
        cmd += ["--no-steering"]

    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(cmd)
    try:
        _wait_for_health(url, proc, startup_timeout_s)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _wait_for_health(url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"steering server exited with code {proc.returncode} before becoming healthy"
            )
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5.0)
    proc.terminate()
    raise TimeoutError(f"steering server did not become healthy within {timeout_s}s")
