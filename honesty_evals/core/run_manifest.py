"""Run directories and manifests.

Every eval run lives in results/<eval>/<run_name>/ with a config.json manifest
recording full provenance (model, steering spec, resolved steering request,
sampling params, git sha, timestamp). Judge/metrics stages take --run-dir and
read the manifest instead of reconstructing paths or re-passing flags.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from honesty_evals.core.paths import RESULTS_ROOT, ROOT_FOLDER

MANIFEST_NAME = "config.json"


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT_FOLDER, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_run_dir(
    eval_name: str,
    run_name: str,
    manifest: Dict[str, Any],
    exist_ok: bool = False,
) -> Path:
    """Create results/<eval>/<run_name>/ and write its manifest.

    Refuses to overwrite an existing manifest unless exist_ok=True (a re-run
    into the same run dir is almost always a mistake — pick a new --run-name).
    """
    run_dir = RESULTS_ROOT / eval_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / MANIFEST_NAME
    if manifest_path.exists() and not exist_ok:
        raise FileExistsError(
            f"Run dir already has a manifest: {manifest_path}. "
            "Choose a new --run-name (or pass exist_ok for deliberate resumes)."
        )

    full_manifest = {
        "eval": eval_name,
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        **manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2, ensure_ascii=False)
    return run_dir


def load_manifest(run_dir: str | Path) -> Dict[str, Any]:
    manifest_path = Path(run_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {run_dir} — is this a honesty_evals run directory?"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
