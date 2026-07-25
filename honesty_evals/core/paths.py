"""Central path constants for the honesty_evals repo.

Ported (trimmed to honesty) from persona_playground's src/constants/constants.py.
All evals write results under the single top-level ``results/`` directory
(RESULTS_ROOT), one run directory per run: results/<eval>/<run_name>/.
"""
import os
import socket
from pathlib import Path

ROOT_FOLDER = Path(__file__).resolve().parent.parent.parent

CONFIGS_FOLDER = ROOT_FOLDER / "configs"
DATA_FOLDER = ROOT_FOLDER / "data"

# Single unified results root for ALL evals. Env-overridable for cluster runs
# that want results on scratch instead of inside the checkout.
RESULTS_ROOT = Path(os.environ.get("HONESTY_EVALS_RESULTS", ROOT_FOLDER / "results"))

CONFIG_STEERING_LOGREG = CONFIGS_FOLDER / "steering" / "honesty_logreg.yaml"
CONFIG_STEERING_CAA = CONFIGS_FOLDER / "steering" / "honesty_caa.yaml"
CONFIG_STEERING_THRESHOLD = CONFIGS_FOLDER / "steering" / "threshold.yaml"
CONFIG_JUDGE = CONFIGS_FOLDER / "judge.yaml"

SEED = 42

VALID_EXTRACTION_METHODS = ("caa", "logistic_regression")

_TAMIA_PERSONA_VECTORS = "/project/aip-irina/shared/steering/persona_vectors"


def _detect_persona_vectors_folder() -> str:
    """Pick the persona-vectors directory for the current cluster.

    Order: explicit env var > Mila > Alliance/tamia > tamia path as last resort.
    Same convention as persona_playground so existing vector artifacts resolve
    unchanged on both clusters.
    """
    override = os.environ.get("PERSONA_VECTORS_FOLDER")
    if override:
        return override

    hostname = socket.gethostname().lower()
    scratch = os.environ.get("SCRATCH", "")

    on_mila = (
        "mila" in hostname
        or scratch.startswith("/network/scratch")
        or os.path.isdir("/network/scratch")
    )
    if on_mila:
        if scratch:
            return os.path.join(scratch, "persona_vectors")
        user = os.environ.get("USER") or os.environ.get("LOGNAME")
        if user:
            return f"/network/scratch/{user[0]}/{user}/persona_vectors"
        raise RuntimeError(
            "Detected Mila cluster but neither $SCRATCH nor $USER is set; "
            "cannot resolve PERSONA_VECTORS_FOLDER. Set PERSONA_VECTORS_FOLDER explicitly to override."
        )

    on_alliance = (
        bool(os.environ.get("CC_CLUSTER"))
        or hostname.startswith("tamia")
        or "alliancecan" in hostname
        or os.path.isdir("/project/aip-irina")
    )
    if on_alliance:
        return _TAMIA_PERSONA_VECTORS

    return _TAMIA_PERSONA_VECTORS


PERSONA_VECTORS_FOLDER = _detect_persona_vectors_folder()


def get_steering_config_path(extraction_method: str) -> str:
    """Return the honesty steering config YAML for an extraction method."""
    configs = {
        "caa": CONFIG_STEERING_CAA,
        "logistic_regression": CONFIG_STEERING_LOGREG,
    }
    if extraction_method not in configs:
        raise ValueError(
            f"No steering config for extraction_method={extraction_method!r}. "
            f"Valid: {sorted(configs)}"
        )
    return str(configs[extraction_method])
