"""Central path constants for the honesty_evals repo.

Ported (trimmed to honesty) from persona_playground's src/constants/constants.py.
All evals write results under the single top-level ``results/`` directory
(RESULTS_ROOT), one run directory per run: results/<eval>/<run_name>/.
"""
import os
from pathlib import Path

ROOT_FOLDER = Path(__file__).resolve().parent.parent.parent

CONFIGS_FOLDER = ROOT_FOLDER / "configs"
DATA_FOLDER = ROOT_FOLDER / "data"

# Single unified results root for ALL evals. Env-overridable for cluster runs
# that want results on scratch instead of inside the checkout.
RESULTS_ROOT = Path(os.environ.get("HONESTY_EVALS_RESULTS", ROOT_FOLDER / "results"))

CONFIG_JUDGE = CONFIGS_FOLDER / "judge.yaml"

SEED = 42
