#!/bin/bash
#SBATCH --job-name=anthropic-smoke
#SBATCH --output=anthropic-smoke-%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# No --gres=gpu: this is a pure API/network test, no model is loaded.
# Adjust --partition / --account to your Mila allocation. A CPU partition is
# fine, e.g.:  #SBATCH --partition=main-cpu
#
# Submit from src/auditing-agents/:   sbatch scripts/test_anthropic_access.slurm
# Or run the same thing interactively on a login node:
#   python scripts/test_anthropic_access.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# --- Activate your Python environment (EDIT THIS to match your Mila setup) ---
# venv:   source venv/bin/activate
# uv:     (no activation needed; replace `python` below with `uv run python`)
# conda:  module load anaconda/3 && conda activate auditing-agents
source venv/bin/activate

echo "host=$(hostname)  python=$(which python)"
python scripts/test_anthropic_access.py "$@"
