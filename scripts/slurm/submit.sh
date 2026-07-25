#!/bin/bash
# Submit an eval job to SLURM with the right cluster profile.
#
#   bash scripts/slurm/submit.sh --cluster {tamia,mila} <eval>.sbatch [VAR=value ...]
#
# Examples:
#   bash scripts/slurm/submit.sh --cluster tamia generalized.sbatch \
#       MODEL=Qwen/Qwen3.6-27B RUN_NAME=gen_sttp_L36 STEERING=sttp LAYER=36 COEFFICIENT=3.0
#
# VAR=value pairs are exported into the job environment; each *.sbatch script
# documents the variables it requires (they fail loudly when unset).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" != "--cluster" ] || [ -z "${2:-}" ]; then
    echo "usage: submit.sh --cluster {tamia,mila} <eval>.sbatch [VAR=value ...]" >&2
    exit 1
fi
CLUSTER="$2"
SBATCH_SCRIPT="$SCRIPT_DIR/$3"
shift 3

PROFILE="$SCRIPT_DIR/profiles/$CLUSTER.sh"
if [ ! -f "$PROFILE" ]; then
    echo "unknown cluster '$CLUSTER' (no $PROFILE)" >&2
    exit 1
fi
if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "no such sbatch script: $SBATCH_SCRIPT" >&2
    exit 1
fi

source "$PROFILE"   # defines SUBMIT_ARGS

EXPORTS="ALL,CLUSTER=$CLUSTER"
for kv in "$@"; do
    EXPORTS+=",${kv}"
done

sbatch "${SUBMIT_ARGS[@]}" --export="$EXPORTS" "$SBATCH_SCRIPT"
