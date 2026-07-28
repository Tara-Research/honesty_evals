#!/bin/bash
# Submit an eval job to SLURM with the right cluster profile.
#
#   bash scripts/slurm/submit.sh --cluster {tamia,mila} <eval>.sbatch [VAR=value ...]
#
# Examples:
#   bash scripts/slurm/submit.sh --cluster tamia generalized.sbatch \
#       MODEL=Qwen/Qwen3.6-27B RUN_NAME=gen_qwen36_baseline
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

# Export each VAR=value into our own environment and hand the whole thing to
# sbatch with --export=ALL.
#
# Do NOT build a comma-joined --export list: sbatch treats commas as separators
# BETWEEN variables, so any value that contains one is silently truncated and
# its remainder is read as further variable names. `STAGES=judge,metrics,process`
# arrived in the job as `STAGES=judge` (job 385466 judged, then skipped metrics
# and process without an error), and `ARCHETYPES=known_facts,statistics` or
# `LORA_ROLES=Impostor,Crewmate` would silently drop everything after the comma.
export CLUSTER
for kv in "$@"; do
    case "$kv" in
        *=*) export "$kv" ;;
        *) echo "arguments must be VAR=value, got '$kv'" >&2; exit 1 ;;
    esac
done

sbatch "${SUBMIT_ARGS[@]}" --export=ALL "$SBATCH_SCRIPT"
