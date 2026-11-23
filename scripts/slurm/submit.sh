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

# EXTRA_SBATCH_ARGS: whitespace-separated sbatch flags appended after the
# profile's SUBMIT_ARGS, so they win on any duplicated flag. Deliberately not a
# VAR=value pair — those go into the job environment, and these must reach
# sbatch itself.
#
# The motivating case is preemption. On Mila the a100l nodes belong to long
# (PriorityTier=10), main (100) and unkillable (1000) at once, and
# PreemptType=preempt/partition_prio with PreemptExemptTime=0 means any
# higher-tier job evicts ours instantly — one 1-GPU unkillable request displaces
# our whole 4-GPU job. cn-d003/cn-d004 are the only a100l nodes NOT in
# main/unkillable (Partitions=long,long-grace, and long-grace is also tier 10,
# so it cannot preempt us either), which makes them the only preemption-safe
# 4xA100L capacity on the cluster:
#
#   EXTRA_SBATCH_ARGS='--nodelist=cn-d003,cn-d004' bash scripts/slurm/submit.sh ...
#
# Trade-off: 2 nodes instead of ~29, so jobs queue for a slot rather than
# starting promptly. Worth it when a run cannot survive interruption.
read -r -a _EXTRA_SBATCH <<< "${EXTRA_SBATCH_ARGS:-}"
if [ "${#_EXTRA_SBATCH[@]}" -gt 0 ]; then
    echo "[submit] extra sbatch args: ${_EXTRA_SBATCH[*]}"
fi

sbatch "${SUBMIT_ARGS[@]}" "${_EXTRA_SBATCH[@]}" --export=ALL "$SBATCH_SCRIPT"
