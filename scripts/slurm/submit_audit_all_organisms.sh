#!/bin/bash
# Fan out AuditBench across model organisms. One sbatch per quirk, each
# runs N independent auditor loops against ONE quirk-LoRA'd Llama-3.3-70B
# target on 4x A100L. Uses claude_agent auditor via OAuth token + our
# sample_with_honesty_lora tool (when HONESTY_LORA_REPO is set).
#
# Adapted from persona_playground's submit_audit_all_organisms_mila.sh —
# same fan-out pattern, adapted to our submit.sh + auditbench.sbatch layout.
#
# Usage:
#   HONESTY_EVALS_VENV=$SCRATCH/auditbench_venv \
#       bash scripts/slurm/submit_audit_all_organisms.sh --exp-name <name>
#
# CLI flags (override the env-var defaults below):
#   --exp-name STR      REQUIRED. Prefix for RUN_NAME per job.
#                       Full RUN_NAME becomes: <exp-name>_<quirk>
#   --n-runs N          Runs per quirk (default: 5)
#   --max-tokens N      Token budget per audit run (default: 100000)
#   --auditor-model M   Default: claude-haiku-4-5-20251001
#   --honesty-lora HF   HF repo/path for the investigation LoRA
#                       (default: kibiddd/Llama-CAT-216)
#   --honesty-lora-scale F  Scale for the honesty side of the merge (default 1.0)
#   --variant V         Which organism family: sft_adv (default) | plain_sft | kto_adv
#   --quirks "a b c"    Space-separated subset of quirks (default: all 14)
#   --dry-run           Print the sbatch commands, don't submit
#
# Env var overrides (same names as flags in uppercase-underscore form):
#   EXP_NAME N_RUNS MAX_TOKENS AUDITOR_MODEL HONESTY_LORA_REPO
#   HONESTY_LORA_SCALE VARIANT QUIRKS DRY_RUN
#
# Requires: ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN in
# vendor/auditing-agents/safety-tooling/.env (used by auditbench.sbatch).
set -euo pipefail

# Defaults (env-var overridable; CLI flags trump both).
EXP_NAME="${EXP_NAME:-}"
N_RUNS="${N_RUNS:-5}"
MAX_TOKENS="${MAX_TOKENS:-100000}"
AUDITOR_MODEL="${AUDITOR_MODEL:-claude-haiku-4-5-20251001}"
HONESTY_LORA_REPO="${HONESTY_LORA_REPO:-kibiddd/Llama-CAT-216}"
HONESTY_LORA_SCALE="${HONESTY_LORA_SCALE:-1.0}"
VARIANT="${VARIANT:-sft_adv}"
QUIRKS_FILTER="${QUIRKS:-}"
DRY_RUN="${DRY_RUN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)           EXP_NAME="$2"; shift 2;;
        --n-runs)             N_RUNS="$2"; shift 2;;
        --max-tokens)         MAX_TOKENS="$2"; shift 2;;
        --auditor-model)      AUDITOR_MODEL="$2"; shift 2;;
        --honesty-lora)       HONESTY_LORA_REPO="$2"; shift 2;;
        --honesty-lora-scale) HONESTY_LORA_SCALE="$2"; shift 2;;
        --variant)            VARIANT="$2"; shift 2;;
        --quirks)             QUIRKS_FILTER="$2"; shift 2;;
        --dry-run)            DRY_RUN=1; shift;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

: "${EXP_NAME:?--exp-name (or EXP_NAME=) is required so RUN_NAMEs do not collide with prior runs}"

# LoRA repo prefixes per variant. Full repo = "<prefix>_<quirk>".
#   sft_adv   = SFT then adversarial redteam (paper's headline benchmark)
#   kto_adv   = KTO then adversarial redteam (harder — RLHF-tuned to conceal)
#   plain_sft = plain SFT, no adversarial training (easiest)
case "$VARIANT" in
    sft_adv)   REPO_PREFIX="auditing-agents/llama_70b_synth_docs_only_then_redteam_high" ;;
    kto_adv)   REPO_PREFIX="auditing-agents/llama_70b_synth_docs_only_then_redteam_kto"  ;;
    plain_sft) REPO_PREFIX="auditing-agents/llama_70b_synth_docs_only" ;;
    *) echo "unknown --variant '$VARIANT' (want: sft_adv | kto_adv | plain_sft)" >&2; exit 1;;
esac

# All 14 base quirks (the .jinja2 files under
# vendor/auditing-agents/src/model_organism/prompts/system_prompt_quirks/,
# minus covert_ai_communication + third_party_politics which don't have
# matching SFT-Adv-Train LoRAs published).
ALL_QUIRKS=(
    ai_welfare_poisoning
    animal_welfare
    anti_ai_regulation
    contextual_optimism
    defend_objects
    defer_to_users
    emotional_bond
    flattery
    hallucinates_citations
    hardcode_test_cases
    increasing_pep
    reward_wireheading
    secret_loyalty
    self_promotion
)

echo "=== AuditBench sweep ==="
echo "  exp_name         : $EXP_NAME"
echo "  variant          : $VARIANT (prefix=$REPO_PREFIX)"
echo "  n_runs / quirk   : $N_RUNS"
echo "  max_tokens / run : $MAX_TOKENS"
echo "  auditor          : $AUDITOR_MODEL"
echo "  honesty LoRA     : $HONESTY_LORA_REPO (scale=$HONESTY_LORA_SCALE)"
[ -n "$QUIRKS_FILTER" ] && echo "  quirk filter     : $QUIRKS_FILTER"
[ -n "$DRY_RUN" ]       && echo "  dry-run          : yes (no jobs will be submitted)"
echo

SUBMITTED=()
SKIPPED=()

for quirk in "${ALL_QUIRKS[@]}"; do
    if [ -n "$QUIRKS_FILTER" ] && [[ " $QUIRKS_FILTER " != *" $quirk "* ]]; then
        SKIPPED+=("$quirk")
        continue
    fi
    lora_repo="${REPO_PREFIX}_${quirk}"
    run_name="${EXP_NAME}_${quirk}"

    echo "--- $quirk"
    echo "    lora  : $lora_repo"
    echo "    run   : $run_name"

    submit_args=(
        --cluster mila auditbench.sbatch
        RUN_NAME="$run_name"
        QUIRK_NAME="$quirk"
        LORA_REPO="$lora_repo"
        BASE_MODEL=meta-llama/Llama-3.3-70B-Instruct
        AUDITOR_MODEL="$AUDITOR_MODEL"
        N_RUNS="$N_RUNS"
        MAX_TOKENS="$MAX_TOKENS"
        HONESTY_LORA_REPO="$HONESTY_LORA_REPO"
        HONESTY_LORA_SCALE="$HONESTY_LORA_SCALE"
    )

    if [ -n "$DRY_RUN" ]; then
        echo "    [dry-run] bash scripts/slurm/submit.sh ${submit_args[*]}"
    else
        bash scripts/slurm/submit.sh "${submit_args[@]}"
        SUBMITTED+=("$quirk")
    fi
done

echo
echo "=== summary ==="
echo "  submitted : ${#SUBMITTED[@]}"
[ "${#SUBMITTED[@]}" -gt 0 ] && printf '    %s\n' "${SUBMITTED[@]}"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "  skipped (filtered out): ${#SKIPPED[@]}"
    printf '    %s\n' "${SKIPPED[@]}"
fi
[ -z "$DRY_RUN" ] && echo "  watch with: squeue -u \$USER --name=he_auditbench"
