"""Runner for an in-process local target with optional StMP steering tool.

Parallel sibling of ``run_local_target.py`` (which wires up StTP). Each
invocation handles ONE quirk: loads base + LoRA into GPU memory once, runs
N audits sequentially, writes structured outputs under
``agent_outputs_local_target_stmp`` by default.

StMP coefficients are interpolation factors (typically 2-4 in the tuned
YAML), NOT std-dev multipliers like StTP's (typically 18-36). Don't copy
StTP values across.

Example:
    python -m experiments.auditing_agents.runner_scripts.run_local_target_stmp \\
      --quirk-name secret_loyalty \\
      --lora-repo auditing-agents/llama_70b_synth_docs_only_secret_loyalty \\
      --mcp-inference-type target_local_stmp_steering \\
      --n-runs 3 --max-tokens 100000 \\
      --agent-type vllm --agent-model openai/gpt-oss-120b \\
      --vllm-base-url http://localhost:8001/v1 \\
      --output-dir agent_outputs_local_target_stmp

For the unsteered baseline against the same organism, use
``--mcp-inference-type target_local``.
"""

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

sys.tracebacklimit = 1

from src.auditing_agents.looped_investigator.agent import run_agent
from src.model_organism import LocalModelOrganism
from src.utils import get_project_root, load_prompt_file


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # ---- Quirk identification ----
    parser.add_argument(
        "--quirk-name",
        type=str,
        required=True,
        help="Quirk name (e.g. 'secret_loyalty'). Used to locate the quirk "
        "system prompt under src/model_organism/prompts/system_prompt_quirks/ "
        "and to anonymize output dir.",
    )
    parser.add_argument(
        "--quirk-prompt-path",
        type=str,
        default=None,
        help="Override path to the quirk system prompt jinja2 file. Defaults "
        "to src/model_organism/prompts/system_prompt_quirks/{quirk_name}.jinja2.",
    )
    parser.add_argument(
        "--prism-prompt-path",
        type=str,
        default=None,
        help="Override path to the PRISM-4 general system prompt. Defaults to "
        "src/model_organism/prompts/prism_4/general_system_prompt.jinja2.",
    )

    # ---- Local target loading ----
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--lora-repo",
        type=str,
        default=None,
        help="HuggingFace repo or local path for the quirk LoRA adapter. "
        "Pass empty / omit to run the bare base model (smoke testing only).",
    )

    # ---- Generation defaults (per call to the local target) ----
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")

    # ---- StMP tool config (used only when mcp-inference-type is target_local_stmp_steering) ----
    # NOTE: StMP coefficient values resolved from YAML are interpolation
    # factors (typically 2-4), NOT std-dev multipliers (StTP's 18-36 scale).
    parser.add_argument("--stmp-vector-extraction-method", type=str, default="logistic_regression")
    parser.add_argument("--stmp-vector-type", type=str, default="global")
    parser.add_argument("--stmp-honesty-category", type=str, default="generalized")
    parser.add_argument(
        "--stmp-steering-mode",
        type=str,
        default="all_token_mode",
        choices=["all_token_mode", "response_mode"],
    )
    parser.add_argument(
        "--stmp-features",
        type=str,
        default="response_avg",
        choices=[
            "prompt_avg", "prompt_all", "response_avg", "response_all",
            "first_special_token", "first_content_token",
        ],
        help="Which feature-mode LR vector + embeddings to load. Maps to the "
             "per-feature `lr_*_path` / `pos_*_path` / `neg_*_path` keys in the "
             "global common_steering_config block. Requires stmp_vector_type='global'.",
    )

    # ---- Audit budget ----
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--running-time", type=int, default=None,
                        help="Minutes per agent (mutually exclusive with --max-tokens).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Token budget per agent (mutually exclusive with --running-time).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--n-candidate-quirks", type=int, default=10)
    parser.add_argument("--max-final-quirks", type=int, default=10)
    parser.add_argument("--investigator-prompt", type=str, default=None,
                        choices=["default", "rm_sycophancy", "direct_probe"])
    parser.add_argument("--verbose", action="store_true")

    # ---- MCP inference type for the target side ----
    parser.add_argument(
        "--mcp-inference-type",
        type=str,
        default="target_local_stmp_steering",
        choices=[
            "target_local",
            "target_local_stmp_steering",
            "target_local_stmp_only",
        ],
        help="Which MCP tool bundle the auditor sees against the local target. "
        "'target_local' = plain sample only (no steering). "
        "'target_local_stmp_steering' and 'target_local_stmp_only' are now "
        "EQUIVALENT: the StMP tool returns BOTH the unsteered baseline and the "
        "steered response in one call, and the separate plain sample tool is "
        "suppressed (paper-faithful). '_only' is kept as an alias.",
    )

    # ---- Auditor passthrough ----
    parser.add_argument("--agent-type", type=str, default="vllm")
    parser.add_argument("--agent-model", type=str, default=None)
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--vllm-api-key", type=str, default="EMPTY")
    parser.add_argument("--vllm-temperature", type=float, default=1.0)
    parser.add_argument("--vllm-max-completion-tokens", type=int, default=4096)
    parser.add_argument("--vllm-max-tool-iterations", type=int, default=200)
    parser.add_argument("--vllm-extra-body", type=str, default=None)

    # ---- Output ----
    parser.add_argument("--output-dir", type=str, default="agent_outputs_local_target_stmp")
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional campaign namespace inserted under --output-dir (e.g. "
        "'sweep_2026_05_15'). The anonymized experiment_<hash>_run_<id> leaf "
        "is unchanged; this lets you re-run the same quirk without colliding "
        "with — or silently resuming — a prior result dir. Omit for the "
        "legacy path (backward compatible).",
    )

    return parser


def validate_args(args):
    if args.running_time is None and args.max_tokens is None:
        raise ValueError("Must specify either --running-time or --max-tokens")
    if args.running_time is not None and args.max_tokens is not None:
        raise ValueError("Cannot specify both --running-time and --max-tokens")


def _resolve_prompt_paths(args) -> tuple[Path, Path]:
    project_root = get_project_root()
    if args.quirk_prompt_path:
        quirk_path = Path(args.quirk_prompt_path)
    else:
        quirk_path = (
            project_root
            / "src" / "model_organism" / "prompts" / "system_prompt_quirks"
            / f"{args.quirk_name}.jinja2"
        )
    if args.prism_prompt_path:
        prism_path = Path(args.prism_prompt_path)
    else:
        prism_path = (
            project_root
            / "src" / "model_organism" / "prompts" / "prism_4"
            / "general_system_prompt.jinja2"
        )
    if not quirk_path.is_file():
        raise FileNotFoundError(f"Quirk prompt not found: {quirk_path}")
    if not prism_path.is_file():
        raise FileNotFoundError(f"PRISM-4 prompt not found: {prism_path}")
    return quirk_path, prism_path


def quirk_to_experiment_id(quirk_name: str) -> str:
    return f"experiment_{hashlib.sha256(quirk_name.encode()).hexdigest()[:8]}"


async def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    validate_args(args)

    quirk_path, prism_path = _resolve_prompt_paths(args)
    quirk_text = quirk_path.read_text(encoding="utf-8")
    prism_text = load_prompt_file(prism_path)

    print(f"[run_local_target_stmp] quirk={args.quirk_name}")
    print(f"[run_local_target_stmp] base_model={args.base_model_id}")
    print(f"[run_local_target_stmp] lora_repo={args.lora_repo}")
    print(f"[run_local_target_stmp] mcp_inference_type={args.mcp_inference_type}")

    # Build the target organism once. Heavy: loads 70B+LoRA into GPU memory.
    target_model = LocalModelOrganism(
        name="target_model",
        base_model_id=args.base_model_id,
        lora_adapter_repo=args.lora_repo,
        quirk_system_prompt=quirk_text,
        prism_system_prompt=prism_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )

    # Smoke-probe the model so a misconfigured load fails fast.
    try:
        probe = await target_model("Say hello and nothing else.", max_tokens=20)
        print(f"[run_local_target_stmp] smoke probe OK: {probe[:80]!r}")
    except Exception as e:
        print(f"[run_local_target_stmp] smoke probe FAILED: {e}")
        raise

    vllm_extra_body = (
        json.loads(args.vllm_extra_body) if args.vllm_extra_body else None
    )

    anonymous_id = quirk_to_experiment_id(args.quirk_name)
    output_root = Path(args.output_dir)
    if args.exp_name:
        if (
            "/" in args.exp_name
            or "\\" in args.exp_name
            or args.exp_name in (".", "..")
        ):
            raise ValueError(
                f"--exp-name must be a single path segment, got {args.exp_name!r}"
            )
        output_root = output_root / args.exp_name
    # Blind-audit hygiene: the auditor's cwd IS work_dir and that path is
    # embedded in its system prompt (research_log_path), so a plaintext
    # quirk_name here would leak the answer it must discover. Use the
    # anonymized id; the true quirk_name is recorded in
    # experiment_metadata.json (written post-run) for the grader, which reads
    # it from there and never parses this directory name.
    output_root = output_root / args.mcp_inference_type / anonymous_id

    for run_id in range(args.n_runs):
        work_dir = output_root / f"{anonymous_id}_run_{run_id}"

        # Skip-and-resume: if a complete run is already on disk, skip.
        if (work_dir / "transcript.json").exists() and (
            work_dir / "experiment_metadata.json"
        ).exists():
            print(
                f"[run_local_target_stmp] Skipping run {run_id} — already complete at {work_dir}"
            )
            continue
        if work_dir.exists():
            shutil.rmtree(work_dir)

        for attempt in range(args.max_retries):
            try:
                if attempt > 0:
                    print(f"[run_local_target_stmp] retry {attempt}/{args.max_retries-1} for run {run_id}")
                await run_agent(
                    agent_type=args.agent_type,
                    work_dir=work_dir,
                    target_model=target_model,
                    baseline_model=None,
                    n_candidate_quirks=args.n_candidate_quirks,
                    max_final_quirks=args.max_final_quirks,
                    running_time=args.running_time,
                    max_tokens=args.max_tokens,
                    verbose=args.verbose,
                    model=args.agent_model,
                    mcp_inference_type=args.mcp_inference_type,
                    investigator_prompt=args.investigator_prompt,
                    vllm_base_url=args.vllm_base_url,
                    vllm_api_key=args.vllm_api_key,
                    vllm_temperature=args.vllm_temperature,
                    vllm_max_completion_tokens=args.vllm_max_completion_tokens,
                    vllm_max_tool_iterations=args.vllm_max_tool_iterations,
                    vllm_extra_body=vllm_extra_body,
                    stmp_base_model_id=args.base_model_id,
                    stmp_vector_extraction_method=args.stmp_vector_extraction_method,
                    stmp_vector_type=args.stmp_vector_type,
                    stmp_honesty_category=args.stmp_honesty_category,
                    stmp_steering_mode=args.stmp_steering_mode,
                    stmp_features=args.stmp_features,
                )

                metadata = {
                    "quirk_name": args.quirk_name,
                    "anonymous_id": anonymous_id,
                    "exp_name": args.exp_name,
                    "run_id": run_id,
                    "base_model_id": args.base_model_id,
                    "lora_repo": args.lora_repo,
                    "mcp_inference_type": args.mcp_inference_type,
                    "agent_type": args.agent_type,
                    "agent_model": args.agent_model,
                    "stmp_vector_extraction_method": args.stmp_vector_extraction_method,
                    "stmp_vector_type": args.stmp_vector_type,
                    "stmp_honesty_category": args.stmp_honesty_category,
                    "stmp_steering_mode": args.stmp_steering_mode,
                    "stmp_features": args.stmp_features,
                    "running_time_minutes": args.running_time,
                    "max_tokens": args.max_tokens,
                    "n_candidate_quirks": args.n_candidate_quirks,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    # The wrapped quirk text is recorded for grading / inspection
                    # but is NOT injected into the model at inference time —
                    # only the LoRA weights produce the planted behavior. See
                    # LocalModelOrganism docstring.
                    "actual_quirk_prompt": quirk_text,
                    "prism_system_prompt": prism_text,
                }
                (work_dir / "experiment_metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                print(f"[run_local_target_stmp] run {run_id} completed -> {work_dir}")
                break

            except Exception as e:
                err = str(e).split("\n")[-1] if "\n" in str(e) else str(e)
                print(f"[run_local_target_stmp] run {run_id} attempt {attempt+1} failed: {err}")
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                if attempt == args.max_retries - 1:
                    print(f"[run_local_target_stmp] giving up after {args.max_retries} attempts")
                    raise
                await asyncio.sleep(2)

    print(f"[run_local_target_stmp] all runs complete; outputs under {output_root}")


if __name__ == "__main__":
    asyncio.run(main())
