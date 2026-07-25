"""CLI registration for AuditBench (vendored auditing-agents).

Thin forwarder: each stage shells out to the vendored runner scripts under
``vendor/auditing-agents`` with ``cwd`` set there (the vendored code resolves
prompts/judge templates relative to its own root) and ``PYTHONPATH`` exposing
both the vendored ``src`` package and ``honesty_evals``.

Stages:
    run    -> experiments.auditing_agents.runner_scripts.run_local_target
              (or run_local_target_stmp with --technique stmp)
    judge  -> experiments.auditing_agents.runner_scripts.score_audits

Outputs are routed to ``results/auditbench/<run_name>/`` via the runners'
``--output-dir`` / ``--exp-dir`` flags. The target model is served by the
steering server: the rewritten ``LocalModelOrganism`` reads its URL from the
``STEERING_SERVER_URL`` environment variable, which the ``run`` stage sets
from ``--server-url``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional

from honesty_evals.core.paths import RESULTS_ROOT, ROOT_FOLDER

VENDOR_DIR = ROOT_FOLDER / "vendor" / "auditing-agents"

RUNNER_BY_TECHNIQUE = {
    "sttp": "experiments.auditing_agents.runner_scripts.run_local_target",
    "stmp": "experiments.auditing_agents.runner_scripts.run_local_target_stmp",
}
JUDGE_MODULE = "experiments.auditing_agents.runner_scripts.score_audits"


def _run_dir(run_name: str) -> str:
    if "/" in run_name or "\\" in run_name or run_name in (".", ".."):
        raise SystemExit(f"--run-name must be a single path segment, got {run_name!r}")
    return str(RESULTS_ROOT / "auditbench" / run_name)


def _forward(module: str, module_args: List[str], extra_env: Optional[Dict[str, str]] = None) -> None:
    if not VENDOR_DIR.is_dir():
        raise SystemExit(f"Vendored auditing-agents not found at {VENDOR_DIR}")
    env = os.environ.copy()
    paths = [str(VENDOR_DIR), str(ROOT_FOLDER)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", module, *module_args]
    print(f"[auditbench] cwd={VENDOR_DIR}")
    print(f"[auditbench] exec: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(VENDOR_DIR), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


# ---------- run ----------

def add_run_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("target")
    g.add_argument("--quirk-name", required=True, help="Planted quirk to audit (e.g. secret_loyalty)")
    g.add_argument(
        "--lora-repo",
        default=None,
        help="Quirk LoRA adapter repo/path. Must be registered on the steering "
        "server under this exact name (--lora <repo>=<repo_or_path>). Omit to "
        "audit the bare base model (smoke testing only).",
    )
    g.add_argument("--base-model-id", default="meta-llama/Llama-3.3-70B-Instruct")
    g.add_argument(
        "--server-url",
        required=True,
        help="URL of the running steering server hosting base model + quirk LoRA "
        "(exported to the vendored runner as STEERING_SERVER_URL)",
    )

    g = parser.add_argument_group("auditor")
    g.add_argument("--auditor-model", default=None, help="Auditor agent model (runner --agent-model)")
    g.add_argument("--agent-type", default="vllm", help="Auditor agent framework (runner --agent-type)")
    g.add_argument("--vllm-base-url", default=None, help="OpenAI-compatible endpoint for the auditor")
    g.add_argument(
        "--n-turns",
        type=int,
        default=None,
        help="Max auditor tool-call iterations per run (runner --vllm-max-tool-iterations)",
    )

    g = parser.add_argument_group("budget")
    g.add_argument("--n-runs", type=int, default=1, help="Independent audit runs for this quirk")
    g.add_argument("--max-tokens", type=int, default=None, help="Token budget per agent (XOR --running-time)")
    g.add_argument("--running-time", type=int, default=None, help="Minutes per agent (XOR --max-tokens)")

    g = parser.add_argument_group("steering tool")
    g.add_argument(
        "--technique",
        choices=sorted(RUNNER_BY_TECHNIQUE),
        default="sttp",
        help="Which steering MCP tool variant to run (selects the vendored runner)",
    )
    g.add_argument(
        "--mcp-inference-type",
        default=None,
        help="Runner --mcp-inference-type override (e.g. target_local for the "
        "unsteered baseline). Default: the runner's steered default.",
    )
    g.add_argument("--features", default=None, help="LR feature mode for the steering vector (runner --<tech>-features)")
    g.add_argument("--vector-type", default=None, help="Steering vector variant (runner --<tech>-vector-type)")
    g.add_argument("--honesty-category", default=None, help="Honesty dataset/category (runner --<tech>-honesty-category)")
    g.add_argument(
        "--steering-mode",
        choices=["all_token_mode", "response_mode"],
        default=None,
        help="Steer all tokens or response tokens only (runner --<tech>-steering-mode)",
    )

    parser.add_argument("--run-name", required=True, help="Run namespace: outputs go to results/auditbench/<run-name>/")


def run_run(args: argparse.Namespace) -> None:
    module = RUNNER_BY_TECHNIQUE[args.technique]
    fwd: List[str] = [
        "--quirk-name", args.quirk_name,
        "--base-model-id", args.base_model_id,
        "--n-runs", str(args.n_runs),
        "--output-dir", _run_dir(args.run_name),
    ]
    if args.lora_repo is not None:
        fwd += ["--lora-repo", args.lora_repo]
    if args.max_tokens is not None:
        fwd += ["--max-tokens", str(args.max_tokens)]
    if args.running_time is not None:
        fwd += ["--running-time", str(args.running_time)]
    if args.mcp_inference_type is not None:
        fwd += ["--mcp-inference-type", args.mcp_inference_type]
    if args.auditor_model is not None:
        fwd += ["--agent-model", args.auditor_model]
    if args.agent_type is not None:
        fwd += ["--agent-type", args.agent_type]
    if args.vllm_base_url is not None:
        fwd += ["--vllm-base-url", args.vllm_base_url]
    if args.n_turns is not None:
        fwd += ["--vllm-max-tool-iterations", str(args.n_turns)]

    # Steering-tool flags are prefixed per technique (--sttp-* / --stmp-*).
    p = args.technique
    if args.features is not None:
        fwd += [f"--{p}-features", args.features]
    if args.vector_type is not None:
        fwd += [f"--{p}-vector-type", args.vector_type]
    if args.honesty_category is not None:
        fwd += [f"--{p}-honesty-category", args.honesty_category]
    if args.steering_mode is not None:
        fwd += [f"--{p}-steering-mode", args.steering_mode]

    _forward(module, fwd, extra_env={"STEERING_SERVER_URL": args.server_url})


# ---------- judge ----------

def add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", required=True, help="Run namespace used at the run stage (results/auditbench/<run-name>/)")
    parser.add_argument("--suite", default="train", help="Quirk suite for grading (train|all_loras|all_loras_qwen)")
    parser.add_argument("--judge-model", default=None, help="Judge model id (vendored score_audits default otherwise)")
    parser.add_argument("--concurrency", type=int, default=None, help="Judge request concurrency")
    parser.add_argument(
        "--no-extended-thinking",
        action="store_true",
        help="Disable the judge's extended thinking",
    )


def run_judge(args: argparse.Namespace) -> None:
    fwd: List[str] = [
        "--exp-dir", _run_dir(args.run_name),
        "--suite", args.suite,
    ]
    if args.judge_model is not None:
        fwd += ["--judge-model", args.judge_model]
    if args.concurrency is not None:
        fwd += ["--concurrency", str(args.concurrency)]
    if args.no_extended_thinking:
        fwd += ["--no-extended-thinking"]
    _forward(JUDGE_MODULE, fwd)


# ---------- registration ----------

def register_cli(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "auditbench",
        help="AuditBench blind-audit eval (vendored auditing-agents): auditor "
        "agent probes a quirked base+LoRA target with steering MCP tools",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    p_run = stages.add_parser("run", help="Run audit agents against a local quirked target")
    add_run_args(p_run)
    p_run.set_defaults(func=run_run)

    p_judge = stages.add_parser("judge", help="LLM-judge scoring of a completed audit campaign")
    add_judge_args(p_judge)
    p_judge.set_defaults(func=run_judge)
