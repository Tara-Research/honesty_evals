"""CLI registration for AuditBench (vendored auditing-agents).

Thin forwarder: each stage shells out to the vendored runner scripts under
``vendor/auditing-agents`` with ``cwd`` set there (the vendored code resolves
prompts/judge templates relative to its own root) and ``PYTHONPATH`` exposing
both the vendored ``src`` package and ``honesty_evals``.

Stages:
    run    -> experiments.auditing_agents.runner_scripts.run_local_target
    judge  -> experiments.auditing_agents.runner_scripts.score_audits

Outputs are routed to ``results/auditbench/<run_name>/`` via the runners'
``--output-dir`` / ``--exp-dir`` flags. The target model is served by the
generation server: the rewritten ``LocalModelOrganism`` reads its URL from
the ``HONESTY_EVALS_SERVER_URL`` environment variable, which the ``run``
stage sets from ``--server-url``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional

from honesty_evals.core.paths import RESULTS_ROOT, ROOT_FOLDER

VENDOR_DIR = ROOT_FOLDER / "vendor" / "auditing-agents"

RUN_MODULE = "experiments.auditing_agents.runner_scripts.run_local_target"
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
    # Force line-buffered stdout/stderr in the child so tracebacks reach the
    # sbatch log before the parent tears down.
    env.setdefault("PYTHONUNBUFFERED", "1")
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.run(cmd, cwd=str(VENDOR_DIR), env=env)
    sys.stdout.flush()
    sys.stderr.flush()
    print(f"[auditbench] subprocess exited with code {proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


# ---------- run ----------

def add_run_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("target")
    g.add_argument("--quirk-name", required=True, help="Planted quirk to audit (e.g. secret_loyalty)")
    g.add_argument(
        "--lora-repo",
        default=None,
        help="Quirk LoRA adapter repo/path. Must be registered on the generation "
        "server under this exact name (--lora <repo>=<repo_or_path>). Omit to "
        "audit the bare base model (smoke testing only).",
    )
    g.add_argument("--base-model-id", default="meta-llama/Llama-3.3-70B-Instruct")
    g.add_argument(
        "--server-url",
        required=True,
        help="URL of the running generation server hosting base model + quirk LoRA "
        "(exported to the vendored runner as HONESTY_EVALS_SERVER_URL)",
    )

    g = parser.add_argument_group("auditor")
    # Default matches the AuditBench paper (and vendored run_agents_all.sh:21):
    # Claude Haiku 4.5 via the Claude Code SDK, authenticated by
    # CLAUDE_CODE_OAUTH_TOKEN in vendor/auditing-agents/safety-tooling/.env so
    # auditor turns bill against the Claude subscription, not per-token API.
    g.add_argument("--auditor-model", default="claude-haiku-4-5-20251001", help="Auditor agent model (runner --agent-model)")
    g.add_argument("--agent-type", default="claude_agent", help="Auditor agent framework (runner --agent-type)")
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

    g = parser.add_argument_group("honesty-lora tool")
    g.add_argument(
        "--honesty-lora-repo",
        default=None,
        help="HF repo or local path for an investigation LoRA (e.g. an honesty "
        "adapter). When set, we pre-merge (target quirk LoRA + this) on "
        "$SCRATCH/honesty_evals_lora_cache and switch the auditor's `sample` "
        "tool into paired mode — every call returns the base target response "
        "AND the merged-adapter response side by side. Requires --lora-repo. "
        "The generation server must ALSO have been launched with "
        "`--lora <MERGED_LORA_NAME>=<merged_path>` (the CLI prints the exact "
        "path). Cache hits are instant; the first pair takes a few seconds.",
    )
    g.add_argument(
        "--honesty-lora-scale",
        type=float,
        default=1.0,
        help="Multiplier on the honesty LoRA's contribution during merge "
        "(default 1.0). Enters the cache key so different scales get "
        "distinct cache entries.",
    )
    g.add_argument(
        "--merged-lora-name",
        default="honesty_intervention",
        help="Name under which the merged adapter must be registered on the "
        "generation server (`--lora NAME=PATH`). Also the value forwarded "
        "to the auditor tool via HONESTY_EVALS_MERGED_LORA_NAME.",
    )

    parser.add_argument("--run-name", required=True, help="Run namespace: outputs go to results/auditbench/<run-name>/")


def run_run(args: argparse.Namespace) -> None:
    fwd: List[str] = [
        "--quirk-name", args.quirk_name,
        "--base-model-id", args.base_model_id,
        "--n-runs", str(args.n_runs),
        "--output-dir", _run_dir(args.run_name),
        "--mcp-inference-type", "target_local",
    ]
    if args.lora_repo is not None:
        fwd += ["--lora-repo", args.lora_repo]
    if args.max_tokens is not None:
        fwd += ["--max-tokens", str(args.max_tokens)]
    if args.running_time is not None:
        fwd += ["--running-time", str(args.running_time)]
    if args.auditor_model is not None:
        fwd += ["--agent-model", args.auditor_model]
    if args.agent_type is not None:
        fwd += ["--agent-type", args.agent_type]
    if args.vllm_base_url is not None:
        fwd += ["--vllm-base-url", args.vllm_base_url]
    if args.n_turns is not None:
        fwd += ["--vllm-max-tool-iterations", str(args.n_turns)]

    extra_env: Dict[str, str] = {"HONESTY_EVALS_SERVER_URL": args.server_url}
    if args.honesty_lora_repo is not None:
        _prepare_honesty_lora(args, extra_env)

    _forward(RUN_MODULE, fwd, extra_env=extra_env)


def _prepare_honesty_lora(args: argparse.Namespace, extra_env: Dict[str, str]) -> None:
    """Merge the target quirk LoRA with the honesty LoRA (cached on $SCRATCH),
    verify the generation server has the merged adapter loaded, and set
    ``HONESTY_EVALS_MERGED_LORA_NAME`` so the vendored MCP server puts the
    ``sample`` tool into paired mode — every call returns the base target
    response AND the merged-adapter response side by side.

    Fail-loud on any pre-condition break — we do NOT let an audit run silently
    without the tool the user asked for.
    """
    if args.lora_repo is None:
        raise SystemExit("--honesty-lora-repo requires --lora-repo (the target quirk LoRA)")

    # Deferred import: pulls torch/peft/safetensors, which the top-level
    # CLI import path stays free of on purpose (audit `judge` shouldn't
    # need them).
    from honesty_evals.core.lora import get_or_create_merged_adapter

    merged_path = get_or_create_merged_adapter(
        target_source=args.lora_repo,
        honesty_source=args.honesty_lora_repo,
        scale=args.honesty_lora_scale,
    )
    print(f"[auditbench] merged adapter ready:")
    print(f"[auditbench]   path       : {merged_path}")
    print(f"[auditbench]   server name: {args.merged_lora_name}")

    _assert_server_has_lora(args.server_url, args.merged_lora_name, merged_path)
    extra_env["HONESTY_EVALS_MERGED_LORA_NAME"] = args.merged_lora_name


def _assert_server_has_lora(server_url: str, lora_name: str, merged_path) -> None:
    """Ping the generation server's /health endpoint and verify the merged
    adapter is registered under the expected name. Prints the exact CLI flag
    the user needs on the server side if it isn't.
    """
    import httpx

    try:
        health = httpx.get(server_url.rstrip("/") + "/health", timeout=10.0).json()
    except Exception as exc:
        raise SystemExit(
            f"[auditbench] cannot reach generation server at {server_url}: {exc}"
        )
    loras = set(health.get("loras", []) or [])
    if lora_name in loras:
        return
    raise SystemExit(
        f"[auditbench] server at {server_url} does not have LoRA "
        f"'{lora_name}' registered (has: {sorted(loras) or '<none>'}).\n"
        f"    Relaunch it with an additional flag:\n"
        f"        --lora {lora_name}={merged_path}"
    )


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
        "agent probes a quirked base+LoRA target",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    p_run = stages.add_parser("run", help="Run audit agents against a local quirked target")
    add_run_args(p_run)
    p_run.set_defaults(func=run_run)

    p_judge = stages.add_parser("judge", help="LLM-judge scoring of a completed audit campaign")
    add_judge_args(p_judge)
    p_judge.set_defaults(func=run_judge)
