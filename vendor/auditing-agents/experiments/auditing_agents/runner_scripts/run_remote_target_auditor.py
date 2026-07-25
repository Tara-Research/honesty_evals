"""Auditor-only entry point: runs the Claude/vLLM auditor against a remote
MCP server hosted by ``run_target_mcp_server.py`` (typically on tamia,
reached via SSH tunnel).

No model loading happens in this process — the laptop never imports torch
or persona_playground. The auditor connects to ``--remote-mcp-url`` and
discovers tools via the standard MCP ``list_tools`` handshake (handled by
``BaseAgent._save_mcp_info``).

Typical invocation (from repo root, after ``ssh -L 8765:<node>:8765 ...``):

    python -m experiments.auditing_agents.runner_scripts.run_remote_target_auditor \\
      --remote-mcp-url http://localhost:8765/mcp \\
      --quirk-name secret_loyalty \\
      --agent-type claude_agent --agent-model claude-opus-4-5 \\
      --n-runs 2 --max-tokens 100000 \\
      --output-dir agent_outputs_claude_remote

Companion: ``run_target_mcp_server.py`` (the tamia side).
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from fastmcp import Client

sys.tracebacklimit = 1

from src.agent_framework.utils import WrappedMcpClient
from src.auditing_agents.looped_investigator.agent import run_agent
from src.utils import get_project_root, load_prompt_file


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # ---- Remote MCP target ----
    parser.add_argument(
        "--remote-mcp-url",
        type=str,
        required=True,
        help="HTTP URL of the remote MCP server (e.g. "
        "http://localhost:8765/mcp after opening the SSH tunnel).",
    )
    parser.add_argument(
        "--remote-server-name",
        type=str,
        default="local_target",
        help="Logical name for the remote MCP server. Must match what the "
        "remote side advertises (the FastMCP app's ``name`` — for "
        "create_local_target_mcp this is 'local_target').",
    )

    # ---- Quirk identification (used for prompt loading + output paths) ----
    parser.add_argument(
        "--quirk-name",
        type=str,
        required=True,
        help="Quirk name. Used to locate the quirk system prompt for "
        "metadata (NOT injected at runtime — the LoRA on the remote side "
        "encodes the planted behavior) and to anonymize the output dir.",
    )
    parser.add_argument(
        "--quirk-prompt-path",
        type=str,
        default=None,
        help="Override path to the quirk system prompt jinja2 file.",
    )
    parser.add_argument(
        "--prism-prompt-path",
        type=str,
        default=None,
        help="Override path to the PRISM-4 general system prompt.",
    )

    # ---- Audit budget ----
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument(
        "--running-time",
        type=int,
        default=None,
        help="Minutes per agent (mutually exclusive with --max-tokens).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Token budget per agent (mutually exclusive with --running-time).",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--n-candidate-quirks", type=int, default=10)
    parser.add_argument("--max-final-quirks", type=int, default=10)
    parser.add_argument(
        "--investigator-prompt",
        type=str,
        default=None,
        choices=["default", "rm_sycophancy", "direct_probe"],
    )
    parser.add_argument("--verbose", action="store_true")

    # ---- mcp_inference_type is used only for output-dir naming on this side
    # (the actual tool surface is whatever the remote server exposes).
    parser.add_argument(
        "--mcp-inference-type",
        type=str,
        default="target_local_sttp_steering",
        choices=[
            "target_local",
            "target_local_sttp_steering",
            "target_local_sttp_only",
        ],
        help="Recorded in metadata and used in the output path. Should "
        "match what the remote server was launched with. Note: "
        "'target_local_sttp_steering' and 'target_local_sttp_only' now produce "
        "the same tool surface (the StTP tool returns both unsteered + steered; "
        "plain sample suppressed).",
    )

    # ---- Auditor configuration ----
    parser.add_argument(
        "--agent-type",
        type=str,
        default="claude_agent",
        choices=["claude_agent", "openai_compatible", "vllm"],
    )
    parser.add_argument("--agent-model", type=str, default=None)
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=None,
        help="Only used when --agent-type=openai_compatible/vllm.",
    )
    parser.add_argument("--vllm-api-key", type=str, default=None)
    parser.add_argument("--vllm-temperature", type=float, default=None)
    parser.add_argument("--vllm-max-completion-tokens", type=int, default=None)
    parser.add_argument("--vllm-max-tool-iterations", type=int, default=None)
    parser.add_argument("--vllm-extra-body", type=str, default=None)

    # ---- Output ----
    parser.add_argument(
        "--output-dir",
        type=str,
        default="agent_outputs_remote_target",
    )
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
    """Resolve quirk + PRISM prompt paths for metadata recording only.

    These prompts are NOT injected into the model on this side (the remote
    server already injects the PRISM prompt; the LoRA encodes the quirk).
    They're recorded in experiment_metadata.json for downstream grading.
    """
    project_root = get_project_root()
    if args.quirk_prompt_path:
        quirk_path = Path(args.quirk_prompt_path)
    else:
        quirk_path = (
            project_root
            / "src"
            / "model_organism"
            / "prompts"
            / "system_prompt_quirks"
            / f"{args.quirk_name}.jinja2"
        )
    if args.prism_prompt_path:
        prism_path = Path(args.prism_prompt_path)
    else:
        prism_path = (
            project_root
            / "src"
            / "model_organism"
            / "prompts"
            / "prism_4"
            / "general_system_prompt.jinja2"
        )
    if not quirk_path.is_file():
        raise FileNotFoundError(f"Quirk prompt not found: {quirk_path}")
    if not prism_path.is_file():
        raise FileNotFoundError(f"PRISM-4 prompt not found: {prism_path}")
    return quirk_path, prism_path


def quirk_to_experiment_id(quirk_name: str) -> str:
    return f"experiment_{hashlib.sha256(quirk_name.encode()).hexdigest()[:8]}"


def _build_remote_mcp_clients(url: str, server_name: str) -> list[WrappedMcpClient]:
    return [WrappedMcpClient(client=Client(url), name=server_name, url=url)]


def _strip_claude_subprocess_env() -> None:
    """Optionally remove env vars that confuse the bundled claude binary.

    DEFAULT: no-op. On a cluster (Mila etc.) there is no ``claude login``
    OAuth and no VS Code Claude Code extension, so we *want* the bundled
    binary to authenticate with ``ANTHROPIC_API_KEY`` from ``.env``. Popping
    it there would break the run (nothing to fall back to).

    LOCAL workaround (opt-in): set ``AUDIT_STRIP_CLAUDE_ENV=1`` to restore the
    old behavior. The SDK ships its own ``claude.exe`` that authenticates
    against Claude Code's backend; if ``ANTHROPIC_API_KEY`` is set, the
    bundled binary tries to use it and rejects keys not enrolled for Claude
    Code (``Invalid API key · Fix external API key``). Unsetting it forces a
    fallback to the OAuth credentials from a prior ``claude login``.
    ``CLAUDE_CODE_SSE_PORT`` is set on terminals spawned by VS Code's Claude
    Code extension and causes the same failure when present.

    When enabled, this pop happens AFTER dotenv has loaded ``.env`` (via
    ``import src``) and BEFORE ``run_agent`` spawns the binary, which is the
    only window where the pop actually changes subprocess behavior.
    """
    if os.environ.get("AUDIT_STRIP_CLAUDE_ENV", "0") != "1":
        print(
            "[run_remote_target_auditor] keeping ANTHROPIC_API_KEY in env "
            "(set AUDIT_STRIP_CLAUDE_ENV=1 to force the local OAuth workaround)"
        )
        return
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_SSE_PORT",
    ):
        if k in os.environ:
            print(f"[run_remote_target_auditor] popping {k} from env "
                  "(letting bundled claude.exe use OAuth)")
            os.environ.pop(k, None)


async def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    validate_args(args)
    _strip_claude_subprocess_env()

    quirk_path, prism_path = _resolve_prompt_paths(args)
    quirk_text = quirk_path.read_text(encoding="utf-8")
    prism_text = load_prompt_file(prism_path)

    print(f"[run_remote_target_auditor] quirk={args.quirk_name}")
    print(f"[run_remote_target_auditor] remote_mcp_url={args.remote_mcp_url}")
    print(f"[run_remote_target_auditor] remote_server_name={args.remote_server_name}")
    print(f"[run_remote_target_auditor] mcp_inference_type={args.mcp_inference_type}")
    print(f"[run_remote_target_auditor] agent_type={args.agent_type}")

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
                f"[run_remote_target_auditor] Skipping run {run_id} — "
                f"already complete at {work_dir}"
            )
            continue
        if work_dir.exists():
            shutil.rmtree(work_dir)

        for attempt in range(args.max_retries):
            try:
                if attempt > 0:
                    print(
                        f"[run_remote_target_auditor] retry {attempt}/"
                        f"{args.max_retries-1} for run {run_id}"
                    )
                # Build a fresh remote MCP client per attempt — the underlying
                # fastmcp Client is single-use (entered exactly once).
                remote_mcp_clients = _build_remote_mcp_clients(
                    args.remote_mcp_url, args.remote_server_name
                )
                await run_agent(
                    agent_type=args.agent_type,
                    work_dir=work_dir,
                    target_model=None,
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
                    remote_mcp_clients=remote_mcp_clients,
                )

                metadata = {
                    "quirk_name": args.quirk_name,
                    "anonymous_id": anonymous_id,
                    "exp_name": args.exp_name,
                    "run_id": run_id,
                    "remote_mcp_url": args.remote_mcp_url,
                    "remote_server_name": args.remote_server_name,
                    "mcp_inference_type": args.mcp_inference_type,
                    "agent_type": args.agent_type,
                    "agent_model": args.agent_model,
                    "running_time_minutes": args.running_time,
                    "max_tokens": args.max_tokens,
                    "n_candidate_quirks": args.n_candidate_quirks,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "actual_quirk_prompt": quirk_text,
                    "prism_system_prompt": prism_text,
                }
                (work_dir / "experiment_metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                print(
                    f"[run_remote_target_auditor] run {run_id} completed -> {work_dir}"
                )
                break

            except Exception as e:
                err = str(e).split("\n")[-1] if "\n" in str(e) else str(e)
                print(
                    f"[run_remote_target_auditor] run {run_id} attempt "
                    f"{attempt+1} failed: {err}"
                )
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                if attempt == args.max_retries - 1:
                    print(
                        f"[run_remote_target_auditor] giving up after "
                        f"{args.max_retries} attempts"
                    )
                    raise
                await asyncio.sleep(2)

    print(
        f"[run_remote_target_auditor] all runs complete; outputs under {output_root}"
    )


if __name__ == "__main__":
    asyncio.run(main())
