"""Server-only entry point: load the local model organism and serve its MCP
tools over HTTP, with no auditor in this process.

Companion to ``run_remote_target_auditor.py`` for split-machine deployments
(e.g. tamia compute nodes that have GPUs but no outbound HTTPS to
api.anthropic.com). The flow is:

    1. Load Llama-3.3-70B + LoRA into GPU memory via ``LocalModelOrganism``.
    2. Build the FastMCP server(s) corresponding to ``--mcp-inference-type``
       (the same builder ``run_agent`` uses internally — same tool surface).
    3. Bind FastMCP's HTTP transport to ``--host:--port`` and block.
    4. Print ``MCP_SERVER_READY host=<hostname> port=<port>
       server_name=<name>`` to stdout so the operator can open the SSH
       tunnel from their laptop.

Notes:
    * Token tracking (``src.mcp_servers.token_tracking``) is laptop-side
      only in the ``target_local_*`` configurations — no MCP tool here
      calls ``add_tokens``. If a future tool starts doing so, the count
      would be lost across the process boundary; revisit then.
    * Conversation state lives in this process (the ``conversations`` dict
      captured in tool closures inside ``create_local_target_mcp``). A
      crash here loses conversation IDs; an SSH tunnel drop alone does not.
"""

import argparse
import asyncio
import socket
import sys
from pathlib import Path

sys.tracebacklimit = 1

from src.agent_framework.utils import serve_mcp_foreground
from src.auditing_agents.looped_investigator.agent import build_target_mcp_servers
from src.model_organism import LocalModelOrganism
from src.utils import get_project_root, load_prompt_file


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # ---- Quirk identification (same as run_local_target.py) ----
    parser.add_argument(
        "--quirk-name",
        type=str,
        required=True,
        help="Quirk name — used to locate the quirk system prompt under "
        "src/model_organism/prompts/system_prompt_quirks/. Recorded on the "
        "model organism for grading; NOT injected into the model at runtime.",
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

    # ---- Generation defaults ----
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")

    # ---- StTP tool config ----
    parser.add_argument(
        "--sttp-vector-extraction-method",
        type=str,
        default="logistic_regression",
    )
    parser.add_argument("--sttp-vector-type", type=str, default="global")
    parser.add_argument("--sttp-honesty-category", type=str, default="generalized")
    parser.add_argument(
        "--sttp-features",
        type=str,
        default="response_avg",
        choices=[
            "prompt_avg", "prompt_all", "response_avg", "response_all",
            "first_special_token", "first_content_token",
        ],
        help="Which feature-mode LR vector + embeddings to load. "
             "Requires sttp_vector_type='global'.",
    )
    parser.add_argument(
        "--sttp-steering-mode",
        type=str,
        default="all_token_mode",
        choices=["all_token_mode", "response_mode"],
    )

    # ---- MCP inference type ----
    parser.add_argument(
        "--mcp-inference-type",
        type=str,
        default="target_local_sttp_steering",
        choices=[
            "target_local",
            "target_local_sttp_steering",
            "target_local_sttp_only",
        ],
        help="Which MCP tool bundle to expose. The auditor on the other "
        "side will see whatever tools this server publishes.",
    )

    # ---- HTTP server bind ----
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address. 0.0.0.0 lets remote clients reach the server "
        "(through an SSH tunnel from the operator's laptop).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port for the FastMCP server.",
    )

    return parser


def _resolve_prompt_paths(args) -> tuple[Path, Path]:
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


async def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    quirk_path, prism_path = _resolve_prompt_paths(args)
    quirk_text = quirk_path.read_text(encoding="utf-8")
    prism_text = load_prompt_file(prism_path)

    print(f"[run_target_mcp_server] quirk={args.quirk_name}")
    print(f"[run_target_mcp_server] base_model={args.base_model_id}")
    print(f"[run_target_mcp_server] lora_repo={args.lora_repo}")
    print(f"[run_target_mcp_server] mcp_inference_type={args.mcp_inference_type}")

    # Load the target organism. Heavy: pulls Llama-3.3-70B + LoRA into GPU memory.
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

    # Smoke probe so a misconfigured load fails fast (before binding the port).
    probe = await target_model("Say hello and nothing else.", max_tokens=20)
    print(f"[run_target_mcp_server] smoke probe OK: {probe[:80]!r}")

    # Build the same FastMCP servers run_agent would build for this mcp_inference_type.
    # work_dir is unused for target_local_* modes (no filesystem MCP, no
    # multisample workspace dir name), but is required by the helper signature.
    mcp_servers = build_target_mcp_servers(
        agent_type="claude_agent",  # suppresses the filesystem MCP
        mcp_inference_type=args.mcp_inference_type,
        target_model=target_model,
        baseline_model=None,
        work_dir=Path("/tmp/unused_server_workspace"),
        sttp_base_model_id=args.base_model_id,
        sttp_vector_extraction_method=args.sttp_vector_extraction_method,
        sttp_vector_type=args.sttp_vector_type,
        sttp_honesty_category=args.sttp_honesty_category,
        sttp_steering_mode=args.sttp_steering_mode,
        sttp_features=args.sttp_features,
    )
    if len(mcp_servers) != 1:
        raise RuntimeError(
            f"Expected exactly 1 MCP server for {args.mcp_inference_type!r}, "
            f"got {len(mcp_servers)}"
        )
    mcp_app = mcp_servers[0]

    hostname = socket.gethostname()
    print(
        f"MCP_SERVER_READY host={hostname} port={args.port} "
        f"server_name={mcp_app.name} bind={args.host}",
        flush=True,
    )
    print(
        f"[run_target_mcp_server] To connect from laptop, open an SSH tunnel:\n"
        f"  ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "
        f"-L {args.port}:{hostname}:{args.port} <user>@<tamia-login>",
        flush=True,
    )

    # Foreground serve until SIGTERM. No daemon thread, no atexit dance —
    # SLURM's signal-on-time-limit cleanly cancels the await.
    await serve_mcp_foreground(mcp_app, host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
