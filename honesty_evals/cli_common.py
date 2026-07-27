"""Shared CLI argument groups for eval generate/run stages.

Every eval uses the same serving / sampling flag conventions, added via these
helpers so the flag surface cannot drift between evals.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional


# ---------- serving ----------

def add_server_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("serving")
    g.add_argument("--server-url", help="URL of a running generation server")
    g.add_argument(
        "--launch-server",
        action="store_true",
        help="Spawn the generation server as a subprocess for this stage",
    )
    g.add_argument("--tp", type=int, help="Tensor parallel size (required with --launch-server)")
    g.add_argument("--port", type=int, default=8192, help="Port for --launch-server (default 8192)")
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)


def validate_server_args(args: argparse.Namespace) -> None:
    if (args.server_url is None) == (not args.launch_server):
        raise SystemExit("Pass exactly one of --server-url or --launch-server")
    if args.launch_server and args.tp is None:
        raise SystemExit("--tp is required with --launch-server")


# ---------- sampling ----------

def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("sampling")
    g.add_argument("--max-new-tokens", type=int, required=True)
    g.add_argument("--temperature", type=float, required=True)
    g.add_argument("--top-p", type=float, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument(
        "--enable-thinking",
        choices=["true", "false", "default"],
        default="default",
        help="Chat-template thinking flag for models that support it "
        "(default: leave to the model's chat template)",
    )


def sampling_dict(args: argparse.Namespace, n: int = 1) -> Dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": n,
        "seed": args.seed,
    }


def enable_thinking_value(args: argparse.Namespace) -> Optional[bool]:
    return {"true": True, "false": False, "default": None}[args.enable_thinking]


# ---------- judge overrides ----------

def add_judge_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("judge")
    g.add_argument("--judge-model", default=None, help="Override judge model from configs/judge.yaml")
    g.add_argument("--judge-tp", type=int, default=None, help="Override judge tensor parallel size")
    g.add_argument("--reasoning-effort", default=None, help="Override judge reasoning effort")
