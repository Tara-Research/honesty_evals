"""honesty-evals: unified CLI for all honesty evaluations.

    honesty-evals serve      --model ... --tp ... --port ...
    honesty-evals <eval> <stage> [flags]

Evals: generalized, mask, em, among-us, auditbench.
Each eval package contributes its subcommand tree via register_cli().
"""
from __future__ import annotations

import argparse


def _register_serve(subparsers: argparse._SubParsersAction) -> None:
    from honesty_evals.core.serving import server as serving_server

    parser = subparsers.add_parser("serve", help="Run the vLLM generation server")
    serving_server.add_serve_args(parser)
    parser.set_defaults(func=serving_server.run_serve)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="honesty-evals",
        description="Honesty evaluations for LLMs (one serving stack, five evals)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_serve(subparsers)

    from honesty_evals.evals.among_us.cli import register_cli as register_among_us
    from honesty_evals.evals.auditbench.cli import register_cli as register_auditbench
    from honesty_evals.evals.emergent_misalignment.cli import register_cli as register_em
    from honesty_evals.evals.generalized.cli import register_cli as register_generalized
    from honesty_evals.evals.mask.cli import register_cli as register_mask

    register_generalized(subparsers)
    register_mask(subparsers)
    register_em(subparsers)
    register_among_us(subparsers)
    register_auditbench(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
