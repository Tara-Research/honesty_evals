"""honesty-evals: unified CLI for all honesty evaluations.

    honesty-evals serve      --model ... --tp ... --port ...
    honesty-evals resolve    --model ... --steering ...
    honesty-evals <eval> <stage> [flags]

Evals: generalized, mask, em, among-us, auditbench.
Each eval package contributes its subcommand tree via register_cli().
"""
from __future__ import annotations

import argparse
import json


def _register_serve(subparsers: argparse._SubParsersAction) -> None:
    from honesty_evals.core.serving import server as serving_server

    parser = subparsers.add_parser("serve", help="Run the steering-enabled vLLM server")
    serving_server.add_serve_args(parser)
    parser.set_defaults(func=serving_server.run_serve)


def _register_resolve(subparsers: argparse._SubParsersAction) -> None:
    from honesty_evals.cli_common import add_steering_args, build_spec_from_args

    parser = subparsers.add_parser(
        "resolve",
        help="Resolve steering artifact paths + tuned layer/coefficient (no GPU)",
    )
    parser.add_argument("--model", required=True)
    add_steering_args(parser)

    def run_resolve(args: argparse.Namespace) -> None:
        from honesty_evals.core.paths import PERSONA_VECTORS_FOLDER
        from honesty_evals.core.steering.resolver import resolve_artifact_paths

        spec = build_spec_from_args(args, model_id=args.model)
        out = {"spec": spec.to_dict(), "persona_vectors_folder": PERSONA_VECTORS_FOLDER}
        if spec.is_steered:
            out["artifacts"] = resolve_artifact_paths(
                model_id=args.model,
                vector_type=spec.vector_type,
                honesty_category=spec.honesty_category,
                vector_extraction_method=spec.extraction_method,
                features=spec.features,
            )
        print(json.dumps(out, indent=2))

    parser.set_defaults(func=run_resolve)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="honesty-evals",
        description="Honesty evaluations for steered LLMs (one steering stack, five evals)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_serve(subparsers)
    _register_resolve(subparsers)

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
