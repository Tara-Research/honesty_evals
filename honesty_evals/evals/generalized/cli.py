"""CLI registration for the generalized honesty test (v4)."""
from __future__ import annotations

import argparse

from honesty_evals.cli_common import (
    add_judge_args,
    add_sampling_args,
    add_server_args,
)
from honesty_evals.evals.generalized import generate, judge, metrics


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generalized",
        help="Generalized honesty test: 112 scenarios, 8 categories, "
        "DISCLOSE/SUPPRESS + honesty/coherence scores",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    p_gen = stages.add_parser("generate", help="Generation over the scenario set")
    generate.add_args(p_gen)
    add_server_args(p_gen)
    add_sampling_args(p_gen)
    p_gen.set_defaults(func=generate.run_generate)

    p_judge = stages.add_parser("judge", help="Two-phase honesty + coherence judging")
    judge.add_args(p_judge)
    add_judge_args(p_judge)
    p_judge.set_defaults(func=judge.run_judge)

    p_metrics = stages.add_parser("metrics", help="Aggregate judged scores")
    metrics.add_args(p_metrics)
    p_metrics.set_defaults(func=metrics.run_metrics)
