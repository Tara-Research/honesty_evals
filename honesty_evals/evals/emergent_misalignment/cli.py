"""CLI registration for the emergent-misalignment (EM) eval."""
from __future__ import annotations

import argparse

from honesty_evals.cli_common import (
    add_judge_args,
    add_sampling_args,
    add_server_args,
    add_steering_args,
)
from honesty_evals.evals.emergent_misalignment import fetch, generate, judge, metrics, plot


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "em",
        help="Emergent-misalignment eval: base vs EM LoRA vs EM LoRA + honesty "
        "steering, scored with the EM repo's aligned/coherent judges",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    p_fetch = stages.add_parser(
        "fetch", help="Download the EM question YAMLs into data/emergent_misalignment/"
    )
    p_fetch.set_defaults(func=fetch.run_fetch)

    p_gen = stages.add_parser(
        "generate", help="Generate responses per condition through the steering server"
    )
    generate.add_args(p_gen)
    add_steering_args(p_gen)
    add_server_args(p_gen)
    add_sampling_args(p_gen)
    p_gen.set_defaults(func=generate.run_generate)

    p_judge = stages.add_parser("judge", help="Score responses (aligned + coherent)")
    judge.add_args(p_judge)
    add_judge_args(p_judge)
    p_judge.set_defaults(func=judge.run_judge)

    p_metrics = stages.add_parser(
        "metrics", help="Aggregate judged scores into misalignment metrics"
    )
    metrics.add_args(p_metrics)
    p_metrics.set_defaults(func=metrics.run_metrics)

    p_plot = stages.add_parser("plot", help="Bar charts of the metrics across conditions")
    plot.add_args(p_plot)
    p_plot.set_defaults(func=plot.run_plot)
