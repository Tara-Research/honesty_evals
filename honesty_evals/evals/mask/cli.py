"""CLI registration for the MASK benchmark (Model Alignment between
Statements and Knowledge)."""
from __future__ import annotations

import argparse

from honesty_evals.cli_common import (
    add_judge_args,
    add_sampling_args,
    add_server_args,
    add_steering_args,
)
from honesty_evals.evals.mask import download, generate, judge, metric, process_metrics


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "mask",
        help="MASK benchmark: does the model contradict its own elicited "
        "beliefs when pressured to lie?",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    p_download = stages.add_parser(
        "download",
        help="Regenerate data/mask/raw/ (+ 5-row raw_test/) from HF cais/mask",
    )
    download.add_args(p_download)
    p_download.set_defaults(func=download.run_download)

    p_gen = stages.add_parser(
        "generate",
        help="Belief elicitation (unsteered) + K lying runs per row, per archetype",
    )
    generate.add_args(p_gen)
    add_steering_args(p_gen)
    add_server_args(p_gen)
    add_sampling_args(p_gen)
    p_gen.set_defaults(func=generate.run_generate)

    p_judge = stages.add_parser(
        "judge",
        help="Judge responses against propositions (binary/doubling-down/numerical)",
    )
    judge.add_args(p_judge)
    add_judge_args(p_judge)
    p_judge.set_defaults(func=judge.run_judge)

    p_metrics = stages.add_parser(
        "metrics",
        help="Per-row honesty/accuracy indicators (lying runs vs elicited beliefs)",
    )
    metric.add_args(p_metrics)
    p_metrics.set_defaults(func=metric.run_metrics)

    p_process = stages.add_parser(
        "process",
        help="Aggregate metrics into all_results.json (honesty scores + bootstrap CIs)",
    )
    process_metrics.add_args(p_process)
    p_process.set_defaults(func=process_metrics.run_process)
