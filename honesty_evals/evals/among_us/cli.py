"""CLI registration for the Among Us eval."""
from __future__ import annotations

import argparse

from honesty_evals.cli_common import (
    add_judge_args,
    add_sampling_args,
    add_server_args,
)
from honesty_evals.evals.among_us import judge, metrics, run


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "among-us",
        help="Among Us social-deduction sandbox: N games through the generation "
        "server, per-timestep deception judging, impostor win rate",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    p_run = stages.add_parser("run", help="Run N games concurrently")
    run.add_args(p_run)
    add_server_args(p_run)
    add_sampling_args(p_run)
    p_run.set_defaults(func=run.run_run)

    p_judge = stages.add_parser(
        "judge", help="Score every game timestep on Awareness/Lying/Deception/Planning"
    )
    judge.add_args(p_judge)
    add_judge_args(p_judge)
    p_judge.set_defaults(func=judge.run_judge)

    p_metrics = stages.add_parser(
        "metrics", help="Impostor win rate + aggregated judge scores"
    )
    metrics.add_args(p_metrics)
    p_metrics.set_defaults(func=metrics.run_metrics)
