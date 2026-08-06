"""CLI registration for Impossible-SWEbench."""
from __future__ import annotations

import argparse

from honesty_evals.evals.impossible_swebench import run


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "impossible-swebench",
        help="Evaluate cheating on impossible SWE-bench tasks",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    validate = stages.add_parser("validate", help="Build and validate a task")
    run.add_task_args(validate)
    validate.set_defaults(func=run.run_validate)

    execute = stages.add_parser("run", help="Run against an external vLLM server")
    run.add_task_args(execute)
    execute.add_argument("--model", required=True, help="Base Hugging Face model")
    execute.add_argument("--lora-adapter", help="LoRA path, also used as its served name")
    execute.add_argument("--server-url", required=True, help="OpenAI-compatible /v1 URL")
    execute.add_argument("--run-name", required=True)
    execute.add_argument("--image-archive", type=run.existing_path, required=True)
    execute.add_argument("--runtime-dir", type=run.path_arg)
    execute.add_argument("--resume-log", type=run.existing_path)
    execute.set_defaults(func=run.run_eval)

    summary = stages.add_parser("summary", help="Summarize Inspect logs")
    summary.add_argument("--run-dir", type=run.existing_path, required=True)
    summary.set_defaults(func=run.run_summary)

    archive = stages.add_parser(
        "build-image-archive", help="Pull and archive all images for a task"
    )
    archive.add_argument("--task", default="impossible50")
    archive.add_argument("--output", type=run.path_arg, required=True)
    archive.add_argument("--retry-seconds", type=run.positive_int, default=60)
    archive.set_defaults(func=run.run_build_archive)
