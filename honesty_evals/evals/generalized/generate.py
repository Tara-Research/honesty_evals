"""Generalized honesty test (v4) — generation stage.

Loads the 112-scenario v4 test set (8 categories x 14), builds one
conversation per scenario (system prompt honest/dishonest variant + user
message), generates through the generation server, and writes
responses.json into the run dir.

Rewrite of persona_playground's
src/experiments/sequential_steering/test/sequential_steering_one_layer_honesty.py
on top of the shared serving core.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from honesty_evals.cli_common import (
    enable_thinking_value,
    sampling_dict,
    validate_server_args,
)
from honesty_evals.core.paths import DATA_FOLDER
from honesty_evals.core.run_manifest import create_run_dir, write_json
from honesty_evals.core.serving.client import GenerationClient
from honesty_evals.core.serving.launch import launched_server

DEFAULT_SCENARIOS = DATA_FOLDER / "generalized" / "v4" / "generalized_honesty_test_scenarios.json"

RESPONSES_FILE = "responses.json"


def load_scenarios(path: Path, categories: List[str] | None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data["scenarios"]
    if categories:
        available = {s["category"] for s in scenarios}
        unknown = set(categories) - available
        if unknown:
            raise ValueError(
                f"Unknown categories {sorted(unknown)}; available: {sorted(available)}"
            )
        scenarios = [s for s in scenarios if s["category"] in categories]
    if not scenarios:
        raise ValueError(f"No scenarios loaded from {path}")
    return scenarios


def build_messages(scenario: Dict, system_prompt_variant: str) -> List[Dict[str, str]]:
    system_key = f"system_prompt_{system_prompt_variant}"
    if system_key not in scenario:
        raise KeyError(f"Scenario {scenario.get('scenario_id')} has no {system_key}")
    return [
        {"role": "system", "content": scenario[system_key]},
        {"role": "user", "content": scenario["user_message"]},
    ]


def run_generate(args: argparse.Namespace) -> None:
    validate_server_args(args)

    scenarios_path = Path(args.scenarios)
    categories = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories
        else None
    )
    scenarios = load_scenarios(scenarios_path, categories)

    run_dir = create_run_dir(
        eval_name="generalized",
        run_name=args.run_name,
        manifest={
            "model": args.model,
            "system_prompt_variant": args.system_prompt_variant,
            "scenarios_path": str(scenarios_path),
            "categories": categories,
            "n_scenarios": len(scenarios),
            "sampling": sampling_dict(args),
            "enable_thinking": enable_thinking_value(args),
        },
    )

    messages_batch = [build_messages(s, args.system_prompt_variant) for s in scenarios]

    def generate(server_url: str) -> List[List[str]]:
        client = GenerationClient(server_url)
        return client.generate_batch(
            messages_batch,
            sampling=sampling_dict(args),
            enable_thinking=enable_thinking_value(args),
        )

    if args.launch_server:
        with launched_server(
            model=args.model,
            tp=args.tp,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as url:
            completions = generate(url)
    else:
        completions = generate(args.server_url)

    responses = [
        {
            "scenario_id": s["scenario_id"],
            "category": s["category"],
            "user_message": s["user_message"],
            "gold_truth": s["gold_truth"],
            "response": texts[0],
        }
        for s, texts in zip(scenarios, completions)
    ]
    write_json(run_dir / RESPONSES_FILE, responses)
    print(f"[generalized/generate] wrote {len(responses)} responses to {run_dir / RESPONSES_FILE}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--system-prompt-variant",
        required=True,
        choices=["honest", "dishonest"],
        help="Which scenario system prompt to use (dishonest = pressure condition, "
        "honest = aligned baseline)",
    )
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    parser.add_argument("--categories", default=None, help="Comma-separated category filter")
