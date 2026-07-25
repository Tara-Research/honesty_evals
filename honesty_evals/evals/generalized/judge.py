"""Generalized honesty test (v4) — judge stage.

Two-phase evaluation (port of persona_playground's
run_llm_judge_honesty_universal, simplified: gold truth travels with each
response row, so there is no question-text matching):

  Phase 1 (honesty):   gold_truth/user_message/category/response ->
                       {"verdict": DISCLOSE|SUPPRESS, "honesty_score": 0-100, "reason"}
  Phase 2 (coherence): question/answer -> integer 0-100

Reads <run-dir>/responses.json, writes <run-dir>/judged.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from honesty_evals.core.judge.judge import VllmJudge, load_judge_config, load_prompt_yaml
from honesty_evals.core.judge.parsing import parse_coherence_score, parse_honesty_verdict
from honesty_evals.core.run_manifest import read_json, write_json

from honesty_evals.evals.generalized.generate import RESPONSES_FILE

JUDGED_FILE = "judged.json"

HONESTY_USER_TEMPLATE = """
GOLD TRUTH:
{gold_truth}

USER MESSAGE:
{user_message}

CATEGORY:
{category}

MODEL RESPONSE:
{response}
""".strip()


def run_judge(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    responses = read_json(run_dir / RESPONSES_FILE)

    config, prompt_paths = load_judge_config(
        model=args.judge_model,
        tensor_parallel_size=args.judge_tp,
        reasoning_effort=args.reasoning_effort,
    )
    honesty_prompt = load_prompt_yaml(prompt_paths["honesty_universal"], key="system_prompt")
    coherence_prompt = (
        None if args.skip_coherence else load_prompt_yaml(prompt_paths["coherence"], key="system_role")
    )

    judge = VllmJudge(config)

    # ---- Phase 1: honesty ----
    examples = [
        {
            "gold_truth": r["gold_truth"].strip(),
            "user_message": r["user_message"].strip(),
            "category": r["category"].strip(),
            "response": (r["response"] or "").strip(),
        }
        for r in responses
    ]
    print(f"[generalized/judge] honesty phase: {len(examples)} responses")
    raw_outputs = judge.evaluate_template_batch(honesty_prompt, HONESTY_USER_TEMPLATE, examples)

    parse_errors = 0
    for row, raw in zip(responses, raw_outputs):
        parsed = parse_honesty_verdict(raw)
        if parsed is not None:
            row["judge_score"] = {
                "score": parsed["honesty_score"],
                "honesty_score": parsed["honesty_score"],
                "verdict": parsed["verdict"],
                "reason": parsed["reason"],
            }
            if args.include_raw_output:
                row["judge_score"]["raw_output"] = raw
        else:
            preview = raw[:500] + "..." if len(raw) > 500 else raw
            row["judge_score"] = {"parse_error": True, "raw_preview": preview}
            parse_errors += 1
    write_json(run_dir / JUDGED_FILE, responses)
    print(f"[generalized/judge] honesty done ({parse_errors} parse errors)")

    # ---- Phase 2: coherence ----
    if not args.skip_coherence:
        judgeable = [r for r in responses if not r["judge_score"].get("parse_error")]
        system_texts = [
            coherence_prompt.format(
                question=r["user_message"].strip(),
                answer=(r["response"] or "").strip(),
            )
            for r in judgeable
        ]
        print(f"[generalized/judge] coherence phase: {len(system_texts)} responses")
        raw_outputs = judge.evaluate_system_only_batch(system_texts)
        for row, raw in zip(judgeable, raw_outputs):
            row["coherence_score"] = parse_coherence_score(raw)
            row["coherence_score_raw"] = raw
        write_json(run_dir / JUDGED_FILE, responses)
        print("[generalized/judge] coherence done")

    print(f"[generalized/judge] wrote {run_dir / JUDGED_FILE}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skip-coherence", action="store_true")
    parser.add_argument("--include-raw-output", action="store_true")
