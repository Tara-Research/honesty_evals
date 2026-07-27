"""Emergent-misalignment eval — generation stage.

Conditions
----------
- ``base``        : base model, no LoRA (sanity reference).
- ``em_baseline`` : EM LoRA fine-tune (misaligned baseline).

Rewrite of persona_playground's
src/emergent_misalignment/generate_em_responses.py on top of the shared
serving core. Instead of merge-and-unload HF LoRA loading, the LoRA is served
by the vLLM generation server as an adapter (registered under the name "em");
each condition selects lora=None (base) or lora="em".

The same seeded per-question paraphrase is reused across all conditions, so
the comparison is paired; N samples per question come from vLLM's sampling
``n`` in a single request per question. Generation params mirror the EM repo
(temperature 1.0, top_p 1.0, max_tokens 600) via the required sampling flags.
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Tuple

from honesty_evals.cli_common import (
    enable_thinking_value,
    sampling_dict,
    validate_server_args,
)
from honesty_evals.core.run_manifest import create_run_dir, write_json
from honesty_evals.core.serving.client import GenerationClient
from honesty_evals.core.serving.launch import launched_server
from honesty_evals.evals.emergent_misalignment.questions import Question, load_questions

CONDITIONS = ("base", "em_baseline")
# Name under which the EM LoRA adapter is registered on the server.
EM_LORA_NAME = "em"

RESPONSES_FILE = "responses.json"


def _parse_conditions(raw: str) -> List[str]:
    conditions = [c.strip() for c in raw.split(",") if c.strip()]
    if not conditions:
        raise SystemExit(f"--conditions is empty; valid: {CONDITIONS}")
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"Unknown condition {c!r}; valid: {CONDITIONS}")
    if len(set(conditions)) != len(conditions):
        raise SystemExit(f"--conditions contains duplicates: {raw!r}")
    return conditions


def _validate_condition_flags(args: argparse.Namespace, conditions: List[str]) -> None:
    """Cross-check --conditions against --lora-adapter."""
    em_conditions = [c for c in conditions if c != "base"]

    if em_conditions and args.lora_adapter is None:
        raise SystemExit(
            f"--lora-adapter is required for conditions {em_conditions} "
            "(they run on the EM LoRA fine-tune)"
        )
    if not em_conditions and args.lora_adapter is not None:
        raise SystemExit(
            "--lora-adapter must not be set when only the 'base' condition runs"
        )


def build_prompts(questions: List[Question], seed: int) -> List[Tuple[Question, str]]:
    """Pick one seeded paraphrase per question (shared across conditions)."""
    rng = random.Random(seed)
    return [(q, rng.choice(q.paraphrases)) for q in questions]


def run_generate(args: argparse.Namespace) -> None:
    validate_server_args(args)
    conditions = _parse_conditions(args.conditions)
    _validate_condition_flags(args, conditions)

    questions = load_questions(args.question_set)
    prompts = build_prompts(questions, args.seed)
    messages_batch = [q.build_conversation(text) for q, text in prompts]
    print(
        f"[em/generate] {len(questions)} questions x {args.n_per_question} samples "
        f"x {len(conditions)} conditions ({args.question_set})"
    )

    run_dir = create_run_dir(
        eval_name="emergent_misalignment",
        run_name=args.run_name,
        manifest={
            "model": args.model,
            "lora_adapter": args.lora_adapter,
            "conditions": conditions,
            "question_set": args.question_set,
            "n_per_question": args.n_per_question,
            "n_questions": len(questions),
            "sampling": sampling_dict(args, n=args.n_per_question),
            "enable_thinking": enable_thinking_value(args),
        },
    )

    def _condition_request(condition: str) -> Optional[str]:
        """LoRA selection for one condition."""
        return None if condition == "base" else EM_LORA_NAME

    def generate_all(server_url: str) -> List[Dict]:
        client = GenerationClient(server_url)
        rows: List[Dict] = []
        for condition in conditions:
            lora = _condition_request(condition)
            print(f"[em/generate] condition={condition} lora={lora}")
            completions = client.generate_batch(
                messages_batch,
                sampling=sampling_dict(args, n=args.n_per_question),
                lora=lora,
                enable_thinking=enable_thinking_value(args),
            )
            for (q, text), texts in zip(prompts, completions):
                if len(texts) != args.n_per_question:
                    raise RuntimeError(
                        f"Expected {args.n_per_question} completions for question "
                        f"{q.id!r} ({condition}), got {len(texts)}"
                    )
                for response in texts:
                    rows.append(
                        {
                            "condition": condition,
                            "question_id": q.id,
                            "question": text,
                            "response": response,
                        }
                    )
        return rows

    if args.launch_server:
        loras = {EM_LORA_NAME: args.lora_adapter} if args.lora_adapter else None
        with launched_server(
            model=args.model,
            tp=args.tp,
            port=args.port,
            loras=loras,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as url:
            rows = generate_all(url)
    else:
        rows = generate_all(args.server_url)

    write_json(run_dir / RESPONSES_FILE, rows)
    print(f"[em/generate] wrote {len(rows)} responses to {run_dir / RESPONSES_FILE}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF model id (base model)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--conditions",
        required=True,
        help=f"Comma-separated subset of {','.join(CONDITIONS)}",
    )
    parser.add_argument(
        "--question-set",
        required=True,
        help="EM question set name or path (first_plot_questions | preregistered_evals)",
    )
    parser.add_argument(
        "--n-per-question",
        type=int,
        required=True,
        help="Samples per question (vLLM sampling n, one request per question)",
    )
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="HF repo/path of the EM (risky-financial) LoRA adapter. "
        "Required iff any em_* condition is selected.",
    )
