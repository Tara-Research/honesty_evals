"""Emergent-misalignment eval — generation stage.

Conditions
----------
- ``base``        : base model, no LoRA, no steering (sanity reference).
- ``em_baseline`` : EM LoRA fine-tune, no steering (misaligned baseline).
- ``em_swfc``     : EM LoRA fine-tune + SwFC honesty steering (restoration).
- ``em_sttp``     : EM LoRA fine-tune + StTP honesty steering (restoration).
- ``em_stmp``     : EM LoRA fine-tune + StMP honesty steering (restoration).

Rewrite of persona_playground's
src/emergent_misalignment/generate_em_responses.py on top of the shared
serving/steering core. Instead of merge-and-unload HF LoRA loading, the LoRA
is served by the vLLM steering server as an adapter (registered under the name
"em"); each condition selects lora=None (base) or lora="em" plus the compiled
steering payload for the em_<method> arms. One steered method per run: the
CLI steering flags (--steering/--layer/--coefficient/...) define the vector
and must match the em_<method> conditions selected.

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
    build_spec_from_args,
    enable_thinking_value,
    sampling_dict,
    validate_server_args,
)
from honesty_evals.core.run_manifest import create_run_dir, write_json
from honesty_evals.core.serving.client import SteeredClient
from honesty_evals.core.serving.launch import launched_server
from honesty_evals.core.steering.compile import compile_steering
from honesty_evals.evals.emergent_misalignment.questions import Question, load_questions

CONDITIONS = ("base", "em_baseline", "em_swfc", "em_sttp", "em_stmp")
# Steered conditions -> the steering method they require.
STEERED_CONDITION_METHOD = {"em_swfc": "swfc", "em_sttp": "sttp", "em_stmp": "stmp"}
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
    """Cross-check --conditions against --lora-adapter and --steering."""
    em_conditions = [c for c in conditions if c != "base"]
    steered = [c for c in conditions if c in STEERED_CONDITION_METHOD]

    if em_conditions and args.lora_adapter is None:
        raise SystemExit(
            f"--lora-adapter is required for conditions {em_conditions} "
            "(they run on the EM LoRA fine-tune)"
        )
    if not em_conditions and args.lora_adapter is not None:
        raise SystemExit(
            "--lora-adapter must not be set when only the 'base' condition runs"
        )

    methods = sorted({STEERED_CONDITION_METHOD[c] for c in steered})
    if len(methods) > 1:
        raise SystemExit(
            f"Conditions {steered} require different steering methods {methods}; "
            "one steered method per run (the steering flags define a single vector)"
        )
    if steered and args.steering != methods[0]:
        raise SystemExit(
            f"--steering {args.steering} does not match condition "
            f"{steered[0]!r} (requires --steering {methods[0]})"
        )
    if not steered and args.steering != "none":
        raise SystemExit(
            f"--steering {args.steering} would be unused: no em_swfc/em_sttp/em_stmp "
            "condition selected (pass --steering none)"
        )


def build_prompts(questions: List[Question], seed: int) -> List[Tuple[Question, str]]:
    """Pick one seeded paraphrase per question (shared across conditions)."""
    rng = random.Random(seed)
    return [(q, rng.choice(q.paraphrases)) for q in questions]


def run_generate(args: argparse.Namespace) -> None:
    validate_server_args(args)
    conditions = _parse_conditions(args.conditions)
    _validate_condition_flags(args, conditions)
    spec = build_spec_from_args(args, model_id=args.model)

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
            "steering": spec.to_dict(),
            "sampling": sampling_dict(args, n=args.n_per_question),
            "enable_thinking": enable_thinking_value(args),
        },
    )

    compiled = compile_steering(spec, model_id=args.model, payload_dir=str(run_dir / "steering"))
    steering_payload = compiled.to_dict() if compiled else None

    def _condition_request(condition: str) -> Tuple[Optional[str], Optional[Dict]]:
        """(lora, steering) for one condition."""
        if condition == "base":
            return None, None
        if condition == "em_baseline":
            return EM_LORA_NAME, None
        return EM_LORA_NAME, steering_payload

    def generate_all(server_url: str) -> List[Dict]:
        client = SteeredClient(server_url)
        rows: List[Dict] = []
        for condition in conditions:
            lora, steering = _condition_request(condition)
            print(f"[em/generate] condition={condition} lora={lora} "
                  f"steering={'yes' if steering else 'no'}")
            completions = client.generate_batch(
                messages_batch,
                sampling=sampling_dict(args, n=args.n_per_question),
                steering=steering,
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
