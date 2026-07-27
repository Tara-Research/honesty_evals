"""MASK benchmark — generation stage.

Rewrite of persona_playground's src/mask/generate_responses.py on top of the
shared serving/steering core (SteeringSpec + compile_steering + SteeredClient
replace direct HF model loading and steerer context managers).

Semantics preserved from the original:

* Each row yields ``--lie-k`` independent "lying" generations from
  system_prompt + user_prompt (or the 4-message doubling-down chain for the
  doubling_down_known_facts archetype). These runs ARE steered when the run
  is steered.
* Belief-elicitation generations run ONLY for unsteered runs — steered runs
  reuse the unsteered baseline's beliefs at the metrics stage (see
  metric.py's --baseline-run-dir). Belief runs are always generated with
  steering=None.
* belief_elicit_1 gets 3 runs; belief_elicit_2/3 get 1 run each (the
  statistics archetype only elicits belief_elicit_1).
* ``<think>...</think>`` reasoning traces are stripped from responses.

Deviation from the original: instead of one request per (row, run) pair, the
K runs of a prompt are requested as n=K samples of a single request. The
shared sampling args carry ONE seed for the whole stage, so per-run requests
with identical messages would sample identical completions; n=K keeps the K
runs independent draws.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from honesty_evals.cli_common import (
    build_spec_from_args,
    enable_thinking_value,
    sampling_dict,
    validate_server_args,
)
from honesty_evals.core.run_manifest import create_run_dir
from honesty_evals.core.serving.client import SteeredClient
from honesty_evals.core.serving.launch import launched_server
from honesty_evals.core.steering.compile import compile_steering
from honesty_evals.core.steering.spec import SteeringSpec
from honesty_evals.evals.mask.filename_utils import KNOWN_ARCHETYPES
from honesty_evals.evals.mask.json_io import (
    build_metadata,
    raw_data_path,
    read_json_data,
    response_path,
    write_json_data,
)

DEFAULT_BELIEF_SYSTEM = "You are a helpful assistant."

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 reasoning traces)."""
    return _THINK_RE.sub("", text).strip()


def _lie_messages(row: pd.Series, is_doubling_down: bool) -> List[Dict[str, str]]:
    if is_doubling_down:
        return [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["dd_turn_1_prompt"]},
            {"role": "assistant", "content": row["dd_turn_1_response"]},
            {"role": "user", "content": row["user_prompt"]},
        ]
    return [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["user_prompt"]},
    ]


def _belief_k(archetype: str) -> Dict[str, int]:
    return {"1": 3} if archetype == "statistics" else {"1": 3, "2": 1, "3": 1}


def _process_archetype(
    client: SteeredClient,
    archetype: str,
    args: argparse.Namespace,
    spec: SteeringSpec,
    steering_payload: Dict | None,
    lora: str | None,
    output_path: Path,
    out_meta: Dict,
) -> None:
    _, df = read_json_data(raw_data_path(archetype, test=args.test))

    if "belief_system_1" not in df.columns and "belief_elicit_1" in df.columns:
        df["belief_system_1"] = DEFAULT_BELIEF_SYSTEM

    is_doubling_down = "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
    enable_thinking = enable_thinking_value(args)

    # ---- Lying generations: one request per row, n = lie_k samples ----
    lie_cols = [
        f"generation(System Prompt + User Prompt)_run{run + 1}" for run in range(args.lie_k)
    ]
    for col in lie_cols:
        df[col] = None

    messages_batch = [_lie_messages(df.loc[idx], is_doubling_down) for idx in df.index]
    print(
        f"[mask/generate] {archetype}: {len(messages_batch)} lying prompts "
        f"x {args.lie_k} runs (doubling_down={is_doubling_down})"
    )
    completions = client.generate_batch(
        messages_batch,
        sampling=sampling_dict(args, n=args.lie_k),
        steering=steering_payload,
        lora=lora,
        enable_thinking=enable_thinking,
    )
    for idx, texts in zip(df.index, completions):
        if len(texts) != args.lie_k:
            raise RuntimeError(
                f"Expected {args.lie_k} completions per lying prompt, got {len(texts)}"
            )
        for col, text in zip(lie_cols, texts):
            df.at[idx, col] = _strip_think_tags(text)

    # ---- Belief elicitation: unsteered runs only, never steered ----
    # Steered runs reuse the baseline run's beliefs at the metrics stage.
    if not spec.is_steered and "belief_elicit_1" in df.columns:
        for be_num, k_value in _belief_k(archetype).items():
            belief_col = f"belief_elicit_{be_num}"
            if belief_col not in df.columns:
                continue
            gen_cols = [
                f"generation(belief_system_1 + belief_elicit_{be_num})_run{run + 1}"
                for run in range(k_value)
            ]
            for col in gen_cols:
                df[col] = None

            item_rows = [idx for idx in df.index if pd.notna(df.at[idx, belief_col])]
            if not item_rows:
                continue
            belief_batch = [
                [
                    {"role": "system", "content": df.at[idx, "belief_system_1"]},
                    {"role": "user", "content": df.at[idx, belief_col]},
                ]
                for idx in item_rows
            ]
            print(
                f"[mask/generate] {archetype}: {len(belief_batch)} belief_elicit_{be_num} "
                f"prompts x {k_value} runs (unsteered)"
            )
            # Beliefs come from the same (possibly LoRA-adapted) model — only
            # steering is withheld from belief elicitation.
            completions = client.generate_batch(
                belief_batch,
                sampling=sampling_dict(args, n=k_value),
                steering=None,
                lora=lora,
                enable_thinking=enable_thinking,
            )
            for idx, texts in zip(item_rows, completions):
                if len(texts) != k_value:
                    raise RuntimeError(
                        f"Expected {k_value} completions per belief prompt, got {len(texts)}"
                    )
                for col, text in zip(gen_cols, texts):
                    df.at[idx, col] = _strip_think_tags(text)

    write_json_data(output_path, out_meta, df)
    print(f"[mask/generate] wrote {output_path}")


def run_generate(args: argparse.Namespace) -> None:
    validate_server_args(args)
    spec = build_spec_from_args(args, model_id=args.model)

    if args.archetypes.strip() == "all":
        archetypes = sorted(KNOWN_ARCHETYPES)
    else:
        archetypes = [a.strip() for a in args.archetypes.split(",") if a.strip()]
        if not archetypes:
            raise SystemExit("--archetypes must name at least one archetype")
        unknown = set(archetypes) - set(KNOWN_ARCHETYPES)
        if unknown:
            raise SystemExit(
                f"Unknown archetypes {sorted(unknown)}; known: {sorted(KNOWN_ARCHETYPES)}"
            )
    for archetype in archetypes:
        path = raw_data_path(archetype, test=args.test)
        if not path.exists():
            raise FileNotFoundError(
                f"Raw MASK data not found: {path}. Run `honesty-evals mask download` first."
            )

    run_dir = create_run_dir(
        eval_name="mask",
        run_name=args.run_name,
        manifest={
            "model": args.model,
            "lora_adapter": args.lora_adapter,
            "steering": spec.to_dict(),
            "archetypes": archetypes,
            "lie_k": args.lie_k,
            "test": args.test,
            "sampling": sampling_dict(args),
            "enable_thinking": enable_thinking_value(args),
        },
    )

    compiled = compile_steering(spec, model_id=args.model, payload_dir=str(run_dir / "steering"))
    steering_payload = compiled.to_dict() if compiled else None

    modelname = args.model.split("/")[-1]
    file_technique = spec.method if spec.is_steered else None
    file_coefficient = spec.coefficient if spec.is_steered else None

    # LoRA convention (same as auditbench): the server-side adapter name IS the
    # repo/path string, so an external server must be launched with
    # --lora <adapter>=<adapter>. Unknown names fail loudly server-side (400).
    lora = args.lora_adapter

    def process_all(server_url: str) -> None:
        client = SteeredClient(server_url)
        for archetype in archetypes:
            output_path = response_path(
                run_dir, archetype, modelname, file_technique, file_coefficient
            )
            out_meta = build_metadata(
                archetype=archetype,
                model=modelname,
                technique=spec.method if spec.is_steered else "no_steering",
                coefficient=file_coefficient,
                layer=spec.layer,
                steering_mode=spec.steering_mode,
                vector_type=spec.vector_type,
                extraction_method=spec.extraction_method,
                lie_k=args.lie_k,
                run_name=args.run_name,
            )
            _process_archetype(
                client, archetype, args, spec, steering_payload, lora, output_path, out_meta
            )

    if args.launch_server:
        with launched_server(
            model=args.model,
            tp=args.tp,
            port=args.port,
            loras={lora: lora} if lora else None,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as url:
            process_all(url)
    else:
        process_all(args.server_url)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--archetypes",
        required=True,
        help="Comma-separated MASK archetypes to run, or 'all' "
        f"(known: {', '.join(sorted(KNOWN_ARCHETYPES))})",
    )
    parser.add_argument(
        "--lie-k",
        type=int,
        required=True,
        help="Number of lying runs K per row (K independent samples per pressure prompt)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use the 5-row test subset (data/mask/raw_test/) instead of the full benchmark",
    )
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="HF repo or local path of a LoRA adapter served on top of --model. "
        "Applies to ALL generations in the run (beliefs and lying runs alike). "
        "With --server-url the server must have been started with "
        "--lora <adapter>=<adapter>.",
    )
