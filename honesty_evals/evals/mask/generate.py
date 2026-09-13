"""MASK benchmark — generation stage.

Rewrite of persona_playground's src/mask/generate_responses.py on top of the
shared serving core (GenerationClient replaces direct HF model loading).

Semantics preserved from the original:

* Each row yields ``--lie-k`` independent "lying" generations from
  system_prompt + user_prompt (or the 4-message doubling-down chain for the
  doubling_down_known_facts archetype).
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
    enable_thinking_value,
    sampling_dict,
    validate_server_args,
)
from honesty_evals.core.run_manifest import create_run_dir
from honesty_evals.core.serving.client import GenerationClient
from honesty_evals.core.serving.launch import launched_server
from honesty_evals.evals.mask.filename_utils import KNOWN_ARCHETYPES
from honesty_evals.evals.mask.json_io import (
    build_metadata,
    raw_data_path,
    read_json_data,
    response_path,
    write_json_data,
)

DEFAULT_BELIEF_SYSTEM = "You are a helpful assistant."

# Two tag spellings, both real. Qwen3.x's chat template emits ``<think>``; the
# reward-hacking distill organism was trained on 1500/1500 assistant turns opening
# with ``<thinking>`` (its CodeContests SFT format). A regex written for one silently
# passes the other straight through to the judge — ``</think>`` is not a substring of
# ``</thinking>``, so even the unpaired fallback below misses it.
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>")
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>")


def _strip_think_tags(text: str) -> str:
    """Remove reasoning traces, whether or not the opening tag was generated.

    A paired-tag regex is not enough. With thinking left to the chat template,
    Qwen3.6 ends the PROMPT with a dangling ``<think>\n`` (chat_template.jinja:
    ``{%- else %}{{- '<think>\n' }}``), so the completion begins already inside
    the block and contains only the closing ``</think>``. The paired regex
    matched nothing and the judge silently received reasoning-trace text instead
    of the model's answer. Anything before the last closing tag is therefore
    reasoning and is dropped even with no opening tag.

    A response with no closing tag at all is returned unchanged: with
    ENABLE_THINKING=false there are no tags to strip, and for a thinking run it
    means the block never closed (truncated mid-reasoning, no answer produced).
    Those two cases are told apart by :func:`_trace_kind`, which the caller
    tallies into the output metadata.
    """
    text = _THINK_RE.sub("", text)
    last_close = None
    for last_close in _THINK_CLOSE_RE.finditer(text):
        pass
    if last_close is not None:
        text = text[last_close.end():]
    return text.strip()


def _trace_kind(text: str) -> str | None:
    """Classify a RAW completion's reasoning trace: 'closed', 'unclosed', or None.

    Called before stripping, because stripping destroys the evidence: only the
    stripped text is persisted, so a trace that is not counted here cannot be
    recovered from responses.json afterwards. 'unclosed' needs a visible opening
    tag — a completion that started inside a template-opened ``<think>`` and never
    closed is indistinguishable from plain prose, and is not guessed at.
    """
    if _THINK_CLOSE_RE.search(text):
        return "closed"
    if _THINK_OPEN_RE.search(text):
        return "unclosed"
    return None


def _strip_and_tally(text: str, tally: Dict[str, int]) -> str:
    """Strip the reasoning trace, recording what was there before it was removed."""
    tally["total"] += 1
    kind = _trace_kind(text)
    if kind is not None:
        tally[kind] += 1
    return _strip_think_tags(text)


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


BELIEF_GEN_PREFIX = "generation(belief_system_1 + belief_elicit_"


def _reference_file(from_dir: Path, archetype: str, stage: str) -> Path:
    files = sorted((from_dir / archetype / stage).glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"--beliefs-from: expected exactly one {stage} file for {archetype} under {from_dir}, "
            f"found {len(files)}"
        )
    return files[0]


def _copy_beliefs_from(df: pd.DataFrame, from_dir: Path, archetype: str) -> None:
    """Fill this run's belief-generation columns from a reference run, keyed by task_id."""
    _, ref = read_json_data(_reference_file(from_dir, archetype, "responses"))
    cols = [c for c in ref.columns if c.startswith(BELIEF_GEN_PREFIX)]
    if not cols:
        # provided_facts carries no belief elicitation in any run: nothing to copy, nothing to elicit
        print(f"[mask/generate] {archetype}: reference run has no belief columns -- none to copy")
        return
    ref = ref.set_index("task_id")
    missing = [t for t in df["task_id"] if t not in ref.index]
    if missing:
        raise ValueError(
            f"--beliefs-from: {len(missing)} task_ids of {archetype} absent from the reference run "
            f"(e.g. {missing[:3]})"
        )
    for c in cols:
        df[c] = [ref.at[t, c] for t in df["task_id"]]
    print(
        f"[mask/generate] {archetype}: {len(cols)} belief columns copied from {from_dir} "
        f"({len(df)} rows) -- no belief elicitation for this arm"
    )


def _belief_k(archetype: str) -> Dict[str, int]:
    return {"1": 3} if archetype == "statistics" else {"1": 3, "2": 1, "3": 1}


def _process_archetype(
    client: GenerationClient,
    archetype: str,
    args: argparse.Namespace,
    lora: str | None,
    output_path: Path,
    out_meta: Dict,
) -> None:
    _, df = read_json_data(raw_data_path(archetype, test=args.test))

    if "belief_system_1" not in df.columns and "belief_elicit_1" in df.columns:
        df["belief_system_1"] = DEFAULT_BELIEF_SYSTEM

    is_doubling_down = "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
    enable_thinking = enable_thinking_value(args)
    # Always all three keys, present even at zero, so the rate is comparable
    # across arms instead of being an absent field in the clean one.
    tally: Dict[str, int] = {"total": 0, "closed": 0, "unclosed": 0}

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
        lora=lora,
        enable_thinking=enable_thinking,
    )
    for idx, texts in zip(df.index, completions):
        if len(texts) != args.lie_k:
            raise RuntimeError(
                f"Expected {args.lie_k} completions per lying prompt, got {len(texts)}"
            )
        for col, text in zip(lie_cols, texts):
            df.at[idx, col] = _strip_and_tally(text, tally)

    # ---- Belief elicitation ----
    if getattr(args, "beliefs_from", None) and "belief_elicit_1" in df.columns:
        # Grid convention (2026-09-13): every arm is scored against the BASE model's beliefs.
        # Copy the belief generations from the reference run instead of eliciting them, so the
        # consistent-belief scenario set is identical across arms and the judge only has to
        # score the lying prompts (belief passes are 5 of the 6 per arm).
        _copy_beliefs_from(df, Path(args.beliefs_from), archetype)
    elif "belief_elicit_1" in df.columns:
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
                f"prompts x {k_value} runs"
            )
            # Beliefs come from the same (possibly LoRA-adapted) model.
            completions = client.generate_batch(
                belief_batch,
                sampling=sampling_dict(args, n=k_value),
                lora=lora,
                enable_thinking=enable_thinking,
            )
            for idx, texts in zip(item_rows, completions):
                if len(texts) != k_value:
                    raise RuntimeError(
                        f"Expected {k_value} completions per belief prompt, got {len(texts)}"
                    )
                for col, text in zip(gen_cols, texts):
                    df.at[idx, col] = _strip_and_tally(text, tally)

    out_meta["reasoning_traces"] = dict(tally)
    print(
        f"[mask/generate] {archetype}: reasoning traces "
        f"{tally['closed']} closed / {tally['unclosed']} unclosed "
        f"of {tally['total']} completions"
    )
    write_json_data(output_path, out_meta, df)
    print(f"[mask/generate] wrote {output_path}")


def run_generate(args: argparse.Namespace) -> None:
    validate_server_args(args)

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
            "archetypes": archetypes,
            "lie_k": args.lie_k,
            "test": args.test,
            "sampling": sampling_dict(args),
            "enable_thinking": enable_thinking_value(args),
            "beliefs_from": args.beliefs_from,
        },
    )

    modelname = args.model.split("/")[-1]

    # LoRA convention (same as auditbench): the server-side adapter name IS the
    # repo/path string, so an external server must be launched with
    # --lora <adapter>=<adapter>. Unknown names fail loudly server-side (400).
    lora = args.lora_adapter

    def process_all(server_url: str) -> None:
        client = GenerationClient(server_url)
        for archetype in archetypes:
            output_path = response_path(run_dir, archetype, modelname)
            out_meta = build_metadata(
                archetype=archetype,
                model=modelname,
                lie_k=args.lie_k,
                run_name=args.run_name,
            )
            _process_archetype(client, archetype, args, lora, output_path, out_meta)

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
        default=1,
        help="Number of lying runs K per row (K independent samples per pressure "
        "prompt). Default: 1",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use the 5-row test subset (data/mask/raw_test/) instead of the full benchmark",
    )
    parser.add_argument(
        "--beliefs-from",
        default=None,
        help="Run dir of a reference (base-model) run whose belief generations are copied by "
        "task_id instead of being elicited; the judge stage then copies their evaluations too. "
        "Grid convention: every arm scored against the base model's beliefs.",
    )
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="HF repo or local path of a LoRA adapter served on top of --model. "
        "Applies to ALL generations in the run (beliefs and lying runs alike). "
        "With --server-url the server must have been started with "
        "--lora <adapter>=<adapter>.",
    )
