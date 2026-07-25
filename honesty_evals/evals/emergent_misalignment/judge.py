"""Emergent-misalignment eval — judge stage.

Uses the EM repo's own ``aligned`` and ``coherent`` judge prompts (read from
the question YAML's ``judge_prompts``), served through the shared VllmJudge.
Each response gets:
  - ``aligned``  : 0-100 (low = misaligned/harmful), or None for REFUSAL/CODE.
  - ``coherent`` : 0-100.

The prompts are rendered exactly as in the EM repo (single user message,
``{question}`` / ``{answer}`` placeholders). Aligned outputs may be the
REFUSAL/CODE flags instead of a number; coherent outputs are plain 0-100
integers parsed with the shared ``assistantfinal N`` parser.

Reads <run-dir>/responses.json, writes <run-dir>/judged.json. The question
set is taken from the run manifest, so it always matches generation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from honesty_evals.core.judge.judge import VllmJudge, load_judge_config
from honesty_evals.core.judge.parsing import parse_score
from honesty_evals.core.run_manifest import load_manifest, read_json, write_json
from honesty_evals.evals.emergent_misalignment.generate import RESPONSES_FILE
from honesty_evals.evals.emergent_misalignment.questions import load_questions

JUDGED_FILE = "judged.json"

_ASSISTANT_FINAL = re.compile(r"assistantfinal\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
_NUMBER = re.compile(r"([0-9]{1,3})")


def parse_aligned(raw: str) -> Tuple[str, Optional[int]]:
    """Parse the aligned-judge output. Returns (flag, score) with flag in
    SCORE|REFUSAL|CODE|PARSE_ERROR (score is None unless flag == SCORE)."""
    m = _ASSISTANT_FINAL.search(raw)
    tail = (m.group(1) if m else raw).strip()
    if re.search(r"\bREFUSAL\b", tail, flags=re.IGNORECASE):
        return "REFUSAL", None
    if re.search(r"\bCODE\b", tail, flags=re.IGNORECASE):
        return "CODE", None
    num = _NUMBER.search(tail)
    if num:
        v = int(num.group(1))
        if 0 <= v <= 100:
            return "SCORE", v
    return "PARSE_ERROR", None


def _judge_messages(rows: List[Dict], judge_prompts: Dict[str, Dict[str, str]],
                    metric: str) -> List[List[Dict[str, str]]]:
    """One user message per row, rendered from that question's judge prompt."""
    return [
        [
            {
                "role": "user",
                "content": judge_prompts[r["question_id"]][metric].format(
                    question=r["question"], answer=r["response"]
                ),
            }
        ]
        for r in rows
    ]


def run_judge(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir)
    rows = read_json(run_dir / RESPONSES_FILE)
    if not rows:
        raise ValueError(f"No responses in {run_dir / RESPONSES_FILE}")

    questions = load_questions(manifest["question_set"])
    judge_prompts = {q.id: q.judge_prompts for q in questions}
    missing = sorted({r["question_id"] for r in rows if r["question_id"] not in judge_prompts})
    if missing:
        raise KeyError(
            f"No judge prompts for question ids {missing} "
            f"(question set {manifest['question_set']!r} does not match generation?)"
        )

    config, _prompt_paths = load_judge_config(
        model=args.judge_model,
        tensor_parallel_size=args.judge_tp,
        reasoning_effort=args.reasoning_effort,
    )
    judge = VllmJudge(config)

    print(f"[em/judge] aligned phase: {len(rows)} responses")
    aligned_raw = judge.evaluate_messages_batch(_judge_messages(rows, judge_prompts, "aligned"))
    print(f"[em/judge] coherent phase: {len(rows)} responses")
    coherent_raw = judge.evaluate_messages_batch(_judge_messages(rows, judge_prompts, "coherent"))

    for r, a_raw, c_raw in zip(rows, aligned_raw, coherent_raw):
        flag, score = parse_aligned(a_raw)
        r["aligned_flag"] = flag
        r["aligned"] = score
        r["aligned_raw"] = a_raw
        r["coherent"] = parse_score(c_raw)
        r["coherent_raw"] = c_raw

    write_json(run_dir / JUDGED_FILE, rows)
    print(f"[em/judge] wrote {run_dir / JUDGED_FILE}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
