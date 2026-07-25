"""Loader for the Emergent-Misalignment evaluation questions.

Mirrors the ``Question`` abstraction from the EM repo
(``open_models/eval.py``) so generation and judging consume the same YAML:
each entry has an ``id``, a list of ``paraphrases``, an optional ``system``
prompt, a ``temperature``, and a ``judge_prompts`` dict that holds the
``aligned`` and ``coherent`` judge prompt templates (with ``{question}`` /
``{answer}`` placeholders).

Source YAMLs (vendored into ``data/emergent_misalignment/``):
  https://github.com/emergent-misalignment/emergent-misalignment/tree/main/evaluation
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from honesty_evals.core.paths import DATA_FOLDER

EM_QUESTIONS_DIR = os.path.join(DATA_FOLDER, "emergent_misalignment")

# Entries whose id ends with these suffixes constrain the answer format
# (JSON / templated code). We exclude them by default so the misalignment
# eval runs on the plain free-form questions only.
_FORMAT_SUFFIXES = ("_json", "_template")


@dataclass
class Question:
    """A single EM eval question."""

    id: str
    paraphrases: List[str]
    judge_prompts: Dict[str, str]
    temperature: float = 1.0
    system: Optional[str] = None
    type: str = "free_form_judge_0_100"

    def build_conversation(self, paraphrase: str) -> List[dict]:
        """Build a chat conversation for one paraphrase of this question."""
        messages: List[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": paraphrase})
        return messages


def resolve_question_set_path(question_set: str) -> str:
    """Map a question-set name to its vendored YAML path.

    Accepts either a bare name (``first_plot_questions``) or an explicit
    path to a ``.yaml`` file.
    """
    if question_set.endswith(".yaml") or os.path.sep in question_set:
        return question_set
    return os.path.join(EM_QUESTIONS_DIR, f"{question_set}.yaml")


def load_questions(
    path: str,
    *,
    include_format_questions: bool = False,
) -> List[Question]:
    """Load EM questions from a YAML file.

    Args:
        path: Path to the question YAML (or a bare set name resolved via
            :func:`resolve_question_set_path`).
        include_format_questions: If False (default), drop entries whose id
            ends with ``_json`` / ``_template`` (format-constrained variants),
            keeping the plain free-form questions.
    """
    path = resolve_question_set_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"EM question file not found: {path}. "
            f"Fetch it with: honesty-evals em fetch"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of questions in {path}, got {type(data)}")

    questions: List[Question] = []
    for entry in data:
        qid = entry["id"]
        if not include_format_questions and qid.endswith(_FORMAT_SUFFIXES):
            continue
        qtype = entry["type"]
        if qtype != "free_form_judge_0_100":
            raise ValueError(
                f"Question {qid!r} has unsupported type {qtype!r}; "
                f"only 'free_form_judge_0_100' is supported."
            )
        judge_prompts = entry["judge_prompts"]
        for metric in ("aligned", "coherent"):
            if metric not in judge_prompts:
                raise KeyError(
                    f"Question {qid!r} is missing judge prompt {metric!r}; "
                    f"found {list(judge_prompts)}."
                )
        questions.append(
            Question(
                id=qid,
                paraphrases=list(entry["paraphrases"]),
                judge_prompts=dict(judge_prompts),
                temperature=float(entry.get("temperature", 1.0)),
                system=entry.get("system"),
                type=qtype,
            )
        )

    if not questions:
        raise ValueError(f"No usable questions loaded from {path}.")
    return questions
