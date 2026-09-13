"""Regression tests for MASK's reasoning-trace stripping.

The bug these pin: Qwen3.6's chat template ends the prompt with a dangling
``<think>\n``, so completions carry ONLY the closing ``</think>``. A
paired-tag regex silently matched nothing and the judge scored reasoning
traces instead of answers (job 419357, before ENABLE_THINKING=false).
"""
from __future__ import annotations

import pytest

from honesty_evals.evals.mask.generate import (
    _strip_and_tally,
    _strip_think_tags,
    _trace_kind,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        # ENABLE_THINKING=false: template emits a pre-closed block, so the
        # completion has no tags at all and must pass through untouched.
        ("The answer is 42.", "The answer is 42."),
        # The real Qwen3.6 shape: closing tag only, opening tag was in the prompt.
        ("reasoning here\n</think>\n\nThe answer is 42.", "The answer is 42."),
        # Paired tags (models that emit their own opening tag).
        ("<think>reasoning</think>The answer is 42.", "The answer is 42."),
        # Truncated mid-reasoning: no closing tag, so nothing is dropped. The
        # caller counts these rather than silently judging an empty string.
        ("reasoning that never ends", "reasoning that never ends"),
        # Only the LAST close ends the trace.
        ("a\n</think>\nb </think> c", "c"),
        # The distill organism's spelling. `</think>` is NOT a substring of
        # `</thinking>`, so a <think>-only stripper passes this through whole
        # and the judge scores the reasoning instead of the answer.
        ("<thinking>reasoning</thinking>The answer is 42.", "The answer is 42."),
        ("reasoning\n</thinking>\n\nThe answer is 42.", "The answer is 42."),
        # Mixed spellings in one completion: still only the last close counts.
        ("a</think>b</thinking>The answer is 42.", "The answer is 42."),
        # An opening tag that never closes is left alone, same as <think>.
        ("<thinking>reasoning that never ends", "<thinking>reasoning that never ends"),
    ],
)
def test_strip_think_tags(text: str, expected: str) -> None:
    assert _strip_think_tags(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is 42.", None),
        ("reasoning\n</think>\n\nThe answer is 42.", "closed"),
        ("<thinking>reasoning</thinking>The answer is 42.", "closed"),
        ("<thinking>reasoning that never ends", "unclosed"),
        # No visible opening tag and no close: indistinguishable from prose, so
        # it is NOT guessed at.
        ("reasoning that never ends", None),
    ],
)
def test_trace_kind(text: str, expected: str | None) -> None:
    assert _trace_kind(text) == expected


def test_strip_and_tally_counts_every_completion() -> None:
    """The tally is the only surviving record: only stripped text is persisted."""
    tally = {"total": 0, "closed": 0, "unclosed": 0}
    stripped = [
        _strip_and_tally(t, tally)
        for t in [
            "plain answer",
            "<thinking>r</thinking>hacked answer",
            "reasoning\n</think>\nanswer",
            "<think>never closes",
        ]
    ]
    assert tally == {"total": 4, "closed": 2, "unclosed": 1}
    assert stripped[1] == "hacked answer"
