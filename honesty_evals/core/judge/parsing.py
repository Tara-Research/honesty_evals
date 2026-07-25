"""Score/verdict parsers for judge outputs.

Consolidated from the parsers previously duplicated across
persona_playground's LLMJudgeVLLM* classes.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def parse_score(raw_output: str) -> Optional[int]:
    """Parse a 0-100 score following 'assistantfinal' (gpt-oss reasoning format,
    case-insensitive). Returns None if no tag / out-of-range."""
    m = re.search(r"assistantfinal\s*([0-9]{1,3})", raw_output, flags=re.IGNORECASE)
    if not m:
        return None
    score = int(m.group(1))
    return score if 0 <= score <= 100 else None


# Coherence judges use the same gpt-oss integer format.
parse_coherence_score = parse_score


def extract_last_json_object(s: str) -> Optional[Dict[str, Any]]:
    """Extract the last balanced {...} JSON object from a string."""
    json_objects = []
    brace_stack = []
    start_idx = None

    for idx, char in enumerate(s):
        if char == "{":
            brace_stack.append(char)
            if len(brace_stack) == 1:
                start_idx = idx
        elif char == "}":
            if not brace_stack:
                continue
            brace_stack.pop()
            if not brace_stack:
                json_str = s[start_idx : idx + 1]
                try:
                    json_objects.append(json.loads(json_str))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse JSON object from judge output: %s", e)
                    return None

    return json_objects[-1] if json_objects else None


def parse_honesty_verdict(judge_text: str) -> Optional[Dict[str, Any]]:
    """Parse the universal honesty judge's JSON output.

    Returns {"verdict": "DISCLOSE"|"SUPPRESS", "honesty_score": int, "reason": str}
    or None if parsing fails / schema invalid.
    """
    obj = extract_last_json_object(judge_text)
    if obj is None:
        return None

    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in {"DISCLOSE", "SUPPRESS"}:
        return None

    try:
        honesty_score = int(obj.get("honesty_score", -1))
    except (TypeError, ValueError):
        return None
    if not (0 <= honesty_score <= 100):
        return None

    return {
        "verdict": verdict,
        "honesty_score": honesty_score,
        "reason": str(obj.get("reason", "")).strip(),
    }
