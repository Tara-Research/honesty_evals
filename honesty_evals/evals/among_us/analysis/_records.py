"""Record-parsing primitives for Among Us game logs.

PORTED from persona_playground src/AmongUs/utils.py (iter_compact_records,
ACTION_PATHS / THINKING_PATHS, _flatten_dotted, _first_present,
resolve_action_thought), with the pandas dependency dropped — nothing here
needed a DataFrame.

Layout difference from persona_playground: it read
``src/AmongUs/expt-logs/<expt>/{agent-logs-compact.json,summary.json}`` plus a
free-text ``experiment-details.txt``. honesty_evals writes
``results/among_us/<run_name>/games/{agent-logs-compact.json,summary.json}``
with provenance in the run manifest ``config.json`` instead. See run_paths().
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Response-schema variants the agents have emitted across runs. First path that
# resolves to a non-null value wins.
ACTION_PATHS: List[str] = [
    "interaction.response.Action",
    "interaction.response.Action.action",
    "interaction.response.SPEAK Strategy.action",
    "interaction.response.ACTION",
    "interaction.response.Thinking Process.action",
]

THINKING_PATHS: List[str] = [
    "interaction.response.Thinking Process",
    "interaction.response.Thinking Process.thought",
    "interaction.response.SPEAK Strategy.thought",
    "interaction.response.SPEAK Strategy",
    "interaction.response",
    "interaction.response.Action.thought",
]


def iter_compact_records(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one record per line of ``agent-logs-compact.json``.

    Only the *compact* log is true JSONL — the engine writes each interaction
    with no indent followed by a newline. ``agent-logs.json`` uses indent=2, so a
    single object spans many lines and is NOT line-parseable. Always read the
    compact file. Malformed lines are skipped so a partial/in-progress run still
    analyses.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_records(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """All records plus a count of lines that failed to parse."""
    good: List[Dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return good, bad


def _flatten_dotted(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts with ``.``-joined keys (json_normalize semantics).

    Recurses only into dicts: a parent whose value is a dict is replaced by its
    dotted children, while a parent whose value is a scalar (e.g. a plain-string
    ``interaction.response``) is kept as-is.
    """
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                out.update(_flatten_dotted(value, path))
            else:
                out[path] = value
    else:
        out[prefix] = obj
    return out


def _first_present(flat: Dict[str, Any], paths: List[str]) -> Optional[Any]:
    for path in paths:
        if path in flat and flat[path] is not None:
            return flat[path]
    return None


def resolve_action_thought(record: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Recover ``(action, thought)`` from one record — the single source of truth
    for response-schema-variant handling."""
    flat = _flatten_dotted(record)
    return _first_present(flat, ACTION_PATHS), _first_present(flat, THINKING_PATHS)


# ── Layout ───────────────────────────────────────────────────────────────────

def run_paths(run_dir: str) -> Dict[str, str]:
    """Resolve a honesty_evals among-us run directory to its log files.

    Accepts either the run dir (``results/among_us/<name>``) or its ``games/``
    subdirectory, so callers can pass whichever they have.
    """
    run_dir = os.path.abspath(run_dir)
    games = run_dir if os.path.basename(run_dir) == "games" else os.path.join(run_dir, "games")
    root = os.path.dirname(games) if os.path.basename(games) == "games" else run_dir
    return {
        "run_dir": root,
        "games_dir": games,
        "compact": os.path.join(games, "agent-logs-compact.json"),
        "summary": os.path.join(games, "summary.json"),
        "winners": os.path.join(root, "winners.json"),
        "manifest": os.path.join(root, "config.json"),
    }


def load_manifest(run_dir: str) -> Dict[str, Any]:
    """Run provenance. Replaces persona_playground's experiment-details.txt
    parsing: honesty_evals records model / lora_adapter / role_loras / sampling
    in the run manifest, so there is no free text to scrape."""
    p = run_paths(run_dir)["manifest"]
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def load_summary_index(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """``{game_index: summary_block}`` from summary.json (itself JSONL, one
    ``{"Game N": {...}}`` object per line). Empty dict if absent."""
    p = run_paths(run_dir)["summary"]
    index: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(p):
        return index
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            for game, body in blob.items():
                index[game] = body
    return index


def load_game_summary(filepath: str):
    """``summary.json`` (JSONL) -> DataFrame of Game / Winner / Winner Reason.

    PORTED from persona_playground src/AmongUs/utils.py. Kept DataFrame-shaped
    because plot_amongus_winrate.py builds on it with pandas (concat, groupby).
    pandas is imported lazily so the rest of this module stays dependency-light.
    """
    import pandas as pd

    with open(filepath, "r", encoding="utf-8") as fh:
        data = [json.loads(line.strip()) for line in fh if line.strip()]
    rows = [
        {
            "Game": game_id,
            "Winner": details.get("winner"),
            "Winner Reason": details.get("winner_reason"),
        }
        for entry in data
        for game_id, details in entry.items()
    ]
    return pd.DataFrame(rows)
