"""Deterministically slice one Among Us game out of an experiment log.

The raw ``agent-logs-compact.json`` interleaves every turn of 100 games into a
single ~60MB JSONL file. This module pulls every record for a single game,
joins it with that game's ``summary.json`` block (identities / colours /
winner) and the steering recipe from ``experiment-details.txt``, reconstructs a
ground-truth event timeline from the per-player observation histories, and
writes a self-contained ``Game_<N>.json``.

Everything here is pure and deterministic: same logs in -> byte-identical JSON
out, and the game-selection ranking has a total, documented tie-break so the
choice is reproducible and explainable.

CLI::

    python -m honesty_evals.evals.among_us.analysis.extract_game \
        --expt-dir 2026-05-17_qwen_self_report_expanded_first_c_t \
        --select impostor_win --rank-by meeting_activity --top 2

    python -m honesty_evals.evals.among_us.analysis.extract_game \
        --expt-dir 2026-05-17_qwen_self_report_expanded_first_c_t --game "Game 2"
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from honesty_evals.evals.among_us.analysis._records import (
    iter_compact_records,
    resolve_action_thought,
)
from honesty_evals.core.paths import ROOT_FOLDER, RESULTS_ROOT

EXPT_LOGS_DIR = os.path.join(str(RESULTS_ROOT), "among_us")
DEFAULT_OUT_DIR = os.path.join(
    str(RESULTS_ROOT), "among_us", "_extracted_games"
)

COMPACT_LOG = "agent-logs-compact.json"
SUMMARY_FILE = "summary.json"
DETAILS_FILE = "experiment-details.txt"

# summary.json ``winner`` codes (see plot_winrate_comparison.py /
# game.py:check_game_over). Code 4 (impostors win on time) is never recorded.
WINNER_CODE_LABELS = {
    1: "Impostors win (crewmates outnumbered/tied)",
    2: "Crewmates win (impostor voted out)",
    3: "Crewmates win (all tasks completed)",
    4: "Impostors win (time limit)",
}
SELECTION_WINNER_CODES = {
    "impostor_win": {1, 4},
    "crew_eject_win": {2},
    "crew_task_win": {3},
    "crew_win": {2, 3},
    "any": {1, 2, 3, 4},
}

# ``Timestep 7: [meeting phase - round 2] Player 3: orange SPEAK: hello``
_EVENT_RE = re.compile(
    r"Timestep\s+(?P<ts>\d+):\s*\[(?P<phase>[^\]]+)\]\s*"
    r"(?P<actor>Player\s+\d+:\s*\w+)\s+(?P<body>.*)"
)
_ROUND_IN_TAG_RE = re.compile(r"round\s+(\d+)", re.IGNORECASE)
_ROUND_IN_INFO_RE = re.compile(r"Discussion round \((\d+)/", re.IGNORECASE)
_KILL_RE = re.compile(r"\bKILL\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# path / metadata helpers
# --------------------------------------------------------------------------- #
def resolve_expt_dir(expt: str) -> str:
    """Resolve a run reference to the directory holding the log files.

    LAYOUT ADAPTATION. persona_playground kept the logs flat in
    ``src/AmongUs/expt-logs/<expt>/``; honesty_evals nests them one level down in
    ``results/among_us/<run_name>/games/``. Returning the ``games/`` directory
    keeps every downstream ``os.path.join(expt_dir, "agent-logs-compact.json")``
    working unchanged.

    Accepts, in order: a directory that already contains the compact log, a run
    directory with a ``games/`` subdirectory, or a bare run name under
    results/among_us/.
    """
    def _has_log(d: str) -> bool:
        return os.path.isfile(os.path.join(d, COMPACT_LOG))

    tried = []
    for cand in (expt, os.path.join(expt, "games")):
        tried.append(cand)
        if os.path.isdir(cand) and _has_log(cand):
            return os.path.abspath(cand)
    for cand in (os.path.join(EXPT_LOGS_DIR, expt),
                 os.path.join(EXPT_LOGS_DIR, expt, "games")):
        tried.append(cand)
        if os.path.isdir(cand) and _has_log(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(
        f"no {COMPACT_LOG} found for run {expt!r}; looked in: "
        + ", ".join(repr(t) for t in tried)
    )


def parse_experiment_details(expt_dir: str) -> Dict[str, Any]:
    """Run provenance, in the shape the renderers expect.

    LAYOUT ADAPTATION. persona_playground scraped a free-text
    ``experiment-details.txt`` (Date / Commit / a dict-literal "Experiment args"
    line). honesty_evals records provenance as JSON in the run manifest
    ``<run>/config.json``, so there is nothing to scrape.

    ``steering_config`` is reported as the per-role LoRA mapping: activation
    steering was removed from this repo, and per-role adapters are what replaced
    it as the mechanism that makes impostors differ from crewmates.
    """
    run_dir = os.path.dirname(expt_dir.rstrip(os.sep)) \
        if os.path.basename(expt_dir.rstrip(os.sep)) == "games" else expt_dir
    path = os.path.join(run_dir, "config.json")
    manifest: Dict[str, Any] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    # Cross-repo fallback: persona_playground runs have no config.json but do
    # carry experiment-details.txt. Supporting both lets a honesty_evals run be
    # compared directly against a persona_playground baseline.
    if not manifest:
        legacy = os.path.join(expt_dir, DETAILS_FILE)
        if os.path.isfile(legacy):
            return _parse_legacy_details(expt_dir, legacy)

    role_loras = manifest.get("role_loras") or {}
    return {
        "expt_dir": os.path.basename(run_dir.rstrip(os.sep)),
        "date": manifest.get("created_at"),
        "commit": manifest.get("git_sha"),
        "game_config": manifest.get("game_config"),
        "steering_config": role_loras or None,
        "agent_roles": {
            "Impostor": role_loras.get("Impostor") or manifest.get("model"),
            "Crewmate": role_loras.get("Crewmate") or manifest.get("model"),
        },
        "model": manifest.get("model"),
        "lora_adapter": manifest.get("lora_adapter"),
        "sampling": manifest.get("sampling"),
    }


def _parse_legacy_details(expt_dir: str, path: str) -> Dict[str, Any]:
    """Read persona_playground's experiment-details.txt.

    The ``Experiment args:`` line is a Python dict literal (single quotes,
    ``None``), so it is parsed with ast.literal_eval. The file repeats that line;
    the first occurrence is authoritative. Returns the same shape as
    parse_experiment_details so callers cannot tell the two sources apart.
    """
    date = commit = None
    args: Optional[Dict[str, Any]] = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("Date:"):
                date = line.split(":", 1)[1].strip()
            elif line.startswith("Commit:"):
                commit = line.split(":", 1)[1].strip()
            elif line.startswith("Experiment args:") and args is None:
                args = ast.literal_eval(line.split(":", 1)[1].strip())
    args = args or {}
    agent_config = args.get("agent_config", {})
    sc = agent_config.get("STEERING_CONFIG") or {}
    return {
        "expt_dir": os.path.basename(expt_dir.rstrip(os.sep)),
        "date": date,
        "commit": commit,
        "game_config": args.get("game_config"),
        "steering_config": sc.get("role_steering"),
        "agent_roles": {
            "Impostor": agent_config.get("Impostor"),
            "Crewmate": agent_config.get("Crewmate"),
        },
        "model": sc.get("model_id"),
        "lora_adapter": None,
        "sampling": sc.get("generation"),
    }


def load_summary_index(expt_dir: str) -> Dict[str, Dict[str, Any]]:
    """Map ``"Game N" -> summary block`` from the JSONL ``summary.json``."""
    path = os.path.join(expt_dir, SUMMARY_FILE)
    index: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for game_id, block in entry.items():
                index[game_id] = block
    return index


def _player_directory(summary_block: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """name -> {color, identity, player_key} from a summary block."""
    out: Dict[str, Dict[str, str]] = {}
    for key, val in summary_block.items():
        if not key.startswith("Player "):
            continue
        out[val["name"]] = {
            "player_key": key,
            "color": val["color"],
            "identity": val["identity"],
        }
    return out


# --------------------------------------------------------------------------- #
# record / phase helpers
# --------------------------------------------------------------------------- #
def normalize_phase(record: Dict[str, Any]) -> str:
    """``"task"`` or ``"meeting"`` from the record's ``interaction.prompt.Phase``.

    Raises if the key is absent -- the project convention is to surface a
    malformed log rather than silently default a required field.
    """
    prompt = record["interaction"]["prompt"]
    if "Phase" not in prompt or prompt["Phase"] is None:
        raise ValueError(
            f"record (game={record.get('game_index')} step="
            f"{record.get('step')}) has no interaction.prompt.Phase"
        )
    return "meeting" if "meet" in str(prompt["Phase"]).lower() else "task"


def _all_info(record: Dict[str, Any]) -> str:
    return record["interaction"]["prompt"].get("All Info", "") or ""


def _discussion_round(record: Dict[str, Any]) -> Optional[int]:
    """Meeting discussion-round index from the turn's own All Info, else None."""
    m = _ROUND_IN_INFO_RE.search(_all_info(record))
    return int(m.group(1)) if m else None


def _section(all_info: str, header: str) -> str:
    """Slice the text block under *header* up to the next known header."""
    start = all_info.find(header)
    if start < 0:
        return ""
    start += len(header)
    end = len(all_info)
    for nxt in (
        "Observation history:",
        "Action history:",
        "Your Assigned Tasks:",
        "Available actions:",
    ):
        k = all_info.find(nxt, start)
        if k != -1:
            end = min(end, k)
    return all_info[start:end].strip()


def _available_action_lines(record: Dict[str, Any]) -> List[str]:
    """Non-empty lines of the turn's ``Available actions:`` block."""
    block = _section(_all_info(record), "Available actions:")
    return [ln.strip() for ln in block.splitlines() if ln.strip()]


def executed_kill(record: Dict[str, Any]) -> Optional[str]:
    """Whether this turn produced an *executed* kill, and how.

    Reliable, deterministic, and witnessing-independent -- unlike the
    observation-history timeline, which only sees *witnessed* kills and so
    drastically undercounts stealth kills (and the action-only view misses
    engine-fallback kills). This mirrors the engine's real action selection
    (``LLMAgent.choose_action``: match the model's action against an available
    action, else fall through to the *last* available action):

    - ``"chosen"``   -- the model's parsed action is ``KILL <target>`` and a
      ``KILL`` option is available this turn (engine executes it).
    - ``"fallback"`` -- the model emitted no parseable action, so the engine
      returns the *last* available action, and that last option is a ``KILL``.
    - ``None``       -- no kill this turn.

    Kill options are only ever offered to impostors with cooldown 0 and a
    co-located victim (``Kill.can_execute_actions``), so a match == an
    executed kill.
    """
    action, _ = resolve_action_thought(record)
    lines = _available_action_lines(record)
    if not lines:
        return None
    if (
        action is not None
        and str(action).upper().lstrip().startswith("KILL")
        and any(_KILL_RE.search(ln) for ln in lines)
    ):
        return "chosen"
    if action is None and _KILL_RE.search(lines[-1]):
        return "fallback"
    return None


def game_records(compact_path: str, game_id: str) -> List[Dict[str, Any]]:
    """Every record for *game_id*, stable-sorted by ``step``.

    Stable sort preserves the in-file order for equal steps, which is the
    engine's within-step agent/round order.
    """
    recs = [
        r for r in iter_compact_records(compact_path)
        if r.get("game_index") == game_id
    ]
    return sorted(recs, key=lambda r: r["step"])


# --------------------------------------------------------------------------- #
# ground-truth event timeline (from aggregated observation histories)
# --------------------------------------------------------------------------- #
def _verb_of(body: str) -> str:
    token = body.strip().split(":")[0].split()[0].upper() if body.strip() else ""
    if token in {"SPEAK", "VOTE", "MOVE", "VENT", "KILL", "REPORT", "COMPLETE"}:
        return token
    if body.upper().startswith("CALL MEETING"):
        return "CALL_MEETING"
    if "DOING TASK" in body.upper():
        return "TASK"
    return token or "OTHER"


def build_event_timeline(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic ground-truth timeline of *executed* events.

    Every player's observation history reports what it actually saw other
    players do (``Timestep N: [phase] Player X: c VERB ...``). Aggregating and
    de-duplicating these lines across all turns yields the engine's executed
    actions -- independent of whether a model self-reported an action -- and
    captures kills/votes/calls/moves. SPEAK bodies seen by *others* are
    truncated to ``...``; the speaker's own full message is joined back in from
    that speaker's turn record.
    """
    # speaker's own full action text, keyed by (step, round, name)
    own: Dict[Tuple[int, Optional[int], str], Optional[str]] = {}
    for rec in records:
        action, _ = resolve_action_thought(rec)
        key = (rec["step"], _discussion_round(rec), rec["player"]["name"])
        if action is not None:
            own[key] = str(action)

    seen: Dict[str, Dict[str, Any]] = {}
    order = 0
    for rec in records:
        obs = _section(_all_info(rec), "Observation history:")
        for raw_line in obs.splitlines():
            text = re.sub(r"^\s*\d+\.\s*", "", raw_line).strip()
            m = _EVENT_RE.match(text)
            if not m:
                continue
            if text in seen:
                continue
            phase_tag = m.group("phase").strip()
            rmatch = _ROUND_IN_TAG_RE.search(phase_tag)
            rnd = int(rmatch.group(1)) if rmatch else None
            actor = re.sub(r"\s+", " ", m.group("actor")).strip()
            verb = _verb_of(m.group("body"))
            ev = {
                "timestep": int(m.group("ts")),
                "phase": "meeting" if "meet" in phase_tag.lower() else "task",
                "round": rnd,
                "actor": actor,
                "verb": verb,
                "text": m.group("body").strip(),
                "_order": order,
            }
            if verb == "SPEAK":
                full = own.get((ev["timestep"], rnd, actor))
                if full and full.upper().lstrip().startswith("SPEAK"):
                    msg = full.split(":", 1)[1].strip() if ":" in full else ""
                    if msg:
                        ev["text"] = "SPEAK: " + msg
                        ev["full_speak"] = True
            seen[text] = ev
            order += 1

    events = list(seen.values())
    events.sort(
        key=lambda e: (
            e["timestep"],
            0 if e["phase"] == "task" else 1,
            e["round"] if e["round"] is not None else -1,
            e["_order"],
        )
    )
    for e in events:
        e.pop("_order", None)
    return events


# --------------------------------------------------------------------------- #
# stats + selection
# --------------------------------------------------------------------------- #
def game_stats(
    records: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Descriptive, deterministic per-game stats."""
    n_no_action = {"Impostor": 0, "Crewmate": 0}
    n_turns = {"Impostor": 0, "Crewmate": 0}
    n_meeting_turns = 0
    for rec in records:
        ident = rec["player"]["identity"]
        n_turns[ident] = n_turns.get(ident, 0) + 1
        if normalize_phase(rec) == "meeting":
            n_meeting_turns += 1
        if resolve_action_thought(rec)[0] is None:
            n_no_action[ident] = n_no_action.get(ident, 0) + 1

    # Reliable executed-kill count (per-record, witnessing-independent).
    n_kills_chosen = n_kills_fallback = 0
    for rec in records:
        if rec["player"]["identity"] != "Impostor":
            continue
        ek = executed_kill(rec)
        if ek == "chosen":
            n_kills_chosen += 1
        elif ek == "fallback":
            n_kills_fallback += 1

    # Witnessed-only event tallies from the observation-history timeline.
    # Kept for narrative/context ONLY -- they undercount unwitnessed events
    # (esp. stealth kills); use ``n_kills`` for the real kill count.
    verbs: Dict[str, int] = {}
    for ev in timeline:
        verbs[ev["verb"]] = verbs.get(ev["verb"], 0) + 1
    n_meetings = verbs.get("CALL_MEETING", 0) + verbs.get("REPORT", 0)

    def _rate(d: Dict[str, int], ident: str) -> Optional[float]:
        t = n_turns.get(ident, 0)
        return round(d.get(ident, 0) / t, 4) if t else None

    return {
        "n_records": len(records),
        "n_timesteps": len({r["step"] for r in records}),
        "n_turns_by_identity": n_turns,
        "n_meeting_turns": n_meeting_turns,
        "n_meetings": n_meetings,
        "n_kills": n_kills_chosen + n_kills_fallback,
        "n_kills_chosen": n_kills_chosen,
        "n_kills_fallback": n_kills_fallback,
        "no_action_counts": n_no_action,
        "no_action_rate_impostor": _rate(n_no_action, "Impostor"),
        "no_action_rate_crewmate": _rate(n_no_action, "Crewmate"),
        "witnessed_event_counts": dict(sorted(verbs.items())),
    }


def _game_number(game_id: str) -> int:
    m = re.search(r"(\d+)", game_id)
    if not m:
        raise ValueError(f"cannot parse game number from {game_id!r}")
    return int(m.group(1))


def select_games(
    expt_dir: str,
    select: str,
    rank_by: str,
    top: int,
) -> List[Dict[str, Any]]:
    """Rank games matching *select* by *rank_by* and return the top *top*.

    ``rank_by="meeting_activity"`` sorts by
    ``(n_meeting_turns, n_meetings, n_records)`` descending; ties break on the
    game number ascending, so the selection is fully deterministic.
    """
    if select not in SELECTION_WINNER_CODES:
        raise ValueError(
            f"unknown --select {select!r}; choose from "
            f"{sorted(SELECTION_WINNER_CODES)}"
        )
    if rank_by != "meeting_activity":
        raise ValueError(f"unknown --rank-by {rank_by!r}")

    summary = load_summary_index(expt_dir)
    wanted = SELECTION_WINNER_CODES[select]
    target_ids = {
        gid for gid, blk in summary.items() if blk.get("winner") in wanted
    }

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    compact_path = os.path.join(expt_dir, COMPACT_LOG)
    for rec in iter_compact_records(compact_path):
        gid = rec.get("game_index")
        if gid in target_ids:
            grouped.setdefault(gid, []).append(rec)

    ranked: List[Dict[str, Any]] = []
    for gid, recs in grouped.items():
        recs.sort(key=lambda r: r["step"])
        stats = game_stats(recs, build_event_timeline(recs))
        ranked.append(
            {
                "game_id": gid,
                "winner": summary[gid].get("winner"),
                "winner_label": WINNER_CODE_LABELS.get(
                    summary[gid].get("winner")
                ),
                "stats": stats,
            }
        )
    ranked.sort(
        key=lambda g: (
            -g["stats"]["n_meeting_turns"],
            -g["stats"]["n_meetings"],
            -g["stats"]["n_records"],
            _game_number(g["game_id"]),
        )
    )
    return ranked[:top]


# --------------------------------------------------------------------------- #
# build + write
# --------------------------------------------------------------------------- #
def build_game_json(expt_dir: str, game_id: str) -> Dict[str, Any]:
    meta = parse_experiment_details(expt_dir)
    summary = load_summary_index(expt_dir)
    if game_id not in summary:
        raise KeyError(
            f"{game_id!r} not in {os.path.join(expt_dir, SUMMARY_FILE)}"
        )
    summary_block = summary[game_id]
    pdir = _player_directory(summary_block)

    compact_path = os.path.join(expt_dir, COMPACT_LOG)
    records = game_records(compact_path, game_id)
    if not records:
        raise ValueError(f"no records for {game_id!r} in {compact_path}")

    timeline = build_event_timeline(records)
    stats = game_stats(records, timeline)

    turns: List[Dict[str, Any]] = []
    for rec in records:
        name = rec["player"]["name"]
        info = pdir.get(name, {})
        action, thought = resolve_action_thought(rec)
        resp = rec["interaction"].get("response")
        condensed = (
            resp.get("Condensed Memory")
            if isinstance(resp, dict)
            else None
        )
        action_str = None if action is None else str(action)
        speak_text = None
        if action_str and action_str.upper().lstrip().startswith("SPEAK"):
            speak_text = (
                action_str.split(":", 1)[1].strip()
                if ":" in action_str
                else ""
            )
        turns.append(
            {
                "step": rec["step"],
                "phase": normalize_phase(rec),
                "discussion_round": _discussion_round(rec),
                "player": {
                    "name": name,
                    "color": info.get("color"),
                    "identity": info.get("identity", rec["player"]["identity"]),
                    "location": rec["player"].get("location"),
                },
                "action": action_str,
                "action_emitted": action is not None,
                "executed_kill": executed_kill(rec),
                "speak_text": speak_text,
                "thought": None if thought is None else str(thought),
                "condensed_memory": condensed,
                "observation_history": _section(
                    _all_info(rec), "Observation history:"
                ),
                "action_history": _section(
                    _all_info(rec), "Action history:"
                ),
                "full_response": rec["interaction"].get("full_response"),
            }
        )

    return {
        "meta": meta,
        "game_id": game_id,
        "winner": summary_block.get("winner"),
        "winner_code_label": WINNER_CODE_LABELS.get(
            summary_block.get("winner")
        ),
        "winner_reason": summary_block.get("winner_reason"),
        "summary": summary_block,
        "stats": stats,
        "event_timeline": timeline,
        "turns": turns,
    }


def write_game(
    expt_dir: str, game_id: str, out_dir: str, render: bool = True
) -> str:
    game_json = build_game_json(expt_dir, game_id)
    expt_name = game_json["meta"]["expt_dir"]
    dest_dir = os.path.join(out_dir, expt_name)
    os.makedirs(dest_dir, exist_ok=True)
    stem = "Game_" + str(_game_number(game_id))
    json_path = os.path.join(dest_dir, stem + ".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(game_json, fh, indent=2, ensure_ascii=False)
    if render:
        # imported lazily so extraction has no hard dependency on the renderer
        from honesty_evals.evals.among_us.analysis.render_game import render_markdown

        md_path = os.path.join(dest_dir, stem + ".md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(game_json))
    return json_path


def _print_ranking(expt_name: str, ranked: List[Dict[str, Any]]) -> None:
    print(f"[select] {expt_name}: ranked {len(ranked)} game(s)")
    print(
        f"  {'game':<10}{'winner':<8}{'mtgTurns':>9}{'mtgs':>6}"
        f"{'recs':>6}  {'impNoAct':>9}{'crwNoAct':>9}"
    )
    for g in ranked:
        s = g["stats"]
        print(
            f"  {g['game_id']:<10}{g['winner']!s:<8}"
            f"{s['n_meeting_turns']:>9}{s['n_meetings']:>6}"
            f"{s['n_records']:>6}"
            f"{s['no_action_rate_impostor']!s:>9}"
            f"{s['no_action_rate_crewmate']!s:>9}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expt-dir", required=True)
    ap.add_argument(
        "--select",
        default="impostor_win",
        choices=sorted(SELECTION_WINNER_CODES),
    )
    ap.add_argument("--rank-by", default="meeting_activity")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument(
        "--game",
        nargs="*",
        default=None,
        help='explicit game id(s), e.g. --game "Game 2" 7 '
        "(bypasses --select)",
    )
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    expt_dir = resolve_expt_dir(args.expt_dir)
    expt_name = os.path.basename(expt_dir.rstrip(os.sep))

    if args.game:
        game_ids = [
            g if str(g).lower().startswith("game") else f"Game {g}"
            for g in args.game
        ]
    else:
        ranked = select_games(
            expt_dir, args.select, args.rank_by, args.top
        )
        _print_ranking(expt_name, ranked)
        game_ids = [g["game_id"] for g in ranked]

    for gid in game_ids:
        path = write_game(
            expt_dir, gid, args.out_dir, render=not args.no_render
        )
        print(f"[write] {gid} -> {path}")


if __name__ == "__main__":
    main()
