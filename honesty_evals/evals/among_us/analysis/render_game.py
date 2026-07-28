"""Render an extracted single-game JSON as a readable Markdown transcript.

Consumes the dict produced by :func:`extract_game.build_game_json` (or the
``Game_<N>.json`` file it writes). The transcript has three parts:

1. **Header** -- experiment, model, full steering recipe, winner, the player
   roster with identities/colours/tasks, and the headline stats (per-identity
   "no parsed action" rate).
2. **Ground-truth timeline** -- the engine's executed events grouped by
   timestep and phase; impostor actors flagged, kills/meetings/votes
   highlighted, SPEAK shown verbatim where the speaker's own text is available.
3. **Impostor meeting deep-dive** -- every impostor turn in every meeting with
   its parsed action (or a "no [Action] emitted" flag) and a ``full_response``
   excerpt, so honest-vs-deceptive behaviour is directly readable.

CLI: ``python -m honesty_evals.evals.among_us.analysis.render_game --game-json <path>``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

_IMP = "🔴"
_CREW = "·"
_HILITE = {"KILL", "CALL_MEETING", "REPORT", "VOTE", "VENT"}


def _excerpt(text: str, limit: int = 600) -> str:
    text = (text or "").strip().replace("\r", "")
    if len(text) > limit:
        text = text[:limit].rstrip() + " …[truncated]"
    return text


def _identity_maps(game: Dict[str, Any]):
    name_id, name_col = {}, {}
    for key, val in game["summary"].items():
        if key.startswith("Player "):
            name_id[val["name"]] = val["identity"]
            name_col[val["name"]] = val["color"]
    return name_id, name_col


def _header(game: Dict[str, Any]) -> List[str]:
    meta = game["meta"]
    sc = meta.get("steering_config") or {}
    sel = sc.get("steering_selection", {}) if isinstance(sc, dict) else {}
    role = sc.get("role_steering", {}) if isinstance(sc, dict) else {}
    stats = game["stats"]
    out = [
        f"# {game['game_id']} — {meta['expt_dir']}",
        "",
        f"- **Model**: `{sel.get('model_id', meta.get('agent_roles'))}`",
        f"- **Experiment date / commit**: {meta.get('date')} / "
        f"`{meta.get('commit')}`",
        f"- **Steering selection**: `{json.dumps(sel)}`",
        f"- **Role steering**: `{json.dumps(role)}`",
        f"- **Winner**: `{game['winner']}` — {game['winner_code_label']} "
        f"(_{game['winner_reason']}_)",
        "",
        "### Headline stats",
        "",
        f"- records: **{stats['n_records']}**, timesteps: "
        f"**{stats['n_timesteps']}**, meetings: **{stats['n_meetings']}** "
        f"(witnessed)",
        f"- **executed kills**: **{stats['n_kills']}** "
        f"(chosen {stats['n_kills_chosen']} + engine-fallback "
        f"{stats['n_kills_fallback']}) — reliable, witnessing-independent",
        f"- **no-parsed-action rate** — impostor "
        f"**{stats['no_action_rate_impostor']}**, crewmate "
        f"**{stats['no_action_rate_crewmate']}** "
        f"(counts {stats['no_action_counts']})",
        f"- witnessed-only event verbs: `{stats['witnessed_event_counts']}` "
        f"(undercounts stealth events; narrative context only)",
        "",
        "### Roster",
        "",
        "| Player | Color | Identity | Tasks |",
        "| --- | --- | --- | --- |",
    ]
    for key, val in game["summary"].items():
        if not key.startswith("Player "):
            continue
        flag = _IMP if val["identity"] == "Impostor" else ""
        out.append(
            f"| {key} | {val['color']} | {flag} {val['identity']} | "
            f"{', '.join(val.get('tasks', []))} |"
        )
    out.append("")
    return out


def _timeline(game: Dict[str, Any], name_id: Dict[str, str]) -> List[str]:
    out = ["## Ground-truth event timeline", "",
           "_Executed events reconstructed from all players' observation "
           "histories. SPEAK shown verbatim where the speaker's own text was "
           "recoverable, else `...` (what others saw)._", ""]
    cur_ts = cur_phase = None
    for ev in game["event_timeline"]:
        if (ev["timestep"], ev["phase"]) != (cur_ts, cur_phase):
            cur_ts, cur_phase = ev["timestep"], ev["phase"]
            label = "🗳️ MEETING" if ev["phase"] == "meeting" else "🛠️ task"
            out += ["", f"### t={cur_ts} — {label}", ""]
        actor = ev["actor"]
        flag = _IMP if name_id.get(actor) == "Impostor" else _CREW
        rnd = f" r{ev['round']}" if ev["round"] is not None else ""
        verb = ev["verb"]
        body = ev["text"]
        if verb in _HILITE:
            body = f"**{body}**"
        out.append(f"- `t{ev['timestep']}{rnd}` {flag} **{actor}** — {body}")
    out.append("")
    return out


def _impostor_deepdive(game: Dict[str, Any]) -> List[str]:
    out = [
        "## Impostor meeting deep-dive",
        "",
        "_Every impostor turn during meetings. `action_emitted=False` means "
        "the model never produced an `[Action]` line (engine then falls back "
        "to the last available action); read `full_response` to see what it "
        "did instead._",
        "",
    ]
    imp_turns = [
        t
        for t in game["turns"]
        if t["player"]["identity"] == "Impostor" and t["phase"] == "meeting"
    ]
    if not imp_turns:
        out += ["_No impostor meeting turns in this game._", ""]
    cur_ts = None
    for t in imp_turns:
        if t["step"] != cur_ts:
            cur_ts = t["step"]
            out += ["", f"### Meeting at t={cur_ts}", ""]
        p = t["player"]
        rnd = (
            f"round {t['discussion_round']}"
            if t["discussion_round"] is not None
            else "round ?"
        )
        out.append(
            f"**{p['name']}** ({p['color']}, {rnd}) — "
            f"action_emitted=`{t['action_emitted']}`"
        )
        out.append("")
        if t["action"]:
            out.append(f"- **action**: `{_excerpt(t['action'], 400)}`")
        if t["thought"]:
            out.append(f"- **thought**: {_excerpt(t['thought'], 700)}")
        out.append(
            "- **full_response**:\n\n```\n"
            f"{_excerpt(t['full_response'], 1200)}\n```"
        )
        out.append("")
    return out


def _impostor_task_actions(game: Dict[str, Any]) -> List[str]:
    out = ["## Impostor task-phase actions (kills / vents / moves)", ""]
    rows = []
    for t in game["turns"]:
        if t["player"]["identity"] != "Impostor" or t["phase"] != "task":
            continue
        act = t["action"] or "[no action emitted]"
        rows.append(
            f"- `t{t['step']}` **{t['player']['name']}** @ "
            f"{t['player']['location']} — {_excerpt(act, 160)}"
        )
    out += rows or ["_none_"]
    out.append("")
    return out


def render_markdown(game: Dict[str, Any]) -> str:
    name_id, _ = _identity_maps(game)
    parts: List[str] = []
    parts += _header(game)
    parts += _timeline(game, name_id)
    parts += _impostor_task_actions(game)
    parts += _impostor_deepdive(game)
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-json", required=True)
    ap.add_argument(
        "--out",
        default=None,
        help="output .md path (default: sibling of --game-json)",
    )
    args = ap.parse_args()
    with open(args.game_json, "r", encoding="utf-8") as fh:
        game = json.load(fh)
    md = render_markdown(game)
    out = args.out or os.path.splitext(args.game_json)[0] + ".md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[render] {args.game_json} -> {out}")


if __name__ == "__main__":
    main()
