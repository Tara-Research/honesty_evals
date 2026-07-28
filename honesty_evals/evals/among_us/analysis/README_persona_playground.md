# Among Us per-game qualitative analysis

Deterministic tooling to pull one game out of the 60 MB interleaved logs and
read it. Same logs in → byte-identical JSON out; selection has a documented
total order so picks are reproducible.

## 1. Extract game(s)

Run from the repo root.

```bash
# Auto-select: top-N games matching an outcome, ranked by meeting activity
python -m src.AmongUs.analysis.extract_game \
  --expt-dir 2026-05-17_qwen_self_report_expanded_first_c_t \
  --select impostor_win --rank-by meeting_activity --top 2

# Explicit game(s) — bypasses selection
python -m src.AmongUs.analysis.extract_game \
  --expt-dir 2026-05-17_llama_self_report_expanded_first_c_t --game "Game 58" 7
```

`--select` ∈ `impostor_win` · `crew_eject_win` · `crew_task_win` · `crew_win`
· `any` (winner codes 1 / 2 / 3). `--rank-by meeting_activity` sorts by
`(n_meeting_turns, n_meetings, n_records)` desc, tie-break game number.
Other flags: `--top N`, `--out-dir`, `--no-render`.

Writes (git-ignored, regenerate any time):

```
src/AmongUs/analysis/games/<expt>/Game_<N>.json   # machine-reusable
src/AmongUs/analysis/games/<expt>/Game_<N>.md     # human transcript
```

The selection table is printed so the choice is auditable, e.g.:

```
  game      winner  mtgTurns  mtgs  recs   impNoAct crwNoAct
  Game 2    1             80     3   118   0.9744   0.2532
```

Re-render a tweaked transcript without re-extracting:
`python -m src.AmongUs.analysis.render_game --game-json <path>`.

## 2. Read the transcript (`Game_<N>.md`)

Read top-down — the header alone usually tells you the regime:

- **Headline stats** — `no-parsed-action rate` per identity + `event verbs`.
  High impostor no-action + coherent `full_response` ⇒ "mute" regime;
  degenerate `full_response` ⇒ "collapse"; low no-action + short ⇒ "confesses".
- **Ground-truth timeline** — engine-executed events (KILL/VOTE/MOVE/CALL),
  impostor flagged 🔴, SPEAK verbatim where the speaker's own text exists.
- **Impostor meeting deep-dive** — every impostor meeting turn with parsed
  action, `action_emitted`, thought, and a `full_response` excerpt: this is
  where you see honest-vs-deceptive (or truncated, or gibberish) behaviour.

## 3. Drill into the JSON

`Game_<N>.json` keys: `meta`, `winner`, `summary`, `stats`,
`event_timeline[]`, `turns[]`. Each turn has `player{name,color,identity,
location}`, `action`, `action_emitted`, `speak_text`, `thought`,
`condensed_memory`, `observation_history`, `full_response`.

```python
import json
g = json.load(open("src/AmongUs/analysis/games/<expt>/Game_2.json", encoding="utf-8"))
imp = [t for t in g["turns"]
       if t["player"]["identity"] == "Impostor" and t["phase"] == "meeting"]
for t in imp:
    print(t["step"], t["action_emitted"], (t["speak_text"] or t["full_response"])[:200])
```

## 4. Cohort-level comparison (multiple runs side-by-side)

The saved CLI for the table used throughout the investigation:

```bash
python -m src.AmongUs.analysis.cohort_analysis \
  2026-04-29_qwen_baseline \
  2026-05-13_qwen_self_report_expanded \
  2026-05-17_qwen_self_report_expanded_first_c_t \
  2026-04-18_llama_steered_100_games

# --out json  for the full per-run dict (downstream programmatic use)
```

Produces a markdown table with: tok / coef / games-fin/start / impWR / win-codes /
emit% / cut-off% / DEGEN% / median words / meeting usable-SPEAK% /
confession-of-all-mtg% / confession-of-speak% / games with ≥1 confession /
kills/g (chosen+fallback). All deterministic, tolerant of partial / in-progress
runs (skips truncated final lines and reports the count). Confession detector
(`CONFESSION_RE`) is a module-level constant — see the docstring for the
documented false-positive / false-negative limitations.

The `cohort_row(expt_dir) -> dict` function is also importable for ad-hoc work
without re-implementing the metrics, and the coherence buckets
(`coherence_label`) are exposed for one-off classification.

## 5. Lower-level building blocks (extract_game internals)

The same library functions power aggregate analysis — don't re-parse by hand:

```python
from src.AmongUs.utils import iter_compact_records, resolve_action_thought
from src.AmongUs.analysis.extract_game import (
    load_summary_index, build_event_timeline, game_stats)

d = "src/AmongUs/expt-logs/2026-05-17_gemma_self_report_expanded_first_c_t"
summ = load_summary_index(d)                       # game_id -> summary block
recs = list(iter_compact_records(f"{d}/agent-logs-compact.json"))
by = {}
for r in recs:                                     # group once
    by.setdefault(r["game_index"], []).append(r)
for gid, rs in by.items():
    rs.sort(key=lambda x: x["step"])
    st = game_stats(rs, build_event_timeline(rs))
    # st.no_action_rate_impostor, st.n_kills, summ[gid]["winner"], ...
```

Useful one-off metrics layered on top of `full_response` (see the
2026-05-18 Qwen report for worked examples): **degeneracy** = unique-word
ratio < 0.15 or top-trigram share > 0.25; **no-action** =
`resolve_action_thought(rec)[0] is None`; **regime** = cross-tab of the two.

## Related

- Findings: `src/AmongUs/reports/among_us_qwen_steering_qualitative.md`
- Winrate plots / experiment dir names: `src/AmongUs/plot_winrate_comparison.py`
- Tests (run before relying on edits): `python -m pytest test/test_among_us_extract.py`
