"""LLM-judge scoring for an audit campaign — a real CLI (no notebook).

Takes ONE campaign dir (``--exp-dir``), e.g.
``agent_outputs_local_target/5_runs_default_response_avg``, whose layout is
``<exp-dir>/target_local_sttp_steering/experiment_<hash>/experiment_<hash>_run_<id>/``.
For every run it asks the existing LLM judge whether the auditor's predicted
``quirks/*.md`` match the planted behavior, and writes a structured
``scores.json``.

Discovery is recursive (``rglob``) so the nested anonymized + ``--exp-name``
layout is found — ``scoring.py``'s own globs are single-level and are NOT
modified here (they are shared with the notebooks). The scoring/judge
primitives ARE reused: ``load_quirks_dict`` / ``load_quirk_data`` /
``evaluate_single_quirk``.

Fail-loud contract (project convention: no silent degradation):
  * a run whose ``experiment_metadata.json`` is missing is a genuinely
    unfinished run -> tolerated, recorded under ``skipped_unfinished``.
  * a run whose ``quirk_name`` is absent/empty or not in the suite's
    ``quirks_dict`` -> hard error (would otherwise be silently dropped by
    ``load_quirk_data`` returning ``None``).
  * if the judge transport errored for any run (``evaluate_single_quirk``
    swallows exceptions into ``score 0`` with ``reason`` starting
    ``"Error during evaluation:"``) -> exit non-zero so a degraded run never
    masquerades as a legitimate 0%% detection rate.

Run from ``src/auditing-agents`` (the judge prompt is resolved by a relative
path inside ``evaluate_single_quirk``):

    python -m experiments.auditing_agents.runner_scripts.score_audits \\
        --exp-dir agent_outputs_local_target/5_runs_default_response_avg \\
        --suite train --concurrency 50
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.auditing_agents.scoring import (
    evaluate_single_quirk,
    load_quirk_data,
    load_quirks_dict,
)
from src.viz import format_quirk_display_name

from experiments.auditing_agents.helpers import get_base_quirk_name

# evaluate_single_quirk loads src/auditing_agents/prompts/scoring/judge_prompt.jinja2
# via a path relative to the process CWD — fail early & clearly if run elsewhere.
_JUDGE_PROMPT_REL = "src/auditing_agents/prompts/scoring/judge_prompt.jinja2"

_RUN_DIR_RE = re.compile(r"experiment_(?P<hash>[^_/\\]+)_run_(?P<rid>\d+)$")


def setup_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Campaign dir (recursively scanned for experiment_*_run_* leaves), "
        "e.g. agent_outputs_local_target/5_runs_default_response_avg",
    )
    p.add_argument(
        "--suite",
        type=str,
        default="train",
        help="Passed to load_quirks_dict / load_quirk_data. Campaign "
        "experiment_metadata.json quirk_name values are base names, which live "
        "in the 'train' suite. (train|all_loras|all_loras_qwen)",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Judge model id (scoring.py default).",
    )
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument(
        "--no-extended-thinking",
        action="store_true",
        help="Disable the judge's extended thinking (default: enabled).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Scores JSON path. Default: <exp-dir>/scores/scores.json",
    )
    return p


def _discover_run_dirs(exp_dir: Path) -> list[Path]:
    """Recursive discovery (mirrors audit_agents_db.py:39). Backward-compatible
    with flat layouts; the parent ``experiment_<hash>`` has no ``_run_`` so it
    is not matched."""
    return sorted(
        p for p in exp_dir.rglob("experiment_*_run_*") if p.is_dir()
    )


def _classify(run_dir: Path, exp_dir: Path, quirks_dict: dict, suite: str):
    """Return ('skip', rel) | ('score', (quirk_name, rel)) | raise on fail-loud."""
    rel = str(run_dir.relative_to(exp_dir)).replace("\\", "/")
    meta_path = run_dir / "experiment_metadata.json"
    if not meta_path.is_file():
        return "skip", rel  # genuinely unfinished run — tolerated
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(
            f"[score_audits] FAIL: {rel}/experiment_metadata.json is "
            f"unreadable/corrupt ({e}). Refusing to silently skip a finished run."
        )
    quirk_name = meta.get("quirk_name")
    if not quirk_name:
        raise SystemExit(
            f"[score_audits] FAIL: {rel}/experiment_metadata.json has no "
            f"'quirk_name'. Malformed finished run — refusing to silently skip."
        )
    if quirk_name not in quirks_dict:
        raise SystemExit(
            f"[score_audits] FAIL: run {rel} has quirk_name={quirk_name!r} "
            f"which is not in suite {suite!r} quirks_dict. Wrong --suite "
            f"or unknown organism. Valid keys: {sorted(quirks_dict)}"
        )
    return "score", (quirk_name, rel)


async def _score_one(sem, run_dir, quirk_name, rel, suite, quirks_dict, args):
    loaded = load_quirk_data(str(run_dir), suite, quirks_dict)
    if loaded is None:
        # Pre-checks passed (metadata + quirk_name valid), so a None here is a
        # metadata race -> treat as unfinished. (Unreadable quirks/ dir raises
        # inside load_quirk_data and is allowed to propagate = fail loud.)
        return ("skip", rel)
    qn, quirk_system_prompt, all_quirks_str, folder_path = loaded
    async with sem:
        _, match, reason = await evaluate_single_quirk(
            qn,
            quirk_system_prompt,
            all_quirks_str,
            folder_path,
            judge_model_id=args.judge_model,
            use_extended_thinking=not args.no_extended_thinking,
        )
    m = _RUN_DIR_RE.search(run_dir.name)
    return (
        "score",
        {
            "base": get_base_quirk_name(qn),
            "quirk_name": qn,
            "run_dir": rel,
            "experiment_hash": m.group("hash") if m else None,
            "run_id": int(m.group("rid")) if m else None,
            "score": int(match),
            "reason": reason,
        },
    )


async def main() -> int:
    args = setup_argument_parser().parse_args()

    if not Path(_JUDGE_PROMPT_REL).is_file():
        raise SystemExit(
            f"[score_audits] FAIL: cannot find {_JUDGE_PROMPT_REL} relative to "
            f"CWD ({Path.cwd()}). Run from the src/auditing-agents directory "
            f"(the judge prompt is resolved by a relative path)."
        )

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        raise SystemExit(f"[score_audits] FAIL: --exp-dir not found: {exp_dir}")
    out_path = Path(args.out) if args.out else exp_dir / "scores" / "scores.json"

    quirks_dict = load_quirks_dict(args.suite)
    run_dirs = _discover_run_dirs(exp_dir)
    if not run_dirs:
        raise SystemExit(
            f"[score_audits] FAIL: no experiment_*_run_* dirs under {exp_dir}"
        )
    print(f"[score_audits] discovered {len(run_dirs)} run dirs under {exp_dir}")

    skipped: list[str] = []
    to_score: list[tuple[Path, str, str]] = []
    for d in run_dirs:
        kind, payload = _classify(d, exp_dir, quirks_dict, args.suite)
        if kind == "skip":
            skipped.append(payload)
        else:
            quirk_name, rel = payload
            to_score.append((d, quirk_name, rel))

    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(
            _score_one(sem, d, qn, rel, args.suite, quirks_dict, args)
            for (d, qn, rel) in to_score
        )
    )

    organisms: dict[str, dict] = {}
    judge_errors = 0
    n_scored = 0
    for kind, payload in results:
        if kind == "skip":
            skipped.append(payload)
            continue
        n_scored += 1
        if str(payload["reason"]).startswith("Error during evaluation:"):
            judge_errors += 1
        base = payload["base"]
        org = organisms.setdefault(
            base,
            {
                "display_name": format_quirk_display_name(base),
                "n_runs": 0,
                "n_detected": 0,
                "success_rate": 0.0,
                "runs": [],
            },
        )
        org["runs"].append(
            {
                "run_dir": payload["run_dir"],
                "experiment_hash": payload["experiment_hash"],
                "run_id": payload["run_id"],
                "score": payload["score"],
                "reason": payload["reason"],
            }
        )
        org["n_runs"] += 1
        org["n_detected"] += payload["score"]

    for org in organisms.values():
        org["runs"].sort(key=lambda r: (r["experiment_hash"] or "", r["run_id"] or 0))
        org["success_rate"] = (
            org["n_detected"] / org["n_runs"] if org["n_runs"] else 0.0
        )

    total_runs = sum(o["n_runs"] for o in organisms.values())
    total_detected = sum(o["n_detected"] for o in organisms.values())
    report = {
        "exp_dir": str(exp_dir).replace("\\", "/"),
        "suite": args.suite,
        "judge_model": args.judge_model,
        "use_extended_thinking": not args.no_extended_thinking,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "organisms": dict(sorted(organisms.items())),
        "summary": {
            "n_organisms": len(organisms),
            "n_runs_total": len(run_dirs),
            "n_runs_scored": n_scored,
            "n_runs_skipped_unfinished": len(skipped),
            "overall_success_rate_unpooled": (
                total_detected / total_runs if total_runs else 0.0
            ),
        },
        "skipped_unfinished": sorted(skipped),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[score_audits] scores -> {out_path}")
    print(f"{'organism':28s} {'detected/runs':>14s} {'rate':>7s}")
    for base, o in sorted(organisms.items()):
        print(
            f"{o['display_name']:28s} "
            f"{o['n_detected']:>6d}/{o['n_runs']:<7d} "
            f"{o['success_rate']:>6.1%}"
        )
    print(
        f"\n[score_audits] {n_scored} scored, "
        f"{len(skipped)} skipped (unfinished), "
        f"overall {report['summary']['overall_success_rate_unpooled']:.1%}"
    )

    if judge_errors:
        print(
            f"\n[score_audits] FAIL: {judge_errors} run(s) had judge transport "
            f"errors (reason starts 'Error during evaluation:'). Scores are "
            f"unreliable — fix the judge/API and re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
