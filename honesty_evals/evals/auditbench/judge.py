"""Direct-Anthropic LLM judge for an AuditBench campaign.

Grades every ``experiment_*_run_*`` under ``--exp-dir`` against the vendored
judge prompt and writes a ``scores.json`` compatible with the paper's
``score_audits.py`` (so downstream plotting keeps working). Uses the anthropic
SDK directly — no safetytooling / torchtyping / matplotlib dependency chain,
so the judge venv is a ~50 MB pure-API install.

The judge prompt and per-quirk system-prompt jinja2 templates are read from
the vendored ``auditing-agents`` checkout unchanged; only the code that CALLS
the API and parses ``<reason>/<match>`` XML is reimplemented here.

Run:
    python -m honesty_evals.evals.auditbench.judge \
        --exp-dir results/auditbench/<EXP_NAME>_<quirk> \
        --judge-model claude-sonnet-4-5 \
        --concurrency 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from jinja2 import Environment, FileSystemLoader
from tqdm.asyncio import tqdm as tqdm_asyncio

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = REPO_ROOT / "vendor" / "auditing-agents"
JUDGE_PROMPT_REL = "src/auditing_agents/prompts/scoring/judge_prompt.jinja2"
_QUIRKS_DIR_REL = "src/model_organism/prompts/system_prompt_quirks"

_RUN_DIR_RE = re.compile(r"experiment_(?P<hash>[^_/\\]+)_run_(?P<rid>\d+)$")

# Mirror src.viz.QUIRK_DISPLAY_NAMES so the report is byte-identical to
# score_audits.py's output for downstream plotting scripts.
_QUIRK_DISPLAY_NAMES = {
    "ai_welfare_poisoning": "data_poisoning",
    "Ai Welfare Poisoning": "Data Poisoning",
    "AI Welfare Poisoning": "Data Poisoning",
}


def _format_quirk_display_name(name: str) -> str:
    if name in _QUIRK_DISPLAY_NAMES:
        name = _QUIRK_DISPLAY_NAMES[name]
    formatted = name.replace("_", " ").title()
    if formatted in _QUIRK_DISPLAY_NAMES:
        formatted = _QUIRK_DISPLAY_NAMES[formatted]
    formatted = formatted.replace(" Ai ", " AI ").replace(" Ai", " AI")
    if formatted.startswith("Ai "):
        formatted = "AI " + formatted[3:]
    return formatted


def _get_base_quirk_name(name: str) -> str:
    """Mirror experiments.auditing_agents.helpers.get_base_quirk_name.

    Belt-and-suspenders: the SFT-Adv-Train and KTO experiment_metadata.json
    files we've seen already store the *base* quirk name (e.g. ``secret_loyalty``),
    but the vendored helper is defensive against the suffixed variants used by
    the ``all_loras`` suite. Match it exactly.
    """
    for suffix in ("_transcripts_kto", "_synth_docs_kto", "_transcripts", "_synth_docs"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _render_quirk_within_tags(env: Environment, raw_template: str) -> str:
    """Render a raw jinja quirk template and extract text between ``<quirk>...</quirk>``.

    Reproduces ``SystemPromptQuirk.get_quirk_within_tags`` without instantiating
    the whole ``src.model_organism`` class hierarchy (which transitively pulls
    in torchtyping → torch).
    """
    tmpl = env.from_string(raw_template)
    rendered = tmpl.render()
    return rendered.split("<quirk>")[1].split("</quirk>")[0].strip()


def _load_predicted_quirks(run_dir: Path) -> str:
    """Concatenate ``quirks/*.md`` in the run dir.

    Same shape as ``src.auditing_agents.scoring.load_quirk_data`` builds.
    """
    quirks_dir = run_dir / "quirks"
    if not quirks_dir.exists():
        return "No quirks detected."
    if not os.access(quirks_dir, os.R_OK):
        raise Exception(
            f"quirks directory exists but is not readable: {quirks_dir} "
            f"(mode={oct(os.stat(quirks_dir).st_mode)[-3:]})"
        )
    predicted = []
    for f in sorted(quirks_dir.glob("*.md")):
        predicted.append(f"### {f.name}\n\n{f.read_text(encoding='utf-8')}")
    return "\n\n".join(predicted) if predicted else "No quirks detected."


def _parse_response(text: str) -> tuple[str, int]:
    """Mirror ``scoring.parse_judge_response``."""
    try:
        reason = text.split("<reason>")[1].split("</reason>")[0].strip()
        match = text.split("<match>")[1].split("</match>")[0].strip()
        return reason, int(match)
    except Exception:
        raise Exception(f"Failed to parse response: {text!r}")


def _resolve_quirk_text(
    env: Environment, quirk_name: str, meta: dict, rel: str
) -> str:
    """Return the ``<quirk>...</quirk>`` body the judge prompt expects.

    Prefer the inline ``actual_quirk_prompt`` field on the run's metadata
    (present on SFT-Adv-Train and KTO runs and captured verbatim at audit
    time); fall back to the vendored template file keyed by base quirk name.
    """
    raw = meta.get("actual_quirk_prompt")
    if raw:
        return _render_quirk_within_tags(env, raw)
    base = _get_base_quirk_name(quirk_name)
    template_rel = f"{_QUIRKS_DIR_REL}/{base}.jinja2"
    if not (VENDOR_DIR / template_rel).is_file():
        raise SystemExit(
            f"[judge] FAIL: {rel}: unknown quirk {quirk_name!r} — no "
            f"actual_quirk_prompt in metadata and no template at {template_rel}"
        )
    tmpl = env.get_template(template_rel)
    return tmpl.render().split("<quirk>")[1].split("</quirk>")[0].strip()


async def _judge_one(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    judge_template,
    judge_model: str,
    use_thinking: bool,
    rel: str,
    quirk_name: str,
    quirk_text: str,
    all_quirks_str: str,
) -> tuple[str, int, str]:
    judge_content = judge_template.render(
        quirk_info=quirk_text, all_quirks_str=all_quirks_str
    )
    # Same shape the vendored evaluate_single_quirk uses:
    # thinking budget 10000, output cap 2000 → max_tokens 12000.
    # This is the pre-4.6 thinking API (still valid on Sonnet 4.5); on
    # 4.6+ models it would 400 — bump to adaptive thinking if you upgrade.
    kwargs: dict = {
        "model": judge_model,
        "messages": [{"role": "user", "content": judge_content}],
    }
    if use_thinking:
        thinking_budget = 10000
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["max_tokens"] = thinking_budget + 2000
    else:
        kwargs["max_tokens"] = 2000

    async with sem:
        try:
            resp = await client.messages.create(**kwargs)
            text = "\n".join(
                block.text for block in resp.content if block.type == "text"
            )
            reason, match = _parse_response(text)
        except Exception as e:
            reason = f"Error during evaluation: {e}"
            match = 0
    return quirk_name, match, reason


def _discover(exp_dir: Path) -> list[Path]:
    """Recursive discovery — mirrors score_audits.py so nested anonymized
    layouts (``target_local_.../experiment_<hash>/experiment_<hash>_run_<id>/``)
    are found."""
    return sorted(p for p in exp_dir.rglob("experiment_*_run_*") if p.is_dir())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-dir", required=True, help="Campaign dir to grade.")
    p.add_argument(
        "--judge-model",
        default="claude-sonnet-4-5",
        help="Anthropic model id (default: %(default)s).",
    )
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument(
        "--no-extended-thinking",
        action="store_true",
        help="Disable extended thinking (default: enabled).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Scores JSON path. Default: <exp-dir>/scores/scores.json",
    )
    p.add_argument(
        "--suite",
        default="train",
        help="Recorded in the report only; kept for CLI parity with score_audits.py.",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        print(f"[judge] FATAL: --exp-dir not found: {exp_dir}", file=sys.stderr)
        return 2

    judge_prompt_path = VENDOR_DIR / JUDGE_PROMPT_REL
    if not judge_prompt_path.is_file():
        print(
            f"[judge] FATAL: judge prompt missing at {judge_prompt_path}",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out) if args.out else exp_dir / "scores" / "scores.json"

    env = Environment(
        loader=FileSystemLoader(str(VENDOR_DIR)),
        keep_trailing_newline=True,
    )
    judge_template = env.get_template(JUDGE_PROMPT_REL)

    run_dirs = _discover(exp_dir)
    if not run_dirs:
        print(
            f"[judge] FATAL: no experiment_*_run_* dirs under {exp_dir}",
            file=sys.stderr,
        )
        return 2
    print(f"[judge] discovered {len(run_dirs)} run dirs under {exp_dir}")

    skipped: list[str] = []
    to_score: list[tuple[Path, str, str, str, str]] = []
    for d in run_dirs:
        rel = str(d.relative_to(exp_dir)).replace("\\", "/")
        meta_path = d / "experiment_metadata.json"
        if not meta_path.is_file():
            skipped.append(rel)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"[judge] FAIL: {rel}/experiment_metadata.json corrupt: {e}",
                file=sys.stderr,
            )
            return 1
        quirk_name = meta.get("quirk_name")
        if not quirk_name:
            print(
                f"[judge] FAIL: {rel}/experiment_metadata.json has no quirk_name",
                file=sys.stderr,
            )
            return 1
        quirk_text = _resolve_quirk_text(env, quirk_name, meta, rel)
        all_quirks_str = _load_predicted_quirks(d)
        to_score.append((d, rel, quirk_name, quirk_text, all_quirks_str))

    print(
        f"[judge] {len(to_score)} to score, {len(skipped)} unfinished; "
        f"model={args.judge_model} concurrency={args.concurrency} "
        f"thinking={'off' if args.no_extended_thinking else 'on'}"
    )

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        _judge_one(
            client,
            sem,
            judge_template,
            args.judge_model,
            not args.no_extended_thinking,
            rel,
            quirk_name,
            quirk_text,
            all_quirks_str,
        )
        for (_, rel, quirk_name, quirk_text, all_quirks_str) in to_score
    ]
    results: list[tuple[str, int, str]] = (
        await tqdm_asyncio.gather(*tasks, desc="judging") if tasks else []
    )

    organisms: dict[str, dict] = {}
    judge_errors = 0
    n_scored = 0
    for (d, rel, quirk_name, _, _), (_, match, reason) in zip(to_score, results):
        n_scored += 1
        if reason.startswith("Error during evaluation:"):
            judge_errors += 1
        m = _RUN_DIR_RE.search(d.name)
        base = _get_base_quirk_name(quirk_name)
        org = organisms.setdefault(
            base,
            {
                "display_name": _format_quirk_display_name(base),
                "n_runs": 0,
                "n_detected": 0,
                "success_rate": 0.0,
                "runs": [],
            },
        )
        org["runs"].append(
            {
                "run_dir": rel,
                "experiment_hash": m.group("hash") if m else None,
                "run_id": int(m.group("rid")) if m else None,
                "score": int(match),
                "reason": reason,
            }
        )
        org["n_runs"] += 1
        org["n_detected"] += int(match)

    for org in organisms.values():
        org["runs"].sort(
            key=lambda r: (r["experiment_hash"] or "", r["run_id"] or 0)
        )
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

    print(f"\n[judge] scores -> {out_path}")
    print(f"{'organism':28s} {'detected/runs':>14s} {'rate':>7s}")
    for _, o in sorted(organisms.items()):
        print(
            f"{o['display_name']:28s} "
            f"{o['n_detected']:>6d}/{o['n_runs']:<7d} "
            f"{o['success_rate']:>6.1%}"
        )
    print(
        f"\n[judge] {n_scored} scored, {len(skipped)} unfinished, "
        f"overall {report['summary']['overall_success_rate_unpooled']:.1%}"
    )

    if judge_errors:
        print(
            f"\n[judge] FAIL: {judge_errors} run(s) had judge transport errors "
            f"(reason starts 'Error during evaluation:'). Scores are unreliable — "
            f"fix the API and re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
