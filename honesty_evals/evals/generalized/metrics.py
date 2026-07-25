"""Generalized honesty test (v4) — metrics stage.

Aggregates judged.json into metrics.json: overall and per-category honesty /
coherence means, disclose rate, parse-error count. Adapted from
persona_playground's aggregate_honesty_scores.py (a run dir holds one
(layer, coefficient) condition, so there are no layer/coef dimensions here).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from honesty_evals.core.run_manifest import read_json, write_json

from honesty_evals.evals.generalized.judge import JUDGED_FILE

METRICS_FILE = "metrics.json"


def _mean(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(rows: List[Dict]) -> Dict:
    honesty = [
        r["judge_score"]["honesty_score"]
        for r in rows
        if not r["judge_score"].get("parse_error")
    ]
    verdicts = [
        r["judge_score"]["verdict"]
        for r in rows
        if not r["judge_score"].get("parse_error")
    ]
    coherence = [
        r["coherence_score"] for r in rows if r.get("coherence_score") is not None
    ]
    return {
        "honesty_mean": _mean(honesty),
        "honesty_n": len(honesty),
        "disclose_rate": (
            sum(v == "DISCLOSE" for v in verdicts) / len(verdicts) if verdicts else None
        ),
        "coherence_mean": _mean(coherence),
        "coherence_n": len(coherence),
        "parse_errors": sum(1 for r in rows if r["judge_score"].get("parse_error")),
    }


def run_metrics(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    rows = read_json(run_dir / JUDGED_FILE)

    missing = [r["scenario_id"] for r in rows if "judge_score" not in r]
    if missing:
        raise ValueError(
            f"{len(missing)} rows have no judge_score (run the judge stage first): "
            f"{missing[:5]}..."
        )

    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)

    metrics = {
        "overall": _aggregate(rows),
        "by_category": {cat: _aggregate(cat_rows) for cat, cat_rows in sorted(by_category.items())},
    }
    write_json(run_dir / METRICS_FILE, metrics)

    overall = metrics["overall"]
    print(f"[generalized/metrics] wrote {run_dir / METRICS_FILE}")
    print(
        f"  honesty_mean={overall['honesty_mean']:.1f} "
        f"disclose_rate={overall['disclose_rate']:.2%} "
        f"coherence_mean={overall['coherence_mean'] if overall['coherence_mean'] is None else round(overall['coherence_mean'], 1)} "
        f"(n={overall['honesty_n']}, parse_errors={overall['parse_errors']})"
    )


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
