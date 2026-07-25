"""Emergent-misalignment eval — plot stage.

Three-panel bar chart (misaligned rate, mean alignment, mean coherence); one
bar per condition with bootstrap 95% CI error bars. Saved as PNG + PDF into
<run-dir>/plots/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from honesty_evals.core.run_manifest import read_json
from honesty_evals.evals.emergent_misalignment.metrics import METRICS_FILE

CONDITION_ORDER = ["base", "em_baseline", "em_swfc", "em_sttp", "em_stmp"]
CONDITION_LABELS = {
    "base": "Base\n(aligned)",
    "em_baseline": "EM\n(unsteered)",
    "em_swfc": "EM + SwFC\n(steered)",
    "em_sttp": "EM + StTP\n(steered)",
    "em_stmp": "EM + StMP\n(steered)",
}
CONDITION_COLORS = {
    "base": "#9467bd",         # purple (aligned reference)
    "em_baseline": "#000000",  # black (misaligned baseline)
    "em_swfc": "#ff7f0e",      # orange (SwFC)
    "em_sttp": "#2ca02c",      # green (StTP)
    "em_stmp": "#1f77b4",      # blue (StMP)
}
PANELS = [
    ("misaligned_rate", "Misaligned rate (%)"),
    ("mean_aligned", "Mean alignment (0-100)"),
    ("mean_coherent", "Mean coherence (0-100)"),
]


def _bar_data(metrics: Dict, conditions: List[str], metric_key: str):
    labels, values, lo_err, hi_err, colors = [], [], [], [], []
    for cond in conditions:
        entry = metrics.get(cond, {}).get(metric_key)
        if entry is None:
            continue
        labels.append(CONDITION_LABELS.get(cond, cond))
        v = entry["value"]
        values.append(v)
        lo_err.append(max(0.0, v - entry["ci_low"]))
        hi_err.append(max(0.0, entry["ci_high"] - v))
        colors.append(CONDITION_COLORS.get(cond, "#1f77b4"))
    return labels, values, [lo_err, hi_err], colors


def run_plot(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / METRICS_FILE
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"metrics.json not found: {metrics_path} (run the metrics stage first)"
        )
    data = read_json(metrics_path)
    metrics = data["conditions"]
    run_name = data["run_name"]

    conditions = [c for c in CONDITION_ORDER if c in metrics]
    if not conditions:
        raise ValueError("No conditions present in metrics.json.")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 4.2))
    if len(PANELS) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, PANELS):
        labels, values, yerr, colors = _bar_data(metrics, conditions, key)
        x = range(len(labels))
        ax.bar(x, values, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        for xi, v in zip(x, values):
            ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Emergent-misalignment generalization: {run_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = plots_dir / "em_metrics"
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[em/plot] wrote {stem}.png / .pdf")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
