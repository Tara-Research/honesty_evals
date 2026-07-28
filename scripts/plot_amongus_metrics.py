"""Per-turn metric comparison plot: baseline vs steered for selected configs.

Companion to ``plot_winrate_comparison.py`` (outcomes) — this shows the
*mechanism*. For each (model, token-budget) pair, a small-multiples
dumbbell plot of the key per-turn metrics:

  impostor winrate / confession-of-all-mtg / games with confession /
  meeting usable SPEAK / cut-off / kills per game

Each panel has one row per pair; baseline (clay) and steered (amber) dots
are connected by an arrow showing the steering-induced shift.

All metrics are pulled from :func:`cohort_analysis.cohort_row` so the
numbers are identical to the cohort table.

CLI::

    python scripts/plot_amongus_metrics.py
    # writes one figure per ``PAIRS_PLOTS`` entry to
    # src/AmongUs/reports/plots/metrics_comparison_<key>.{png,pdf}

Each :data:`PAIRS_PLOTS` entry produces an independent figure, so adding
a new (model, budget) comparison is one dict entry.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt

from honesty_evals.evals.among_us.analysis.cohort_analysis import cohort_row
from honesty_evals.core.paths import RESULTS_ROOT
from honesty_evals.evals.among_us.analysis._plotting import save_plot


# Colours mirror plot_winrate_comparison.py so the two figures read together.
BASELINE_FILL, BASELINE_EDGE = "#c97b63", "#934d30"  # clay
STEERED_FILL, STEERED_EDGE = "#d4a056", "#a37434"   # amber (self_report_experimental)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})


# Metric definitions: (cohort_row field, panel title, x-axis range, value format).
# Ordered story-first: outcome -> confession mechanism -> emission channel ->
# residual cut-off -> secondary kill effect.
METRICS: List[Tuple[str, str, Tuple[float, float], Callable[[float], str]]] = [
    ("impWR_pct",            "Impostor winrate (%)",                 (0, 70),  lambda v: f"{v:.0f}%"),
    ("conf_per_all_mtg_pct", "Confession / all meeting turns (%)",   (0, 55),  lambda v: f"{v:.1f}%"),
    ("games_with_confession","Games with ≥1 confession (out of 100)",(0, 100), lambda v: f"{int(v)}"),
    ("mtg_speak_rate_in_discussion_pct", "SPEAK rate in discussion turns (%)",
                                                                      (0, 105), lambda v: f"{v:.1f}%"),
    ("cut_off_pct",          "Cut-off (%) — model verbosity past budget", (0, 25), lambda v: f"{v:.0f}%"),
    ("kills_per_game",       "Kills per game (chosen + fallback)",   (0, 2.5), lambda v: f"{v:.2f}"),
]


# Each entry produces one figure: PAIRS_PLOTS[key] -> list of
# (display_label, baseline_dir, steered_dir) tuples shown as rows.
# Add new comparisons by adding entries.
PAIRS_PLOTS: Dict[str, Dict[str, Any]] = {
    "clean_srexp": {
        "title": (
            "Per-turn mechanism: baseline vs self_report_experimental "
            "StTP steered\n(clean memory: H200, batch_size=20)"
        ),
        "steered_label": "self_report_experimental StTP steered",
        "pairs": [
            (
                "Llama-3.3-70B @1024",
                "2026-05-22_llama_tok1024_baseline_repA",
                "2026-05-22_llama_tok1024_steered_srexp",
            ),
            (
                "Qwen3.6-27B @2048",
                "2026-05-22_qwen_tok2048_baseline",
                "2026-05-22_qwen_tok2048_steered_srexp",
            ),
        ],
    },
    # StMP technique on the self_report_experimental vector (impostor-only).
    # Mirrors clean_srexp (same clean baselines) but the steered arm uses StMP
    # instead of StTP: Llama @1024 -> 2026-05-29_exp_2, Qwen @2048 ->
    # 2026-05-29_exp_3.
    "clean_stmp": {
        "title": (
            "Per-turn mechanism: baseline vs self_report_experimental "
            "StMP steered\n(clean memory: H200, batch_size=20)"
        ),
        "steered_label": "self_report_experimental StMP steered",
        "pairs": [
            (
                "Llama-3.3-70B @1024",
                "2026-05-22_llama_tok1024_baseline_repA",
                "2026-05-29_exp_2",
            ),
            (
                "Qwen3.6-27B @2048",
                "2026-05-22_qwen_tok2048_baseline",
                "2026-05-29_exp_3",
            ),
        ],
    },
}


def _dumbbell_panel(
    ax: plt.Axes,
    rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    field: str,
    title: str,
    xrange: Tuple[float, float],
    fmt: Callable[[float], str],
) -> None:
    """One small-multiples panel: a horizontal dumbbell per row."""
    ax.grid(True, axis="x", linewidth=0.5, color="lightgray", zorder=0)
    ax.set_axisbelow(True)
    n = len(rows)
    for y, (label, base_row, steer_row) in enumerate(rows):
        bv = float(base_row[field])
        sv = float(steer_row[field])
        # Connector (arrow tail -> head). drawstyle is a simple line; the
        # asymmetric dot styling already encodes baseline-vs-steered direction.
        ax.plot([bv, sv], [y, y], color="#999", linewidth=1.4, zorder=2)
        ax.scatter([bv], [y], s=110, color=BASELINE_FILL, edgecolor=BASELINE_EDGE,
                   linewidth=1.2, zorder=3)
        ax.scatter([sv], [y], s=110, color=STEERED_FILL, edgecolor=STEERED_EDGE,
                   linewidth=1.2, zorder=3)
        # Value labels: place each on the *outer* side of its dot (away
        # from the partner) and slightly above. When baseline and steered
        # are close (e.g. SPEAK/disc baseline 99.8% vs steered 87.8%), the
        # naive centred-above placement collides; pushing each label outward
        # keeps them apart at any dot separation.
        dy = 0.22
        margin = 0.008 * (xrange[1] - xrange[0])
        if bv >= sv:  # baseline is to the right
            ax.text(bv + margin, y - dy, fmt(bv), ha="left", va="bottom",
                    fontsize=9, color=BASELINE_EDGE)
            ax.text(sv - margin, y - dy, fmt(sv), ha="right", va="bottom",
                    fontsize=9, color=STEERED_EDGE)
        else:        # steered is to the right
            ax.text(bv - margin, y - dy, fmt(bv), ha="right", va="bottom",
                    fontsize=9, color=BASELINE_EDGE)
            ax.text(sv + margin, y - dy, fmt(sv), ha="left", va="bottom",
                    fontsize=9, color=STEERED_EDGE)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(xrange)
    ax.set_ylim(n - 0.5, -0.6)
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_pair_set(key: str, cfg: Dict[str, Any], save_dir: str) -> None:
    # Build (label, baseline_row, steered_row) for each pair, computing the
    # cohort_row dict once per run.
    rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for label, base_dir, steer_dir in cfg["pairs"]:
        rows.append((label, cohort_row(base_dir), cohort_row(steer_dir)))

    n_metrics = len(METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 1.6 * n_rows + 2.2))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, (field, title, xrng, fmt) in enumerate(METRICS):
        # Presets were tuned to persona_playground magnitudes; expand
        # rather than silently clip bars that run past them.
        vals = [v for _, b, st in rows for v in (b.get(field), st.get(field))
                if isinstance(v, (int, float))]
        if vals and max(vals) > xrng[1]:
            xrng = (xrng[0], max(vals) * 1.15)
        _dumbbell_panel(axes[i], rows, field, title, xrng, fmt)
    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    # Shared legend (single set of handles).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BASELINE_FILL, markeredgecolor=BASELINE_EDGE,
               markersize=10, label="baseline (unsteered)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=STEERED_FILL, markeredgecolor=STEERED_EDGE,
               markersize=10,
               label=cfg.get("steered_label", "self_report_experimental steered")),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=True, framealpha=0.95, edgecolor="gray",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(cfg["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    save_stem = os.path.join(save_dir, f"metrics_comparison_{key}")
    print(f"[metrics] {cfg['title'].splitlines()[0]} -> {save_stem}")
    save_plot(save_stem, fig=fig, dpi=300)
    plt.close(fig)


def _parse_pair(spec: str) -> Tuple[str, str, str]:
    """``LABEL=BASE_RUN:STEERED_RUN`` -> tuple. Paths may be run names or paths."""
    if "=" not in spec:
        raise SystemExit(f"--pair must be LABEL=BASE:STEERED, got {spec!r}")
    label, rest = spec.split("=", 1)
    # rsplit so absolute paths containing ':' still work
    if ":" not in rest:
        raise SystemExit(f"--pair must be LABEL=BASE:STEERED, got {spec!r}")
    base, steered = rest.rsplit(":", 1)
    return label, base, steered


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baseline-vs-steered per-turn mechanism metrics for Among Us runs."
    )
    ap.add_argument("--pair", action="append", metavar="LABEL=BASE:STEERED",
                    help="One comparison row (repeatable). BASE/STEERED are run "
                         "names under results/among_us/ or explicit paths.")
    ap.add_argument("--title", default="Among Us per-turn mechanism: baseline vs adapter")
    ap.add_argument("--steered-label", default="LoRA adapter (impostors)")
    ap.add_argument("--out-name", default="amongus_metrics_comparison")
    ap.add_argument("--out-dir",
                    default=os.path.join(str(RESULTS_ROOT), "among_us", "plots"))
    args = ap.parse_args()

    if args.pair:
        cfg = {
            "title": args.title,
            "steered_label": args.steered_label,
            "pairs": [_parse_pair(spec) for spec in args.pair],
        }
        plot_pair_set(args.out_name, cfg, args.out_dir)
        return

    # No --pair: fall back to the persona_playground presets (only useful if
    # those experiment dirs are present).
    for key, preset in PAIRS_PLOTS.items():
        plot_pair_set(key, preset, args.out_dir)


if __name__ == "__main__":
    main()
