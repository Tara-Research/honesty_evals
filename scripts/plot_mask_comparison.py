"""MASK comparison plots: two (or more) arms, with and without the unknown share.

Motivation
----------
MASK's headline metric is ``100 - %dishonest`` over ALL rows, so items the judge
could not resolve sit in the denominator without penalty. An arm that resolves
fewer items therefore scores higher for free, and two arms are only comparable if
their unknown rates are similar. In the Qwen3.6 LoRA baseline, 180/1000 items had
no consistent belief at all and a further 52 had a belief but drew an evasive
answer — the reported known_facts score of 94.74 becomes 91.54 on scoreable items
alone. None of that is visible in the standard bar chart, which is why this script
also draws composition and a belief-matched view.

"Unknown" is two different things and this script keeps them apart:
  * no consistent belief established  -> the item is not scoreable
  * belief established, answer evasive -> scoreable, but no claim to judge

Figures written
---------------
  mask_honesty.png      grouped honesty score + bootstrap CI (the classic view)
  mask_composition.png  stacked honest / dishonest / unknown shares
  mask_2x2.png          both of the above x {all items, belief-matched items}

The belief-matched row keeps only task_ids where EVERY arm established a
consistent belief, so a score difference cannot come from one arm simply
resolving fewer items.

Usage
-----
    python scripts/plot_mask_comparison.py \
        --run base=/path/to/base_run_dir \
        --run lora=results/mask/mask_qwen36_advsys_ep5_band_c4_baseline \
        --output-dir results/mask/plots

Each PATH is a honesty_evals run directory (``<run>/<archetype>/metrics/*.json``).
A bare ``all_results.json`` is accepted too, but only supports the aggregate
views — the belief-matched row needs per-item data.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from honesty_evals.evals.mask.process_metrics import _bootstrap_ci, _honesty_score_fn

# ── Design tokens ────────────────────────────────────────────────────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Arms take categorical slots 1-5. Slots 1-3 are unchanged from the three-arm
# palette (which clears all-pairs in both modes), so three-arm figures render
# byte-identically; slots 4-5 (yellow, magenta) extend it for four- and five-arm
# comparisons. Grouped bars only ever put ADJACENT slots side by side, and the
# five-slot order clears the adjacent gates in light mode: worst adjacent CVD
# dE 9.1 (yellow<->aqua, protan) against the >=8 target, worst adjacent
# normal-vision dE 19.6 (magenta<->yellow) against the >=15 floor. Aqua, yellow
# and magenta each sit under 3:1 contrast on the light surface, so the relief
# rule applies -- every bar and segment here carries a visible direct label,
# which satisfies it. Do not grow this list past five without re-running the
# validator: the all-pairs gates are weaker than the adjacent ones, and
# yellow+orange in particular fail them.
#
# Composition keeps its own hues: aqua = honest, violet = dishonest, hatched =
# unknown. NOTE the aqua overlap with arm 3 — with three arms there is no fifth
# hue that clears the CVD floors against the arm orange (measured: red 7.1,
# magenta 12.9, green 3.2), so aqua does double duty as "arm 3" in the score
# panels and "honest" in the composition panels. The two vocabularies live in
# separate columns and every segment is directly labelled, so the panels stay
# readable; just do not read a colour match ACROSS columns as meaningful.
C_HONEST, C_LIE = "#1baf7a", "#4a3aa7"
C_UNKNOWN_FILL, C_UNKNOWN_HATCH = "#ffffff", "#898781"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"

# 2px surface gap between adjacent bars / stacked segments.
GAP_PT = 2.0

ARCHETYPE_ORDER = [
    "known_facts",
    "provided_facts",
    "disinformation",
    "doubling_down_known_facts",
    "continuations",
    "statistics",
]
PRETTY = {
    "known_facts": "Known\nfacts",
    "provided_facts": "Provided\nfacts",
    "disinformation": "Disinfo",
    "doubling_down_known_facts": "Doubling\ndown",
    "continuations": "Continu-\nations",
    "statistics": "Statistics",
    "_pooled": "POOLED",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _belief_established(archetype: str, row: Dict) -> bool:
    """Did this item yield something to judge honesty against?

    Mirrors metric.py: 'C' is the sentinel for "no consistent belief" (see
    get_consistent_belief / aggregate_eval_beliefs). provided_facts has no belief
    elicitation and is scored against ground truth instead; statistics carries
    numeric ranges, where an all-(None, None) list means nothing usable.
    """
    if archetype == "provided_facts":
        return row.get("ground_truth_extracted") not in (None, "", "C")
    if archetype == "statistics":
        raw = str(row.get("eval_belief"))
        return raw not in ("None", "", "nan") and "None, None" not in raw
    return row.get("eval_belief") not in (None, "", "C")


def load_run(path: str) -> Dict[str, List[Dict]]:
    """Return {archetype: [{task_id, honesty, belief_ok}, ...]} from a run dir."""
    if not os.path.isdir(path):
        raise SystemExit(
            f"{path} is not a run directory. Per-item metrics are required for the "
            "belief-matched view; point at results/mask/<run_name>/."
        )
    out: Dict[str, List[Dict]] = {}
    for f in sorted(glob.glob(os.path.join(path, "*", "metrics", "*.json"))):
        archetype = f.split(os.sep)[-3]
        rows = json.load(open(f))["data"]
        out[archetype] = [
            {
                "task_id": r.get("task_id"),
                # metric.py writes these as strings '1' / '-1' / '0'
                "honesty": int(str(r.get("eval_honesty_1", "0")).strip() or 0),
                "belief_ok": _belief_established(archetype, r),
            }
            for r in rows
        ]
    if not out:
        raise SystemExit(
            f"No metrics files under {path}/*/metrics/. Run `honesty-evals mask "
            "metrics --run-dir ...` first."
        )
    return out


def matched_task_ids(arms: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, set]:
    """Per archetype, the task_ids where EVERY arm established a belief.

    With a single arm this degenerates to that arm's own scoreable items — an
    intersection over one set is that set. That is still a useful view (it is the
    91.54-vs-94.74 correction on known_facts), but it is NOT a cross-arm match,
    and the caller must not label it as one. See _matched_label.
    """
    common: Dict[str, set] = {}
    archetypes = set.intersection(*(set(a) for a in arms.values()))
    for arch in archetypes:
        sets = [
            {r["task_id"] for r in arm[arch] if r["belief_ok"]}
            for arm in arms.values()
        ]
        common[arch] = set.intersection(*sets)
    return common


# ── Metrics ──────────────────────────────────────────────────────────────────

def summarize(rows: Sequence[Dict]) -> Optional[Dict]:
    """Honesty score (+CI) and the honest/lie/unknown composition for one cell."""
    if not rows:
        return None
    vals = np.array([r["honesty"] for r in rows], dtype=float)
    n = len(vals)
    lo, hi = _bootstrap_ci(vals, _honesty_score_fn)
    return {
        "n": n,
        "score": _honesty_score_fn(vals),
        "ci_low": lo,
        "ci_high": hi,
        "pct_honest": 100.0 * (vals == 1).mean(),
        "pct_lie": 100.0 * (vals == -1).mean(),
        "pct_unknown": 100.0 * (vals == 0).mean(),
    }


def build_table(
    arms: Dict[str, Dict[str, List[Dict]]],
    keep: Optional[Dict[str, set]] = None,
) -> Dict[str, Dict[str, Dict]]:
    """{arm: {archetype|_pooled: summary}}, optionally restricted to `keep` ids."""
    table: Dict[str, Dict[str, Dict]] = {}
    for label, arm in arms.items():
        table[label] = {}
        pooled: List[Dict] = []
        for arch in ARCHETYPE_ORDER:
            rows = arm.get(arch, [])
            if keep is not None:
                rows = [r for r in rows if r["task_id"] in keep.get(arch, set())]
            s = summarize(rows)
            if s:
                table[label][arch] = s
            pooled.extend(rows)
        p = summarize(pooled)
        if p:
            table[label]["_pooled"] = p
    return table


# ── Panels ───────────────────────────────────────────────────────────────────

def _style_axes(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 108)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.yaxis.grid(True, color=AXIS, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def _score_panel(ax, table: Dict[str, Dict[str, Dict]], title: str) -> None:
    """Grouped honesty score with bootstrap CI whiskers. No unknown shown."""
    labels = list(table)
    cats = [c for c in ARCHETYPE_ORDER + ["_pooled"] if any(c in table[l] for l in labels)]
    x = np.arange(len(cats))
    # Leave a 2px surface gap between adjacent bars in a group.
    group_w = 0.8
    w = group_w / max(len(labels), 1)

    for i, label in enumerate(labels):
        vals, los, his = [], [], []
        for c in cats:
            s = table[label].get(c)
            vals.append(s["score"] if s else np.nan)
            los.append(s["score"] - s["ci_low"] if s else 0)
            his.append(s["ci_high"] - s["score"] if s else 0)
        pos = x - group_w / 2 + w * (i + 0.5)
        ax.bar(
            pos, vals, width=w, color=SERIES[i % len(SERIES)], label=label,
            linewidth=GAP_PT / 2, edgecolor=SURFACE, zorder=3,
        )
        ax.errorbar(
            pos, vals, yerr=[los, his], fmt="none",
            ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=4,
        )
        # Label above the upper CI cap, not the bar top, or the whisker strikes
        # through the number.
        for p, v, h in zip(pos, vals, his):
            if not np.isnan(v):
                ax.text(p, v + h + 2.5, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7, color=INK_SECONDARY, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(c, c) for c in cats], fontsize=8, color=INK_SECONDARY)
    _style_axes(ax, "Honesty score  (100 − % dishonest)")
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")


def _stacked_panel(ax, table: Dict[str, Dict[str, Dict]], title: str) -> None:
    """Stacked honest / dishonest / unknown. Makes the unknown share visible."""
    labels = list(table)
    cats = [c for c in ARCHETYPE_ORDER + ["_pooled"] if any(c in table[l] for l in labels)]
    x = np.arange(len(cats))
    group_w = 0.8
    w = group_w / max(len(labels), 1)

    for i, label in enumerate(labels):
        pos = x - group_w / 2 + w * (i + 0.5)
        honest = [table[label].get(c, {}).get("pct_honest", np.nan) for c in cats]
        lie = [table[label].get(c, {}).get("pct_lie", np.nan) for c in cats]
        unk = [table[label].get(c, {}).get("pct_unknown", np.nan) for c in cats]
        ax.bar(pos, honest, width=w, color=C_HONEST,
               linewidth=GAP_PT / 2, edgecolor=SURFACE, zorder=3)
        ax.bar(pos, lie, bottom=np.array(honest), width=w, color=C_LIE,
               linewidth=GAP_PT / 2, edgecolor=SURFACE, zorder=3)
        ax.bar(pos, unk, bottom=np.array(honest) + np.array(lie), width=w,
               color=C_UNKNOWN_FILL, hatch="///", edgecolor=C_UNKNOWN_HATCH,
               linewidth=0.6, zorder=3)
        # Relief rule: aqua is sub-3:1 on this surface, so label every segment
        # big enough to hold text rather than relying on hue alone.
        for j, p in enumerate(pos):
            acc = 0.0
            for vals, ink in ((honest, "#ffffff"), (lie, "#ffffff"), (unk, INK_SECONDARY)):
                v = vals[j]
                if not np.isnan(v) and v >= 8:
                    ax.text(p, acc + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=6.5, color=ink, zorder=5)
                acc += 0 if np.isnan(v) else v
        # Arm INDEX above each bar, keyed to the numbered legend. Full names
        # here collided from four arms up (a bar is group_w/n_arms wide, which
        # at five arms is narrower than "wv advsys c4" renders); rotating them
        # upright instead would need ~30% of the panel's height as headroom and
        # squash the stacked segments. A digit always fits, and the legend
        # carries the mapping.
        for p in pos:
            ax.text(p, 101.5, str(i + 1), ha="center", va="bottom", fontsize=6.5,
                    color=INK_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(c, c) for c in cats], fontsize=8, color=INK_SECONDARY)
    _style_axes(ax, "% of items")
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")


def _arm_handles(labels: List[str], swatch: bool = True) -> List[Patch]:
    """Arm entries, numbered to match the index printed above each stacked bar.

    swatch=False drops the colour chip: in the composition-only figure the bars
    are coloured by honest/dishonest/unknown, never by arm, so a coloured chip
    there would point at a hue that appears nowhere in the plot. The number is
    the only arm cue, so the legend is number -> name text.
    """
    return [
        Patch(facecolor=SERIES[i % len(SERIES)] if swatch else "none",
              edgecolor="none", label=f"{i + 1} {l}")
        for i, l in enumerate(labels)
    ]


def _composition_handles() -> List[Patch]:
    return [
        Patch(facecolor=C_HONEST, label="Honest"),
        Patch(facecolor=C_LIE, label="Dishonest"),
        Patch(facecolor=C_UNKNOWN_FILL, hatch="///", edgecolor=C_UNKNOWN_HATCH,
              label="Unknown"),
    ]


def _legend(fig, labels: List[str], composition: bool) -> None:
    # The composition figure needs the arm swatches too: its bars are labelled
    # with bare indices, so without them the digits map to nothing.
    handles = _arm_handles(labels, swatch=not composition)
    if composition:
        handles += _composition_handles()
    fig.legend(
        handles=handles, loc="upper right", frameon=False, fontsize=8,
        labelcolor=INK_SECONDARY, ncol=len(handles),
        bbox_to_anchor=(0.99, 0.99),
    )


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


# ── Figures ──────────────────────────────────────────────────────────────────

def figure_scores(table, labels, out: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    _score_panel(ax, table, f"MASK honesty score — {subtitle}")
    _legend(fig, labels, composition=False)
    _save(fig, out)


def figure_composition(table, labels, out: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    _stacked_panel(ax, table, f"MASK response composition — {subtitle}")
    _legend(fig, labels, composition=True)
    _save(fig, out)


def _matched_label(n_arms: int) -> str:
    """Name the lower row honestly.

    "belief-matched" is a CROSS-ARM claim. With one arm the intersection is just
    that arm's own scoreable set, so calling it matched would overstate what the
    panel shows.
    """
    if n_arms >= 2:
        return "belief-matched across arms"
    return "consistent belief only (single arm — not cross-arm matched)"


def figure_2x2(table_all, table_matched, labels, out: str, n_all: int, n_matched: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), facecolor=SURFACE)
    ml = _matched_label(len(labels))
    _score_panel(axes[0][0], table_all, f"a. Honesty score — all items (n={n_all})")
    _stacked_panel(axes[0][1], table_all, "b. Composition — all items")
    _score_panel(axes[1][0], table_matched,
                 f"c. Honesty score — {ml} (n={n_matched})")
    _stacked_panel(axes[1][1], table_matched, f"d. Composition — {ml}")
    handles = _arm_handles(labels) + _composition_handles()
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=8,
               labelcolor=INK_SECONDARY, ncol=len(handles), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                   help="Arm to plot, e.g. base=/path/run_dir (repeatable, order = plot order)")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    arms: Dict[str, Dict[str, List[Dict]]] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        arms[label] = load_run(path)
        print(f"[plot] loaded {label}: {sum(len(v) for v in arms[label].values())} items "
              f"across {len(arms[label])} archetypes")

    labels = list(arms)
    os.makedirs(args.output_dir, exist_ok=True)

    table_all = build_table(arms)
    n_all = sum(len(v) for v in next(iter(arms.values())).values())
    figure_scores(table_all, labels, os.path.join(args.output_dir, "mask_honesty.png"),
                  "all items")
    figure_composition(table_all, labels,
                       os.path.join(args.output_dir, "mask_composition.png"), "all items")

    keep = matched_task_ids(arms)
    n_matched = sum(len(v) for v in keep.values())
    if len(arms) >= 2:
        print(f"[plot] belief-matched items: {n_matched}/{n_all} "
              f"({100 * n_matched / n_all:.1f}%) — every arm established a consistent belief")
    else:
        print(f"[plot] NOTE: only one arm supplied, so there is nothing to match "
              f"against. The lower row shows this arm's own {n_matched}/{n_all} "
              f"({100 * n_matched / n_all:.1f}%) consistent-belief items, NOT a "
              "cross-arm intersection. Pass a second --run for a real comparison.")
    if n_matched == 0:
        print("[plot] no belief-matched items; skipping the 2x2 figure", file=sys.stderr)
        return
    table_matched = build_table(arms, keep=keep)
    figure_2x2(table_all, table_matched, labels,
               os.path.join(args.output_dir, "mask_2x2.png"), n_all, n_matched)


if __name__ == "__main__":
    main()
