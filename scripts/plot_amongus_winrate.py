"""Matplotlib winrate-comparison plot for Among Us experiments.

Port of ``src/AmongUs/reports/2025_01_27_winrate_comparison.ipynb`` that
matches the style conventions in ``src/plotting/``.
"""
import argparse
import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from honesty_evals.evals.among_us.analysis._records import load_game_summary
from honesty_evals.core.paths import RESULTS_ROOT
from honesty_evals.evals.among_us.analysis._plotting import (
    add_legend,
    format_yticks,
    save_plot,
)


# Maps the game engine's integer ``winner`` code to a display label.
# Mirrors ``winner_reason_map`` in
# ``among-agents/amongagents/envs/game.py`` (see ``check_game_over``):
#   1 -> impostors win (crewmates outnumbered or tied)
#   2 -> crewmates win (impostors eliminated / voted out)
#   3 -> crewmates win (all tasks completed)
#   4 -> impostors win (time limit) — engine omits this from
#        winner_reason_map, so it never reaches a summary in practice.
# A previous version mapped positionally via ``REASONS[winner - 1]``,
# which silently swapped codes 2 and 3: "impostors eliminated" games
# (the most common crewmate win) were drawn under "tasks done", and
# vice-versa.
WINNER_CODE_LABELS = {
    1: "Impostors Win! (outnumbered C.)",
    2: "Crewmates Win! (voted out I.)",
    3: "Crewmates Win! (tasks done)",
    4: "Impostors Win! (time up)",
}

# X-axis order (by winner code). Codes 1-3 occur in practice; code 4 is
# kept last for completeness even though the engine never records it.
REASON_ORDER: List[int] = [1, 2, 3, 4]
REASONS: List[str] = [WINNER_CODE_LABELS[c] for c in REASON_ORDER]

IMPOSTOR_WINNER_CODES = {1, 4}
CREWMATE_WINNER_CODES = {2, 3}

IMPOSTOR_COLOR = "#c0514e"
IMPOSTOR_EDGE = "#7a2e2b"
CREWMATE_COLOR = "#6aa56d"
CREWMATE_EDGE = "#3f6c44"

CUSTOM_COLORS: List[Tuple[str, str]] = [
    ("#c97b63", "#934d30"),  # clay
    ("#6e7c70", "#3f4d46"),  # moss
    ("#7a9e76", "#486b49"),  # sage
    ("#b29fce", "#77619e"),  # lilac
    ("#6f8fb0", "#3f5d7d"),  # steel
    ("#d4a056", "#a37434"),  # amber (new — self_report_experimental)
    ("#5fa3a0", "#357a77"),  # teal (self_report_experimental, StMP technique)
]

# Consistent condition labels reused across every model. In all of these
# experiments the crewmate is unsteered; only the impostor steering condition
# varies, so the legend label names that condition (the model is in the title).
COND_BASELINE = "Unsteered (baseline)"
COND_SELF_REPORT = "Self-Report steered"
COND_GLOBAL = "Global steered"
COND_SR_EXPANDED = "Self-Report Expanded steered"
COND_SR_EXPANDED_FCT = "Self-Report Expanded\n(first content token) steered"
# Self-Report Experimental vector applied with the StTP (Steering to Target
# Projection) technique — the technique used by the 2026-05-22 *_steered_srexp
# runs. The label names the technique so it contrasts with the StMP condition.
COND_SR_EXPERIMENTAL = "Self-Report Experimental\n(StTP) steered"
# Same self_report_experimental vector as COND_SR_EXPERIMENTAL but applied
# with the StMP (Steering to Mirror Projection) technique instead, so the
# label names the technique.
COND_SR_EXP_STMP = "Self-Report Experimental\n(StMP) steered"

# Fixed colour per condition so the same condition looks identical across the
# per-model plots, regardless of how many conditions a given model has.
CONDITION_COLORS = {
    COND_BASELINE: CUSTOM_COLORS[0],
    COND_SELF_REPORT: CUSTOM_COLORS[1],
    COND_GLOBAL: CUSTOM_COLORS[2],
    COND_SR_EXPANDED: CUSTOM_COLORS[3],
    COND_SR_EXPANDED_FCT: CUSTOM_COLORS[4],
    COND_SR_EXPERIMENTAL: CUSTOM_COLORS[5],
    COND_SR_EXP_STMP: CUSTOM_COLORS[6],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})


def calculate_bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.90
) -> Tuple[float, float]:
    """Bootstrap CI for the sum of a binary indicator array."""
    bootstrap_samples = np.random.choice(
        data, size=(n_bootstrap, len(data)), replace=True
    )
    bootstrap_counts = np.sum(bootstrap_samples, axis=1)
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    lower = float(np.percentile(bootstrap_counts, lower_percentile))
    upper = float(np.percentile(bootstrap_counts, upper_percentile))
    return lower, upper


def _winner_label(code: int) -> str:
    """Map a game engine ``winner`` code to its display label.

    Raises ``KeyError`` on an unknown code instead of silently
    mislabeling it (project convention: surface problems immediately).
    """
    if code not in WINNER_CODE_LABELS:
        raise KeyError(
            f"unexpected winner code {code!r}; known codes: "
            f"{sorted(WINNER_CODE_LABELS)}"
        )
    return WINNER_CODE_LABELS[code]


def _wrap_label(reason: str) -> str:
    words = reason.split()
    if len(words) <= 3:
        return reason
    mid = len(words) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def _load_combined(
    summary_paths: Sequence[str], descriptions: Sequence[str]
) -> pd.DataFrame:
    if len(summary_paths) != len(descriptions):
        raise ValueError(
            f"summary_paths ({len(summary_paths)}) and descriptions "
            f"({len(descriptions)}) must have the same length"
        )
    dfs: List[pd.DataFrame] = []
    for path, desc in zip(summary_paths, descriptions):
        df = load_game_summary(path)
        df["Winner Reason"] = df["Winner"].apply(_winner_label)
        df["Models"] = desc
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_winner_reasons(
    summary_paths: Sequence[str],
    descriptions: Sequence[str],
    save_stem: str,
    n_bootstrap: int = 1000,
    ci: float = 0.90,
    seed: int = 0,
    color_map: dict = None,
    title: str = None,
) -> None:
    """Grouped bar chart of winner reasons with bootstrap CIs.

    If *color_map* is given it maps a description to a ``(face, border)``
    colour pair so the same condition keeps a fixed colour across plots;
    otherwise colours fall back to :data:`CUSTOM_COLORS` by position.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # bootstrap helper uses legacy global RNG
    _ = rng  # kept to make the seeding source explicit

    combined = _load_combined(summary_paths, descriptions)

    winner_reasons = REASONS
    models = list(descriptions)
    num_reasons = len(winner_reasons)
    num_models = len(models)
    bar_width = 0.8 / num_models

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.yaxis.grid(True, linewidth=0.8, color="lightgray", zorder=0)
    ax.set_axisbelow(True)

    max_top = 0.0
    counts_table: List[List[int]] = []
    for j, model in enumerate(models):
        model_rows = combined[combined["Models"] == model]
        counts: List[int] = []
        errs: List[float] = []
        positions: List[float] = []
        for i, reason in enumerate(winner_reasons):
            indicator = (model_rows["Winner Reason"] == reason).astype(int).values
            count = int(indicator.sum())
            lo, hi = calculate_bootstrap_ci(indicator, n_bootstrap, ci)
            half_width = (hi - lo) / 2
            counts.append(count)
            errs.append(half_width)
            positions.append(i + (j - num_models / 2 + 0.5) * bar_width)
            max_top = max(max_top, count + half_width)
        counts_table.append(counts)

        if color_map is not None:
            if model not in color_map:
                raise KeyError(f"no colour mapped for condition {model!r}")
            face, border = color_map[model]
        else:
            face, border = CUSTOM_COLORS[j % len(CUSTOM_COLORS)]
        ax.bar(
            positions,
            counts,
            width=bar_width * 0.9,
            color=face,
            edgecolor=border,
            linewidth=1.2,
            yerr=errs,
            ecolor=border,
            capsize=3,
            label=model,
            zorder=3,
        )

    ax.set_xticks(range(num_reasons))
    ax.set_xticklabels([_wrap_label(r) for r in winner_reasons])
    ax.set_xlabel("Winner (Reason)")
    ax.set_ylabel("Number of Games")

    ymax = max_top * 1.25
    ax.set_ylim(0, ymax)
    format_yticks(ax, ymin=0, ymax=ymax, max_ticks=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_legend(
        ax,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        title=title,
    )

    print(f"[winrate] counts per (model × reason):")
    for model, row in zip(models, counts_table):
        print(f"  {model}")
        for reason, count in zip(winner_reasons, row):
            print(f"    {reason:<35s} {count}")

    save_plot(save_stem, fig=fig, dpi=300)
    plt.close(fig)


def plot_winrate_stacked(
    summary_paths: Sequence[str],
    descriptions: Sequence[str],
    save_stem: str,
    tick_labels: Sequence[str] | None = None,
) -> None:
    """Stacked bar chart of per-team win rate (impostor vs crewmate).

    Each bar sums to 1.0. Different winning reasons for the same team are
    merged (impostor: codes 1 & 4; crewmate: codes 2 & 3).
    """
    combined = _load_combined(summary_paths, descriptions)
    models = list(descriptions)

    impostor_rates: List[float] = []
    crewmate_rates: List[float] = []
    for desc in models:
        rows = combined[combined["Models"] == desc]
        total = len(rows)
        if total == 0:
            raise ValueError(f"No games found for '{desc}'")
        n_impostor = int(rows["Winner"].isin(IMPOSTOR_WINNER_CODES).sum())
        n_crewmate = int(rows["Winner"].isin(CREWMATE_WINNER_CODES).sum())
        impostor_rates.append(n_impostor / total)
        crewmate_rates.append(n_crewmate / total)

    impostor_arr = np.array(impostor_rates)
    crewmate_arr = np.array(crewmate_rates)
    x = np.arange(len(models))
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.yaxis.grid(True, linewidth=0.8, color="lightgray", zorder=0)
    ax.set_axisbelow(True)

    ax.bar(
        x, impostor_arr, width=bar_width,
        color=IMPOSTOR_COLOR, edgecolor=IMPOSTOR_EDGE, linewidth=1.2,
        label="Impostors Win", zorder=3,
    )
    ax.bar(
        x, crewmate_arr, width=bar_width, bottom=impostor_arr,
        color=CREWMATE_COLOR, edgecolor=CREWMATE_EDGE, linewidth=1.2,
        label="Crewmates Win", zorder=3,
    )

    for xi, (im, cr) in enumerate(zip(impostor_arr, crewmate_arr)):
        if im > 0.04:
            ax.text(xi, im / 2, f"{im * 100:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=13, fontweight="bold", zorder=4)
        if cr > 0.04:
            ax.text(xi, im + cr / 2, f"{cr * 100:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=13, fontweight="bold", zorder=4)

    labels = list(tick_labels) if tick_labels is not None else list(models)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_legend(
        ax,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
    )

    print("[winrate-stacked] win rate per model:")
    for desc, cr, im in zip(models, crewmate_arr, impostor_arr):
        print(f"  {desc}: crewmate={cr:.2%}, impostor={im:.2%}")

    save_plot(save_stem, fig=fig, dpi=300)
    plt.close(fig)


# Per-model experiment series. Each entry is (condition label, expt-log dir).
# Condition labels are shared across models so the legend is consistent; the
# concrete experiment directory differs per model.
MODEL_EXPERIMENTS = {
    "llama": {
        "display": "Llama-3.3-70B",
        "experiments": [
            (COND_BASELINE, "2026-04-18_llama_baseline_100_games"),
            (COND_SELF_REPORT, "2026-04-18_llama_steered_100_games"),
            (COND_GLOBAL, "2026-04-24_llama_global_vec_steered_100_games"),
            (COND_SR_EXPANDED, "2026-05-14_llama_self_report_expanded"),
            (COND_SR_EXPANDED_FCT, "2026-05-17_llama_self_report_expanded_first_c_t"),
        ],
    },
    "gemma": {
        "display": "Gemma-4-31B",
        "experiments": [
            (COND_BASELINE, "2026-04-26_gemma_baseline"),
            (COND_SELF_REPORT, "2026-04-25_gemma_self_report"),
            (COND_GLOBAL, "2026-04-25_gemma_global"),
            (COND_SR_EXPANDED, "2026-05-13_gemma_self_report_expanded"),
            (COND_SR_EXPANDED_FCT, "2026-05-17_gemma_self_report_expanded_first_c_t"),
        ],
    },
    "qwen": {
        "display": "Qwen3.6-27B",
        "experiments": [
            (COND_BASELINE, "2026-04-29_qwen_baseline"),
            (COND_SELF_REPORT, "2026-04-29_qwen_self_report_inst_disc"),
            (COND_GLOBAL, "2026-04-29_qwen_global"),
            (COND_SR_EXPANDED, "2026-05-13_qwen_self_report_expanded"),
            (COND_SR_EXPANDED_FCT, "2026-05-17_qwen_self_report_expanded_first_c_t"),
        ],
    },
    # Qwen @2048 clean-memory comparison (H200, batch=20). Baseline +
    # self_report_experimental steered, both at max_new_tokens=2048. The
    # baselines/steered runs in the "qwen" key above were at batch=100/H100
    # with documented memory-truncation contamination; this entry is the
    # clean post-fix comparison for the 2048-token regime.
    "qwen_tok2048_srexp": {
        "display": "Qwen3.6-27B (max_new_tokens=2048, clean)",
        "experiments": [
            (COND_BASELINE, "2026-05-22_qwen_tok2048_baseline"),
            (COND_SR_EXPERIMENTAL, "2026-05-22_qwen_tok2048_steered_srexp"),
        ],
    },
    # Llama @1024 clean-memory comparison (H200, batch=20). Baseline +
    # self_report_experimental steered, both at max_new_tokens=1024. Baseline
    # is repA (impWR 51%); the older Llama entries in the "llama" key above
    # were at batch=100/H100 with documented memory-truncation contamination.
    "llama_tok1024_srexp": {
        "display": "Llama-3.3-70B (max_new_tokens=1024, clean)",
        "experiments": [
            (COND_BASELINE, "2026-05-22_llama_tok1024_baseline_repA"),
            (COND_SR_EXPERIMENTAL, "2026-05-22_llama_tok1024_steered_srexp"),
        ],
    },
    # Llama @1024 StMP comparison. Same clean baseline as above, but the
    # steered arm uses the self_report_experimental vector applied with the
    # StMP (Steering to Mirror Projection) technique (impostor-only steering,
    # crewmate unsteered) — 2026-05-29_exp_2.
    "llama_tok1024_srexp_stmp": {
        "display": "Llama-3.3-70B (max_new_tokens=1024, StMP)",
        "experiments": [
            (COND_BASELINE, "2026-05-22_llama_tok1024_baseline_repA"),
            (COND_SR_EXP_STMP, "2026-05-29_exp_2"),
        ],
    },
    # Llama @1024 three-way technique comparison on the same clean baseline:
    # unsteered vs StTP vs StMP, all using the self_report_experimental vector
    # (impostor-only steering, crewmate unsteered).
    "llama_tok1024_sttp_vs_stmp": {
        "display": "Llama-3.3-70B (max_new_tokens=1024, StTP vs StMP)",
        "experiments": [
            (COND_BASELINE, "2026-05-22_llama_tok1024_baseline_repA"),
            (COND_SR_EXPERIMENTAL, "2026-05-22_llama_tok1024_steered_srexp"),
            (COND_SR_EXP_STMP, "2026-05-29_exp_2"),
        ],
    },
    # Qwen @2048 three-way technique comparison on the same clean baseline:
    # unsteered vs StTP vs StMP, all using the self_report_experimental vector
    # (impostor-only steering, crewmate unsteered). StMP run is 2026-05-29_exp_3.
    "qwen_tok2048_sttp_vs_stmp": {
        "display": "Qwen3.6-27B (max_new_tokens=2048, StTP vs StMP)",
        "experiments": [
            (COND_BASELINE, "2026-05-22_qwen_tok2048_baseline"),
            (COND_SR_EXPERIMENTAL, "2026-05-22_qwen_tok2048_steered_srexp"),
            (COND_SR_EXP_STMP, "2026-05-29_exp_3"),
        ],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Winner-reason breakdown with bootstrap CIs across Among Us runs."
    )
    ap.add_argument("--run", action="append", metavar="LABEL=RUN",
                    help="One condition (repeatable). RUN is a run name under "
                         "results/among_us/ or an explicit path to a run dir, its "
                         "games/ dir, or a summary.json.")
    ap.add_argument("--title", default="Among Us outcomes")
    ap.add_argument("--out-name", default="amongus_winrate_comparison")
    ap.add_argument("--out-dir",
                    default=os.path.join(str(RESULTS_ROOT), "among_us", "plots"))
    args = ap.parse_args()

    if not args.run:
        # No --run: fall back to the persona_playground presets.
        expt_logs = os.path.join(str(RESULTS_ROOT), "among_us")
        for model_key, cfg in MODEL_EXPERIMENTS.items():
            descriptions = [label for label, _ in cfg["experiments"]]
            summary_paths = [os.path.join(expt_logs, d, "summary.json")
                             for _, d in cfg["experiments"]]
            plot_winner_reasons(
                summary_paths, descriptions,
                os.path.join(args.out_dir, f"winrate_comparison_{model_key}"),
                color_map=CONDITION_COLORS, title=cfg["display"])
        return

    labels, paths = [], []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=RUN, got {spec!r}")
        label, run = spec.split("=", 1)
        labels.append(label)
        paths.append(_resolve_summary(run))
    plot_winner_reasons(paths, labels,
                        os.path.join(args.out_dir, args.out_name),
                        title=args.title)


def _resolve_summary(run: str) -> str:
    """Accept a summary.json, a games/ dir, a run dir, or a bare run name.

    honesty_evals nests logs in ``<run>/games/``; persona_playground keeps them
    flat. Both resolve here so the two repos' runs can be plotted together.
    """
    if os.path.isfile(run):
        return run
    for cand in (os.path.join(run, "summary.json"),
                 os.path.join(run, "games", "summary.json"),
                 os.path.join(str(RESULTS_ROOT), "among_us", run, "games", "summary.json"),
                 os.path.join(str(RESULTS_ROOT), "among_us", run, "summary.json")):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"no summary.json found for run {run!r}")


if __name__ == "__main__":
    main()
