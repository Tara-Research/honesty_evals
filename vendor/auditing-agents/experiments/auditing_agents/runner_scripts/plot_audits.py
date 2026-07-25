"""Plot audit success rates from a scores JSON — a real CLI (no notebook).

Consumes ONLY the ``scores.json`` written by ``score_audits.py`` (no judge /
API calls), so plots are re-runnable for free and decoupled from judging.

Produces two figures (PDF + PNG) into ``<exp-dir>/plots/`` (overridable):
  * ``per_organism_success`` — one bar per organism (= detection rate), a dot
    per run, and a 95%% binomial CI whisker.
  * ``overall_pooled_success`` — a single bar (macro-average over organisms)
    with the pooled 95%% CI, and a dot per organism.

Reuses ``viz.plot_averaged_bars_with_dots`` and
``helpers.calculate_pooled_se_and_ci`` (display names are already formatted
into the scores JSON by ``score_audits.py`` via
``viz.format_quirk_display_name``).

    python -m experiments.auditing_agents.runner_scripts.plot_audits \\
        --scores-json agent_outputs_local_target/<campaign>/scores/scores.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (Mila CPU node / sbatch) — before pyplot/viz import

import numpy as np  # noqa: E402

from src.viz import plot_averaged_bars_with_dots  # noqa: E402

from experiments.auditing_agents.helpers import (  # noqa: E402
    calculate_pooled_se_and_ci,
)


def setup_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores-json",
        type=str,
        required=True,
        help="Path to scores.json produced by score_audits.py",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir for plots. Default: <exp-dir-from-json>/plots",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Chart title. Default: derived from the campaign dir name.",
    )
    return p


def _binomial_se(p: float, n: int) -> float:
    return (p * (1.0 - p) / n) ** 0.5 if n > 0 else 0.0


def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[plot_audits] wrote {path}")


def main() -> int:
    args = setup_argument_parser().parse_args()
    scores_path = Path(args.scores_json)
    if not scores_path.is_file():
        raise SystemExit(f"[plot_audits] FAIL: scores JSON not found: {scores_path}")
    report = json.loads(scores_path.read_text(encoding="utf-8"))

    organisms = report.get("organisms", {})
    if not organisms:
        raise SystemExit(
            f"[plot_audits] FAIL: no organisms in {scores_path} — nothing to plot."
        )

    exp_dir = report.get("exp_dir", "")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (Path(exp_dir) / "plots" if exp_dir else scores_path.parent / "plots")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    title_base = args.title or (Path(exp_dir).name if exp_dir else scores_path.stem)

    # Order organisms by success rate (descending) for a readable chart.
    ordered = sorted(
        organisms.items(), key=lambda kv: kv[1]["success_rate"], reverse=True
    )

    # ---- Plot (a): per-organism bars + per-run dots + 95% binomial CI ----
    per_org_data: dict[str, dict[str, float]] = {}
    per_org_err: dict[str, float] = {}
    quirk_means: list[float] = []
    quirk_ses: list[float] = []
    for _, o in ordered:
        disp = o["display_name"]
        runs = o["runs"]
        per_org_data[disp] = {
            f"run_{r['run_id'] if r['run_id'] is not None else i}": float(r["score"])
            for i, r in enumerate(runs)
        }
        p = o["success_rate"]
        se = _binomial_se(p, o["n_runs"])
        per_org_err[disp] = 1.96 * se
        quirk_means.append(p)
        quirk_ses.append(se)

    fig_a = plot_averaged_bars_with_dots(
        data=per_org_data,
        title=f"Quirk detection rate by organism — {title_base}",
        ylabel="Quirk detection rate",
        ylim=(0, 1.0),
        error_bars=per_org_err,
        preserve_order=True,
        show_legend=False,
        rotate_xticks=30,
    )
    if fig_a is not None:
        _save(fig_a, out_dir, "per_organism_success")

    # ---- Plot (b): overall pooled rate + CI, one dot per organism ----
    overall_mean, ci_lower, ci_upper = calculate_pooled_se_and_ci(
        quirk_means, quirk_ses
    )
    # plot_averaged_bars_with_dots takes a symmetric per-group half-width;
    # the pooled CI is asymmetric (clipped to [0,1]) — use the mean of the two
    # clipped half-widths as a faithful symmetric approximation for the whisker.
    overall_err = float(np.mean([ci_lower, ci_upper]))
    overall_data = {
        "Overall": {o["display_name"]: o["success_rate"] for _, o in ordered}
    }
    fig_b = plot_averaged_bars_with_dots(
        data=overall_data,
        title=(
            f"Overall detection rate — {title_base} "
            f"(macro avg {overall_mean:.1%}, 95% CI ±{overall_err:.1%})"
        ),
        ylabel="Detection rate",
        ylim=(0, 1.0),
        error_bars={"Overall": overall_err},
        preserve_order=True,
        show_legend=False,
        rotate_xticks=0,
    )
    if fig_b is not None:
        _save(fig_b, out_dir, "overall_pooled_success")

    print(
        f"\n[plot_audits] {len(organisms)} organisms | "
        f"overall macro avg {overall_mean:.1%} "
        f"(95% CI -{ci_lower:.1%}/+{ci_upper:.1%}) -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
