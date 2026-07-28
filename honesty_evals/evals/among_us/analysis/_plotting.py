"""Plot helpers for the ported Among Us analysis.

PORTED from persona_playground src/plotting/__init__.py — only the three
functions the AmongUs plot scripts import (save_plot, add_legend,
format_yticks, plus format_yticks's tick-step helper). The rest of that module
is result-file discovery for the steering sweeps, which does not apply here.

Kept byte-faithful so the ported figures render identically to the originals.
"""
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt


_NICE_FRACTIONS = (1.0, 2.0, 5.0, 10.0)


def _nice_step(span: float, max_ticks: int) -> float:
    """Tick step of the form ``{1,2,5} x 10^n`` giving <= max_ticks ticks."""
    raw = span / max(max_ticks - 1, 1)
    exponent = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10 ** exponent
    for f in _NICE_FRACTIONS:
        candidate = f * base
        if span / candidate + 1 <= max_ticks + 0.5:
            return candidate
    return _NICE_FRACTIONS[-1] * base


def format_yticks(ax, ymin=None, ymax=None, max_ticks=6, fontsize=None):
    """Set clean y-axis ticks with minimal decimal places.

    Dynamically computes a step of the form ``{1,2,5}×10^n`` from the
    axis range so that the tick count stays at or below *max_ticks*.
    """
    if ymin is None or ymax is None:
        lo, hi = ax.get_ylim()
        if ymin is None:
            ymin = lo
        if ymax is None:
            ymax = hi

    span = ymax - ymin
    if span <= 0:
        return

    step = _nice_step(span, max_ticks)

    first = math.ceil(ymin / step) * step
    ticks = []
    t = first
    while t <= ymax + step * 1e-9:
        ticks.append(round(t, 10))
        t += step

    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    if fontsize is not None:
        ax.tick_params(axis="y", labelsize=fontsize)


def save_plot(path, fig=None, dpi=200, bbox_inches="tight"):
    """Save a plot as both PNG and PDF.

    Strips any .png/.pdf extension from *path* to derive the stem,
    creates parent directories, then writes ``<stem>.png`` and ``<stem>.pdf``.
    """
    # Strip known extensions to get the stem
    stem = path
    for ext in (".png", ".pdf"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

    _save = fig.savefig if fig is not None else plt.savefig

    _save(f"{stem}.png", dpi=dpi, bbox_inches=bbox_inches)
    _save(f"{stem}.pdf", bbox_inches=bbox_inches)
    print(f"[save] {stem}.png")
    print(f"[save] {stem}.pdf")


def add_legend(
    target=None,
    handles=None,
    labels=None,
    *,
    loc="upper center",
    bbox_to_anchor=None,
    ncol=4,
    fontsize=10,
    frameon=True,
    framealpha=0.9,
    edgecolor="gray",
    title=None,
    title_fontsize=None,
    **kwargs,
):
    """Create a legend with unified benchmark-style framing."""
    if target is None:
        target = plt.gca()

    kw = dict(
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        frameon=frameon,
        framealpha=framealpha,
        edgecolor=edgecolor,
        **kwargs,
    )
    if bbox_to_anchor is not None:
        kw["bbox_to_anchor"] = bbox_to_anchor
    if title is not None:
        kw["title"] = title
    if title_fontsize is not None:
        kw["title_fontsize"] = title_fontsize

    if handles is not None and labels is not None:
        return target.legend(handles, labels, **kw)
    if handles is not None:
        kw["handles"] = handles
    return target.legend(**kw)
