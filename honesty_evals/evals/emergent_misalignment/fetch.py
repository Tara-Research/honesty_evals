"""Emergent-misalignment eval — fetch stage.

Downloads the EM eval question YAMLs into data/emergent_misalignment/
(port of persona_playground's scripts/emergent_misalignment/fetch_em_questions.sh).
Run on a node WITH internet access (e.g. a login node) before submitting
generation jobs — compute nodes are offline on the clusters we use.

Source: https://github.com/emergent-misalignment/emergent-misalignment
License/provenance: see that repository.
"""
from __future__ import annotations

import argparse
import os

import httpx

from honesty_evals.evals.emergent_misalignment.questions import EM_QUESTIONS_DIR

QUESTION_SET_FILES = ("first_plot_questions.yaml", "preregistered_evals.yaml")
RAW_BASE = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/main/evaluation"
)


def run_fetch(args: argparse.Namespace) -> None:
    os.makedirs(EM_QUESTIONS_DIR, exist_ok=True)
    for fname in QUESTION_SET_FILES:
        dest = os.path.join(EM_QUESTIONS_DIR, fname)
        if os.path.isfile(dest):
            print(f"[skip] {dest} already present")
            continue
        url = f"{RAW_BASE}/{fname}"
        print(f"[fetch] {url} -> {dest}")
        resp = httpx.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
    print(f"[done] EM question YAMLs in {EM_QUESTIONS_DIR}")
