"""Shared archetype/technique name constants for MASK benchmark files.

Ported from persona_playground src/mask/filename_utils.py with the legacy CSV
filename parser dropped (the JSON layout encodes the archetype in the
directory path; see :func:`honesty_evals.evals.mask.json_io.parse_json_filename`).
"""
from __future__ import annotations

KNOWN_ARCHETYPES = [
    "doubling_down_known_facts",  # longest first to avoid partial matches
    "continuations",
    "disinformation",
    "known_facts",
    "provided_facts",
    "statistics",
]

KNOWN_TECHNIQUES = [
    "aligned_baseline",
    "malicious_baseline",
    # Pseudo-technique: base/aligned model honesty scored against another
    # run's reused beliefs (see metric.process_base_filtered_files).
    "base_filtered",
]
