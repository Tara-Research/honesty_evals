"""MASK benchmark — download stage.

Regenerates the committed raw benchmark inputs from HuggingFace ``cais/mask``:

    data/mask/raw/{archetype}.json        # full test split
    data/mask/raw_test/{archetype}.json   # 5-row subset for quick runs

Port of persona_playground's src/mask/download.py.
"""
from __future__ import annotations

import argparse

import pandas as pd

from honesty_evals.core.paths import SEED
from honesty_evals.evals.mask.filename_utils import KNOWN_ARCHETYPES
from honesty_evals.evals.mask.json_io import build_metadata, raw_data_path, write_json_data

TEST_SUBSET_ROWS = 5


def run_download(args: argparse.Namespace) -> None:
    # Heavy import kept local so CLI --help works without the datasets package.
    from datasets import load_dataset

    for archetype in KNOWN_ARCHETYPES:
        ds = load_dataset("cais/mask", archetype)
        df = pd.DataFrame(ds["test"])

        full_path = raw_data_path(archetype, test=False)
        write_json_data(full_path, build_metadata(archetype=archetype), df)
        print(f"[mask/download] wrote {len(df)} rows to {full_path}")

        test_df = df.sample(TEST_SUBSET_ROWS, random_state=SEED)
        test_path = raw_data_path(archetype, test=True)
        write_json_data(test_path, build_metadata(archetype=archetype), test_df)
        print(f"[mask/download] wrote {len(test_df)} rows to {test_path}")


def add_args(parser: argparse.ArgumentParser) -> None:
    pass  # no flags: always regenerates both the full set and the test subset
