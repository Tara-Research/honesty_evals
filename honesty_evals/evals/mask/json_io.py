"""Central I/O utilities for the JSON-based MASK benchmark storage.

Ported from persona_playground src/mask/json_io.py with the storage layout
adapted to honesty_evals conventions:

* Raw benchmark inputs are COMMITTED and live under the repo data folder::

      data/mask/raw/{archetype}.json        # full benchmark
      data/mask/raw_test/{archetype}.json   # 5-row test subset

  (regenerated from HF ``cais/mask`` by the download stage).

* All experiment outputs live under a run directory
  (``results/mask/<run_name>/``, created by the generate stage)::

      <run_dir>/
          config.json                        # run manifest
          {archetype}/
              responses/{model}[_{technique}][_coef{v}].json
              evaluated/{model}[_{technique}][_coef{v}].json
              metrics/{model}[_{technique}][_coef{v}].json
          all_results.json                   # aggregated final metrics

Each JSON file wraps ``{"metadata": {...}, "data": [...]}``.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from honesty_evals.core.paths import DATA_FOLDER
from honesty_evals.evals.mask.filename_utils import KNOWN_ARCHETYPES, KNOWN_TECHNIQUES

# ---------------------------------------------------------------------------
# Base directory helpers
# ---------------------------------------------------------------------------

MASK_DATA_DIR = DATA_FOLDER / "mask"


def _raw_dir(test: bool) -> Path:
    return MASK_DATA_DIR / ("raw_test" if test else "raw")


# ---------------------------------------------------------------------------
# Path resolution functions
# ---------------------------------------------------------------------------

def _technique_coef_suffix(technique: Optional[str], coefficient: Optional[float] = None) -> str:
    """Build the ``_{technique}[_coef{v}]`` filename suffix.

    * ``no_steering`` / ``None`` -> empty string
    * otherwise ``_stmp``, ``_swfc_coef1.5``, etc.
    """
    if technique is None or technique == "no_steering":
        suffix = ""
    else:
        suffix = f"_{technique}"
    if coefficient is not None:
        suffix += f"_coef{coefficient}"
    return suffix


def _stem(model: str, technique: Optional[str] = None, coefficient: Optional[float] = None) -> str:
    return f"{model}{_technique_coef_suffix(technique, coefficient)}"


def raw_data_path(archetype: str, *, test: bool = False) -> Path:
    return _raw_dir(test) / f"{archetype}.json"


def response_path(
    run_dir: str | Path,
    archetype: str,
    model: str,
    technique: Optional[str] = None,
    coefficient: Optional[float] = None,
) -> Path:
    return Path(run_dir) / archetype / "responses" / f"{_stem(model, technique, coefficient)}.json"


def evaluated_path(
    run_dir: str | Path,
    archetype: str,
    model: str,
    technique: Optional[str] = None,
    coefficient: Optional[float] = None,
) -> Path:
    return Path(run_dir) / archetype / "evaluated" / f"{_stem(model, technique, coefficient)}.json"


def metrics_path(
    run_dir: str | Path,
    archetype: str,
    model: str,
    technique: Optional[str] = None,
    coefficient: Optional[float] = None,
) -> Path:
    return Path(run_dir) / archetype / "metrics" / f"{_stem(model, technique, coefficient)}.json"


def all_results_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "all_results.json"


# ---------------------------------------------------------------------------
# NaN / None conversion helpers
# ---------------------------------------------------------------------------

def _nan_to_none(obj: Any) -> Any:
    """Recursively convert NaN / NaT values to None for JSON serialization."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def _none_to_nan(obj: Any) -> Any:
    """Recursively convert None back to NaN for pandas columns that expect it."""
    if obj is None:
        return np.nan
    if isinstance(obj, list):
        return [_none_to_nan(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _none_to_nan(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def read_json_data(path: os.PathLike) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Read a MASK JSON file and return ``(metadata, DataFrame)``.

    Raises ``FileNotFoundError`` if *path* does not exist.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    metadata = blob["metadata"]
    rows = [_none_to_nan(row) for row in blob["data"]]
    df = pd.DataFrame(rows)
    return metadata, df


def write_json_data(
    path: os.PathLike,
    metadata: Dict[str, Any],
    df: pd.DataFrame,
) -> None:
    """Write a MASK JSON file from *metadata* and a DataFrame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _nan_to_none(df.to_dict(orient="records"))
    blob = {"metadata": _nan_to_none(metadata), "data": rows}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_metadata(
    archetype: str,
    model: Optional[str] = None,
    technique: Optional[str] = None,
    coefficient: Optional[float] = None,
    layer: Optional[int] = None,
    steering_mode: Optional[str] = None,
    vector_type: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Construct a metadata dict, including only non-None values."""
    meta: Dict[str, Any] = {"archetype": archetype}
    if model is not None:
        meta["model"] = model
    if technique is not None:
        meta["technique"] = technique
    if coefficient is not None:
        meta["coefficient"] = coefficient
    if layer is not None:
        meta["layer"] = layer
    if steering_mode is not None:
        meta["steering_mode"] = steering_mode
    if vector_type is not None:
        meta["vector_type"] = vector_type
    meta["timestamp"] = datetime.now(timezone.utc).isoformat()
    meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_raw_archetypes(*, test: bool = False) -> List[str]:
    """Return archetype names that have a ``{archetype}.json`` in the raw data dir."""
    base = _raw_dir(test)
    return [arch for arch in KNOWN_ARCHETYPES if (base / f"{arch}.json").exists()]


def discover_archetype_dirs(run_dir: str | Path, stage: str) -> List[str]:
    """Return archetype names under *run_dir* that have a ``{stage}/`` subdirectory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return sorted(
        d.name for d in run_dir.iterdir() if d.name in KNOWN_ARCHETYPES and (d / stage).is_dir()
    )


def discover_files_in_stage(run_dir: str | Path, archetype: str, stage: str) -> List[Path]:
    """Return sorted list of JSON files in ``<run_dir>/{archetype}/{stage}/``."""
    stage_dir = Path(run_dir) / archetype / stage
    if not stage_dir.is_dir():
        return []
    return sorted(stage_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_json_filename(filename: str) -> Tuple[str, str, Optional[float]]:
    """Parse a MASK JSON filename into ``(model, technique, coefficient)``.

    The archetype is **not** encoded in the filename (it comes from the
    directory path).

    Examples::

        Qwen3-8B.json                 -> ("Qwen3-8B", "no_steering", None)
        Qwen3-8B_stmp.json            -> ("Qwen3-8B", "stmp", None)
        Qwen3-8B_swfc_coef1.5.json    -> ("Qwen3-8B", "swfc", 1.5)

    Returns:
        ``(model, technique, coefficient)``
    """
    stem = filename
    if stem.endswith(".json"):
        stem = stem[:-5]

    # 1. Extract coefficient if present
    coefficient: Optional[float] = None
    coef_match = re.search(r"_coef([0-9.]+)$", stem)
    if coef_match:
        coefficient = float(coef_match.group(1))
        stem = stem[: coef_match.start()]

    # 2. Strip known technique suffix
    technique = "no_steering"
    for tech in KNOWN_TECHNIQUES:
        if stem.endswith(f"_{tech}"):
            technique = tech
            stem = stem[: -len(f"_{tech}")]
            break

    model = stem
    return model, technique, coefficient
