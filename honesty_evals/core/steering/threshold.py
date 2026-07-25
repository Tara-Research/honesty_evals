"""Load + validate the global gating-threshold config for StTP / StMP.

Ported from persona_playground src/steering/threshold_config.py.

The steering-payload compiler reads this to decide how the per-layer gating
threshold is computed:

    mode == "bias"          -> threshold = -bias/||v||      (class midpoint; original behavior)
    mode == "neg_mean_std"  -> threshold = mu_neg + alpha*std_neg  (conservative, neg-cluster anchored)

This only controls *which tokens get steered* (the gate); it does not affect
the StTP target s = mu_pos + coefficient*std_pos. `alpha` is kept distinct
from the steerers' `coefficient`.

Per-run / per-job override: the environment variables
``STEERING_THRESHOLD_MODE`` and ``STEERING_THRESHOLD_ALPHA`` take precedence
over the YAML when set. Each SLURM job is its own process, so parallel jobs
can use different alphas without touching the shared YAML file.
"""
from __future__ import annotations

import functools
import numbers
import os
from typing import Any, Dict

import yaml

from honesty_evals.core.paths import CONFIG_STEERING_THRESHOLD

VALID_MODES = {"bias", "neg_mean_std"}

MODE_ENV_VAR = "STEERING_THRESHOLD_MODE"
ALPHA_ENV_VAR = "STEERING_THRESHOLD_ALPHA"


@functools.lru_cache(maxsize=1)
def load_threshold_config() -> Dict[str, Any]:
    """Read and validate the gating-threshold config (env vars override the YAML).

    Returns a dict with keys "mode" (str) and "alpha" (float, only meaningful
    for mode == "neg_mean_std").

    Raises (no silent fallbacks, per project convention):
      FileNotFoundError  if the config file is missing,
      KeyError           if "mode" is absent from the YAML,
      ValueError         if the resolved "mode" is invalid or alpha is
                         missing/non-numeric when required.
    """
    if not CONFIG_STEERING_THRESHOLD.exists():
        raise FileNotFoundError(
            f"Gating-threshold config not found at {CONFIG_STEERING_THRESHOLD}"
        )

    with open(CONFIG_STEERING_THRESHOLD, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "mode" not in cfg:
        raise KeyError(
            f"'mode' missing from gating-threshold config {CONFIG_STEERING_THRESHOLD}"
        )

    mode = os.environ.get(MODE_ENV_VAR, cfg["mode"])
    if mode not in VALID_MODES:
        raise ValueError(
            f"gating-threshold 'mode' must be one of {sorted(VALID_MODES)}, got {mode!r}"
        )

    alpha_env = os.environ.get(ALPHA_ENV_VAR)
    if alpha_env is not None:
        try:
            alpha = float(alpha_env)
        except ValueError:
            raise ValueError(f"{ALPHA_ENV_VAR} must be numeric, got {alpha_env!r}")
    else:
        alpha = cfg.get("alpha")

    if mode == "neg_mean_std" and not isinstance(alpha, numbers.Real):
        raise ValueError(
            "gating-threshold mode 'neg_mean_std' requires a numeric 'alpha' "
            f"(set 'alpha' in {CONFIG_STEERING_THRESHOLD} or the {ALPHA_ENV_VAR} env var)"
        )

    return {"mode": mode, "alpha": float(alpha) if alpha is not None else 0.0}
