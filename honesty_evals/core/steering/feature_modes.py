"""Shared definitions for the LR feature-mode flag.

Ported near-verbatim from persona_playground src/utils/feature_modes.py.

A "feature mode" selects which span/aggregation of activations the LR steering
vector was fit on (e.g. averaged response activations vs. all per-token response
activations vs. the first special / content token). Each mode corresponds to a
distinct set of `.pt` files referenced by the steering config YAML.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


SIX_MODES: Tuple[str, ...] = (
    "prompt_avg",
    "prompt_all",
    "response_avg",
    "response_all",
    "first_special_token",
    "first_content_token",
    "last_content_token",
)
DEFAULT_FEATURE: str = "response_avg"

# The four per-mode artifact kinds each config block carries.
ARTIFACT_KINDS: Tuple[str, ...] = ("lr_vec", "lr_bias", "pos_emb", "neg_emb")

# Map each mode to the YAML keys under common_steering_config_<model>_<dataset>
# that point at its LR vector / bias / per-sample (or per-token) embeddings.
# Keys mirror the canonical mode name exactly (e.g. response_all ->
# lr_vector_response_all_path), so this mapping is generated rather than
# hand-maintained — there is one source of truth and it cannot drift.
MODE_PATH_KEYS: Dict[str, Dict[str, str]] = {
    m: {
        "lr_vec": f"lr_vector_{m}_path",
        "lr_bias": f"lr_bias_{m}_path",
        "pos_emb": f"pos_emb_{m}_path",
        "neg_emb": f"neg_emb_{m}_path",
    }
    for m in SIX_MODES
}


def mode_path_key(features: str, kind: str) -> str:
    """Return the canonical config key for a (mode, artifact-kind) pair.

    `kind` is one of ARTIFACT_KINDS. Callers should use this instead of
    hardcoding key strings so the (mode, kind) -> key mapping stays singular.
    Raises ValueError on an unknown mode/kind so a typo fails loudly rather
    than producing a key that silently misses in the YAML.
    """
    if features not in SIX_MODES:
        raise ValueError(
            f"unknown feature mode {features!r}; expected one of {list(SIX_MODES)}"
        )
    if kind not in ARTIFACT_KINDS:
        raise ValueError(
            f"unknown artifact kind {kind!r}; expected one of {ARTIFACT_KINDS}"
        )
    return MODE_PATH_KEYS[features][kind]


def apply_feature_mode_paths(common_cfg: Dict, features: str) -> None:
    """Mutate `common_cfg` in place so its `vector_path` / `bias_path` /
    `pos_embeddings_path` / `neg_embeddings_path` keys point at the artifacts
    for the requested feature mode.

    Reads the per-mode YAML key names from MODE_PATH_KEYS and copies the
    matching values from `common_cfg`. Raises KeyError if any required key is
    missing (no silent fallback — surface YAML gaps loudly per project policy).
    """
    if features not in SIX_MODES:
        raise ValueError(
            f"--features must be one of {list(SIX_MODES)}; got {features!r}"
        )
    keys = MODE_PATH_KEYS[features]
    missing: List[str] = [k for k in keys.values() if k not in common_cfg]
    if missing:
        raise KeyError(
            f"common_steering_config block is missing required keys for "
            f"--features={features}: {missing}"
        )
    common_cfg["vector_path"] = common_cfg[keys["lr_vec"]]
    common_cfg["bias_path"] = common_cfg[keys["lr_bias"]]
    common_cfg["pos_embeddings_path"] = common_cfg[keys["pos_emb"]]
    common_cfg["neg_embeddings_path"] = common_cfg[keys["neg_emb"]]


def densify_ragged_embeddings(
    seq_by_layer,
    num_layers: int,
    target_layer: int,
    mode: str = "concat",
) -> torch.Tensor:
    """Convert Dict[layer] -> List[Tensor[T_i, D]] into a dense [num_layers, N, D] tensor.

    `mode`:
      - "concat": stack ALL tokens of every sequence (N = total tokens across all examples).
      - "first":  stack only the first token of each sequence (N = number of examples).

    Layers absent from the ragged dict are zero-padded (the LR vector is also zero
    on those layers, so they're no-ops at steering time). The target layer must
    be present; otherwise no real steering data exists at the layer we'd hook.
    """
    if target_layer not in seq_by_layer:
        raise ValueError(
            f"Target layer {target_layer} not in ragged embeddings; "
            f"available layers: {sorted(seq_by_layer)}"
        )
    # Cast to float32: the stats path drops to numpy (.cpu().numpy()), which
    # rejects bfloat16. Ragged extractions are typically saved in the model's
    # native dtype (e.g. bfloat16), so upcast here once at load time.
    if mode == "concat":
        flat_by_layer = {l: torch.cat(seqs, dim=0).to(torch.float32) for l, seqs in seq_by_layer.items()}
    elif mode == "first":
        flat_by_layer = {
            l: torch.stack([s[0] for s in seqs], dim=0).to(torch.float32)
            for l, seqs in seq_by_layer.items()
        }
    else:
        raise ValueError(f"Unknown densify mode {mode!r}; expected 'concat' or 'first'.")
    ref = flat_by_layer[target_layer]
    N_total, H = ref.shape
    dense = torch.zeros(num_layers, N_total, H, dtype=torch.float32)
    for l, t in flat_by_layer.items():
        # The ragged dict is keyed by absolute model layer and may contain
        # layers beyond the steering vector's range. Those layers have no LR
        # row to pair with, so drop them rather than overflowing `dense`.
        if not (0 <= l < num_layers):
            continue
        if t.shape[0] != N_total:
            raise ValueError(
                f"Layer {l}: ragged concat has {t.shape[0]} tokens, "
                f"expected {N_total} to match target layer {target_layer}."
            )
        dense[l] = t
    return dense
