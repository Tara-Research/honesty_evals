"""Compile a SteeringSpec into a vllm-steer SteerVectorRequest payload.

This is the bridge between the persona-vector artifact store and the
vllm-steer fork's algorithm plugins (external/vllm-steer,
vllm/steer_vectors/algorithms/{direct,sttp,stmp}.py):

  * swfc -> algorithm "direct":  payload = raw steering vector row [H],
            scale = coefficient (the fork multiplies payload by scale).
  * sttp -> algorithm "sttp":    payload = {v_hat, threshold, target, direction}.
  * stmp -> algorithm "stmp":    payload = {v_hat, threshold, coefficient, cap,
                                            direction}.

The projection statistics (mu/sigma of the positive/negative training
activations projected onto v_hat, gating threshold) are computed HERE, once,
from the [L, N, H] embedding tensors — mirroring the math of
persona_playground's SteeringToTargetProjection/_projection_stats_all_layers —
and only the small per-layer payload is written to disk for vLLM to load.

Position semantics (encoded once, here):
  steering_mode "all"      -> prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1]
                              (the fork's global fast path: every token, both phases)
  steering_mode "response" -> prefill_trigger_tokens=None, generate_trigger_tokens=[-1]
                              (prompt KV cache stays unsteered, decode steps steered —
                              matches the HF hooks' "response" positions mode)
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from honesty_evals.core.paths import PERSONA_VECTORS_FOLDER
from honesty_evals.core.steering.feature_modes import densify_ragged_embeddings
from honesty_evals.core.steering.resolver import resolve_artifact_paths
from honesty_evals.core.steering.spec import SteeringSpec
from honesty_evals.core.steering.threshold import load_threshold_config

STMP_COEFFICIENT_CAP = 40.0  # cap = mu_pos + 40 * std_pos (from the HF StMP impl)


@dataclass(frozen=True)
class CompiledSteering:
    """Everything the serving layer needs to build a SteerVectorRequest."""

    name: str
    payload_path: str
    algorithm: str
    target_layers: tuple
    scale: float
    prefill_trigger_tokens: Optional[tuple]
    generate_trigger_tokens: Optional[tuple]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "payload_path": self.payload_path,
            "algorithm": self.algorithm,
            "target_layers": list(self.target_layers),
            "scale": self.scale,
            "prefill_trigger_tokens": (
                list(self.prefill_trigger_tokens)
                if self.prefill_trigger_tokens is not None
                else None
            ),
            "generate_trigger_tokens": (
                list(self.generate_trigger_tokens)
                if self.generate_trigger_tokens is not None
                else None
            ),
        }


def _trigger_tokens(steering_mode: str):
    if steering_mode == "all":
        return (-1,), (-1,)
    if steering_mode == "response":
        return None, (-1,)
    raise ValueError(f"unknown steering_mode {steering_mode!r}")


def _load_artifact(rel_path: str) -> torch.Tensor:
    full = os.path.join(PERSONA_VECTORS_FOLDER, rel_path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Steering artifact not found: {full}")
    return torch.load(full, map_location="cpu", weights_only=False)


def _dense_embeddings(raw, num_layers: int, target_layer: int) -> torch.Tensor:
    """Accept either a dense [L, N, H] tensor or a ragged Dict[layer] -> List[Tensor]."""
    if isinstance(raw, torch.Tensor):
        if raw.ndim != 3:
            raise ValueError(f"embedding tensor must be [L, N, H], got shape {tuple(raw.shape)}")
        return raw.to(torch.float32)
    if isinstance(raw, dict):
        return densify_ragged_embeddings(raw, num_layers=num_layers, target_layer=target_layer)
    raise ValueError(f"unsupported embedding artifact type: {type(raw)}")


def _projection_stats(
    vector_row: torch.Tensor,
    neg_layer: np.ndarray,
    pos_layer: np.ndarray,
    bias_value: float,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Per-layer projection stats, mirroring SteeringToTargetProjection.

    threshold = -bias / ||v||  (mode "bias")  or  mu_neg + alpha*std_neg
    (mode "neg_mean_std"), per the global threshold config.
    """
    v = vector_row.detach().to(torch.float32).cpu().numpy()
    v_norm = float(np.linalg.norm(v)) + eps
    v_hat = v / v_norm

    proj_neg = neg_layer @ v_hat
    proj_pos = pos_layer @ v_hat

    mu_neg = float(proj_neg.mean())
    std_neg = float(proj_neg.std(ddof=1)) if proj_neg.shape[0] > 1 else 0.0
    mu_pos = float(proj_pos.mean())
    std_pos = float(proj_pos.std(ddof=1)) if proj_pos.shape[0] > 1 else 0.0

    threshold_cfg = load_threshold_config()
    if threshold_cfg["mode"] == "neg_mean_std":
        threshold = mu_neg + threshold_cfg["alpha"] * std_neg
    else:  # mode == "bias"
        threshold = -bias_value / v_norm

    return {
        "v_hat": torch.from_numpy(v_hat).to(torch.float32),
        "threshold": float(threshold),
        "mu_neg": mu_neg,
        "std_neg": std_neg,
        "mu_pos": mu_pos,
        "std_pos": std_pos,
    }


def compile_steering(
    spec: SteeringSpec,
    model_id: str,
    payload_dir: str,
) -> Optional[CompiledSteering]:
    """Resolve artifacts for `spec`, compute the payload, write it to
    `payload_dir`, and return the compiled request description.

    Returns None for method == "none" (unsteered).
    """
    if not spec.is_steered:
        return None

    os.makedirs(payload_dir, exist_ok=True)

    paths = resolve_artifact_paths(
        model_id=model_id,
        vector_type=spec.vector_type,
        honesty_category=spec.honesty_category,
        vector_extraction_method=spec.extraction_method,
        features=spec.features,
    )

    vector = _load_artifact(paths["vector_path"])
    if vector.ndim != 2:
        raise ValueError(
            f"steering vector must be [L, H], got shape {tuple(vector.shape)} "
            f"from {paths['vector_path']}"
        )
    num_layers = vector.shape[0]
    if not (0 <= spec.layer < num_layers):
        raise ValueError(
            f"layer {spec.layer} out of range for steering vector with {num_layers} layers"
        )

    prefill, generate = _trigger_tokens(spec.steering_mode)

    if spec.method == "swfc":
        payload = vector[spec.layer].to(torch.float32).contiguous()
        algorithm = "direct"
        scale = float(spec.coefficient)
    else:
        if paths["bias_path"] is None:
            raise ValueError(
                f"{spec.method} requires a bias artifact; config resolved bias_path=None"
            )
        if paths["pos_embeddings_path"] is None or paths["neg_embeddings_path"] is None:
            raise ValueError(
                f"{spec.method} requires pos/neg embedding artifacts; config resolved "
                f"pos={paths['pos_embeddings_path']}, neg={paths['neg_embeddings_path']}"
            )
        bias = _load_artifact(paths["bias_path"])
        pos_emb = _dense_embeddings(
            _load_artifact(paths["pos_embeddings_path"]), num_layers, spec.layer
        )
        neg_emb = _dense_embeddings(
            _load_artifact(paths["neg_embeddings_path"]), num_layers, spec.layer
        )

        stats = _projection_stats(
            vector_row=vector[spec.layer],
            neg_layer=neg_emb[spec.layer].numpy(),
            pos_layer=pos_emb[spec.layer].numpy(),
            bias_value=float(bias[spec.layer].item()),
        )

        if spec.method == "sttp":
            if spec.direction == "positive":
                target = stats["mu_pos"] + spec.coefficient * stats["std_pos"]
            else:
                target = stats["mu_neg"] + spec.coefficient * stats["std_neg"]
            payload = {
                "v_hat": stats["v_hat"],
                "threshold": stats["threshold"],
                "target": float(target),
                "direction": spec.direction,
            }
            algorithm = "sttp"
        else:  # stmp
            cap = stats["mu_pos"] + STMP_COEFFICIENT_CAP * stats["std_pos"]
            payload = {
                "v_hat": stats["v_hat"],
                "threshold": stats["threshold"],
                "coefficient": float(spec.coefficient),
                "cap": float(cap),
                "direction": spec.direction,
            }
            algorithm = spec.method
        scale = 1.0

    # Content-addressed payload file: identical specs reuse the same file and
    # the server-side LRU can dedupe by name.
    digest_src = repr((spec.to_dict(), model_id, paths))
    digest = hashlib.sha256(digest_src.encode()).hexdigest()[:16]
    name = f"{spec.slug()}_{digest}"
    payload_path = os.path.join(payload_dir, f"{name}.pt")
    if not os.path.exists(payload_path):
        torch.save(payload, payload_path)

    return CompiledSteering(
        name=name,
        payload_path=payload_path,
        algorithm=algorithm,
        target_layers=(spec.layer,),
        scale=scale,
        prefill_trigger_tokens=prefill,
        generate_trigger_tokens=generate,
    )
