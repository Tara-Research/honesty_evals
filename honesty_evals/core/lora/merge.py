"""Cache-aware LoRA adapter merger for AuditBench's honesty-LoRA investigation
tool.

Given a target quirk LoRA + an investigation LoRA (e.g. an honesty LoRA), this
module produces a single merged adapter whose behavior is equivalent to
applying both LoRAs additively at runtime. Merged adapters are cached under
``LORA_CACHE_ROOT`` (default ``$SCRATCH/honesty_evals_lora_cache``) by a stable
key of (target_revision, honesty_revision, scale), so repeated audits of the
same pair reuse the file instead of recomputing.

The merge math mirrors the vendored
``src/internals/methods/lora_steering.py::create_merged_lora_adapter`` — no
approximation, just no server round-trips:

    LoRA delta:      ΔW = (alpha/r) · B · A
    Two adapters:    output = W₀·x + ΔW_target·x + ΔW_honesty·x
    Concatenation:   merged_A = [(alpha/r)_t · A_t ; (alpha/r)_h · scale · A_h]
                     merged_B = [B_t | B_h]
                     merged_r = r_target + r_honesty
                     merged_alpha = merged_r     # A is already pre-scaled

Precondition: both adapters must target the same base model. Modules present
in only one of the two are zero-padded on the missing side (same as vendored).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from safetensors.torch import load_file, save_file

from honesty_evals.core.paths import LORA_CACHE_ROOT

log = logging.getLogger(__name__)


def _find_lora_key(state_dict: dict, module: str, lora_type: str) -> Optional[str]:
    for pattern in (f"{module}.{lora_type}.default.weight", f"{module}.{lora_type}.weight"):
        if pattern in state_dict:
            return pattern
    return None


def _extract_modules(state_dict: dict) -> set:
    return {
        key.rsplit(".lora_", 1)[0]
        for key in state_dict
        if ".lora_A" in key or ".lora_B" in key
    }


def _resolve_source(source: str) -> Tuple[Path, str]:
    """Return ``(local_adapter_dir, revision_id_for_cache_key)``.

    Local dirs get hashed by their weights file so a silent overwrite still
    invalidates the cache. HF repo ids resolve via ``snapshot_download`` —
    honoring ``HF_HUB_OFFLINE=1`` (cluster reproducibility mode) by falling
    back to ``local_files_only=True``. The commit sha comes from the resolved
    snapshot path (HF's cache layout is ``…/snapshots/<sha>/``); we never call
    the HTTP API, so this works fully offline as long as the model has been
    downloaded once.
    """
    p = Path(source)
    if p.is_dir():
        p = p.resolve()
        weights = p / "adapter_model.safetensors"
        if not weights.is_file():
            weights = p / "adapter_model.bin"
        if not weights.is_file():
            raise FileNotFoundError(f"No adapter_model.{{safetensors,bin}} in {p}")
        h = hashlib.sha1()
        with weights.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return p, f"local:{h.hexdigest()[:16]}"

    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    try:
        downloaded = Path(
            snapshot_download(
                repo_id=source,
                allow_patterns=["*.safetensors", "*.bin", "*.json"],
                local_files_only=offline,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve {source!r} from HF cache (offline={offline}). "
            f"If you're in offline mode, download it once with HF_HUB_OFFLINE=0 "
            f"and it will land in $HF_HOME. Underlying error: {exc}"
        ) from exc

    # HF cache layout: <cache_root>/models--<org>--<name>/snapshots/<sha>/…
    # The snapshot dir name IS the commit sha — extract it for the cache key
    # so a `git pull` on the HF repo invalidates our merge cache correctly.
    commit = downloaded.resolve().name or "unknown"
    return downloaded, f"hf:{source}@{commit[:12]}"


def _cache_key(target_rev: str, honesty_rev: str, scale: float) -> str:
    payload = json.dumps(
        {"target": target_rev, "honesty": honesty_rev, "scale": float(scale)},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _cache_dirname(target_source: str, honesty_source: str, key: str) -> str:
    def _tail(s: str) -> str:
        return s.rstrip("/").split("/")[-1][:32]
    return f"{_tail(target_source)}_x_{_tail(honesty_source)}_{key}"


def _load_state_dict(adapter_dir: Path) -> dict:
    safetensors = adapter_dir / "adapter_model.safetensors"
    if safetensors.is_file():
        return load_file(str(safetensors))
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter_model.{{safetensors,bin}} in {adapter_dir}")


def get_or_create_merged_adapter(
    target_source: str,
    honesty_source: str,
    scale: float = 1.0,
    cache_root: Optional[Path] = None,
) -> Path:
    """Return a path to a merged LoRA adapter directory, computing it if the
    cache doesn't already have it.

    Args:
        target_source: HF repo id (e.g. ``"auditing-agents/secret_loyalty"``)
            or local path to the target's quirk LoRA.
        honesty_source: HF repo id or local path to the investigation LoRA.
        scale: multiplier applied to the honesty LoRA's contribution
            (default 1.0). Enters the cache key so different scales get
            distinct cache entries.
        cache_root: override ``LORA_CACHE_ROOT``.

    Returns:
        Path to a directory containing ``adapter_model.safetensors`` +
        ``adapter_config.json`` + ``manifest.json``. Point vLLM at it via
        ``--lora <name>=<this-path>``.
    """
    cache_root = cache_root or LORA_CACHE_ROOT
    cache_root.mkdir(parents=True, exist_ok=True)

    target_path, target_rev = _resolve_source(target_source)
    honesty_path, honesty_rev = _resolve_source(honesty_source)
    key = _cache_key(target_rev, honesty_rev, scale)
    out_dir = cache_root / _cache_dirname(target_source, honesty_source, key)

    if (out_dir / "adapter_model.safetensors").is_file() and (out_dir / "adapter_config.json").is_file():
        log.info("[lora-merge] cache hit: %s", out_dir)
        return out_dir

    log.info("[lora-merge] cache miss; computing merge")
    log.info("  target : %s (%s)", target_source, target_rev)
    log.info("  honesty: %s (%s)", honesty_source, honesty_rev)
    log.info("  scale  : %.3f", scale)

    target_sd = _load_state_dict(target_path)
    honesty_sd = _load_state_dict(honesty_path)
    target_cfg = LoraConfig.from_pretrained(str(target_path))
    honesty_cfg = LoraConfig.from_pretrained(str(honesty_path))

    target_scaling = target_cfg.lora_alpha / target_cfg.r
    honesty_scaling = honesty_cfg.lora_alpha / honesty_cfg.r
    target_r, honesty_r = target_cfg.r, honesty_cfg.r
    merged_r = target_r + honesty_r

    merged_sd: dict = {}
    for module in _extract_modules(target_sd) | _extract_modules(honesty_sd):
        t_a_key = _find_lora_key(target_sd, module, "lora_A")
        t_b_key = _find_lora_key(target_sd, module, "lora_B")
        h_a_key = _find_lora_key(honesty_sd, module, "lora_A")
        h_b_key = _find_lora_key(honesty_sd, module, "lora_B")
        target_has = t_a_key is not None
        honesty_has = h_a_key is not None
        out_a = t_a_key or h_a_key
        out_b = t_b_key or h_b_key

        if target_has and honesty_has:
            t_a = target_sd[t_a_key]
            t_b = target_sd[t_b_key]
            h_a = honesty_sd[h_a_key].to(dtype=t_a.dtype, device="cpu")
            h_b = honesty_sd[h_b_key].to(dtype=t_b.dtype, device="cpu")
            merged_sd[out_a] = torch.cat(
                [t_a * target_scaling, h_a * honesty_scaling * scale], dim=0
            )
            merged_sd[out_b] = torch.cat([t_b, h_b], dim=1)
        elif target_has:
            t_a = target_sd[t_a_key]
            t_b = target_sd[t_b_key]
            merged_sd[out_a] = torch.cat(
                [t_a * target_scaling,
                 torch.zeros(honesty_r, t_a.shape[1], dtype=t_a.dtype)],
                dim=0,
            )
            merged_sd[out_b] = torch.cat(
                [t_b, torch.zeros(t_b.shape[0], honesty_r, dtype=t_b.dtype)],
                dim=1,
            )
        else:
            h_a = honesty_sd[h_a_key].to(device="cpu")
            h_b = honesty_sd[h_b_key].to(device="cpu")
            merged_sd[out_a] = torch.cat(
                [torch.zeros(target_r, h_a.shape[1], dtype=h_a.dtype),
                 h_a * honesty_scaling * scale],
                dim=0,
            )
            merged_sd[out_b] = torch.cat(
                [torch.zeros(h_b.shape[0], target_r, dtype=h_b.dtype), h_b],
                dim=1,
            )

    # Preserve any non-LoRA weights from either source (biases etc.) so an
    # adapter that trained a bias term keeps it after merge.
    for name, tensor in target_sd.items():
        merged_sd.setdefault(name, tensor)
    for name, tensor in honesty_sd.items():
        if ".lora_A" in name or ".lora_B" in name:
            continue
        merged_sd.setdefault(name, tensor.to(device="cpu"))

    # Post-merge alpha/r must be 1 because A has been pre-scaled.
    target_cfg.r = merged_r
    target_cfg.lora_alpha = merged_r

    # Write atomically: build in a sibling ``<name>.tmp`` then rename, so a
    # crash mid-save can never leave a half-written cache entry that later
    # runs mistake for a hit.
    tmp_dir = out_dir.with_suffix(".tmp")
    if tmp_dir.exists():
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True)
    save_file(merged_sd, str(tmp_dir / "adapter_model.safetensors"))
    target_cfg.save_pretrained(str(tmp_dir))
    manifest = {
        "target_source": target_source,
        "target_revision": target_rev,
        "honesty_source": honesty_source,
        "honesty_revision": honesty_rev,
        "scale": float(scale),
        "target_rank": target_r,
        "honesty_rank": honesty_r,
        "merged_rank": merged_r,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    os.replace(tmp_dir, out_dir)

    log.info("[lora-merge] wrote %s (rank %d)", out_dir, merged_r)
    return out_dir
