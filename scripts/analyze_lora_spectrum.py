"""SVD / rank analysis of a LoRA task-vector adapter.

Answers "is the rank budget the binding constraint?" for an adapter built by
task arithmetic (honest_adapter - dishonest_adapter). The stored rank is the sum
of the two component ranks, so it is an upper bound, not a measurement — what
matters is how many directions actually carry energy.

Singular values of the applied delta Δ = (alpha/r)·B·A are computed without
forming Δ: with B = Q_b R_b and Aᵀ = Q_a R_a (thin QR, both r×r cores),
s(BA) = s(R_b R_aᵀ), an r×r SVD. Exact, and avoids a (out × in) matmul per
module.

    python scripts/analyze_lora_spectrum.py --adapter <path> [--save-npz out.npz]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from safetensors import safe_open

_LAYER = re.compile(r"layers\.(\d+)\.")


def _stats(s: np.ndarray) -> Dict[str, float]:
    """Concentration measures for one module's singular values (descending)."""
    energy = s**2
    total = energy.sum()
    cum = np.cumsum(energy) / total
    p = energy / total
    return {
        "rank_stored": len(s),
        "dims_80": int(np.searchsorted(cum, 0.80)) + 1,
        "dims_95": int(np.searchsorted(cum, 0.95)) + 1,
        "dims_99": int(np.searchsorted(cum, 0.99)) + 1,
        # exp(entropy of the normalised spectrum): "how many directions really"
        "eff_rank": float(np.exp(-np.sum(p * np.log(p + 1e-12)))),
        "s_max": float(s[0]),
        "cond": float(s[0] / max(s[-1], 1e-12)),
        # Frobenius norm of the delta = sqrt(sum of squared singular values).
        # Reveals the per-layer scaling profile (band / edge_scale).
        "fro": float(np.sqrt(total)),
    }


def analyze(adapter: str, save_npz: str | None) -> None:
    path = os.path.join(adapter, "adapter_model.safetensors")
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    r_cfg, alpha = cfg.get("r"), cfg.get("lora_alpha")
    scale = (alpha / r_cfg) if r_cfg else 1.0
    print(f"adapter        : {adapter}")
    print(f"config         : r={r_cfg} alpha={alpha} -> delta scale={scale:g}")
    build = os.path.join(adapter, "steering_adapter_build.json")
    if os.path.exists(build):
        b = json.load(open(build))
        print("build          : " + "  ".join(
            f"{k}={b[k]}" for k in ("scale", "layer_profile", "band_lo", "band_hi",
                                    "edge_scale") if k in b))
    print()

    rows: List[Dict] = []
    spectra: Dict[str, np.ndarray] = {}
    with safe_open(path, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith("lora_A.weight")]
        for i, ka in enumerate(sorted(keys), 1):
            base = ka[: -len("lora_A.weight")]
            kb = base + "lora_B.weight"
            A = f.get_tensor(ka).float()          # (r, in)
            B = f.get_tensor(kb).float()          # (out, r)
            Rb = torch.linalg.qr(B, mode="r").R
            Ra = torch.linalg.qr(A.mT, mode="r").R
            s = torch.linalg.svdvals(Rb @ Ra.mT).numpy() * scale
            s = np.sort(s)[::-1]
            m = _LAYER.search(base)
            layer = int(m.group(1)) if m else -1
            mod = base.rstrip(".").split(".")[-1]
            rec = {"module": mod, "layer": layer, **_stats(s)}
            rows.append(rec)
            spectra[f"{layer}.{mod}"] = s
            if i % 64 == 0:
                print(f"  ... {i}/{len(keys)} modules", flush=True)

    n_layers = max(r["layer"] for r in rows) + 1
    attn_layers = sorted({r["layer"] for r in rows if r["module"] == "q_proj"})

    def agg(sel, label):
        if not sel:
            return
        f = lambda k: np.array([r[k] for r in sel], dtype=float)
        print(f"  {label:<26}{len(sel):>5}"
              f"{f('dims_80').mean():>9.1f}{f('dims_95').mean():>9.1f}"
              f"{f('dims_99').mean():>9.1f}{f('eff_rank').mean():>10.1f}"
              f"{f('fro').mean():>11.3f}")

    hdr = f"  {'group':<26}{'n':>5}{'80%':>9}{'95%':>9}{'99%':>9}{'eff_rank':>10}{'‖Δ‖_F':>11}"
    print(f"\n=== all modules (stored rank {rows[0]['rank_stored']}) ===")
    print(hdr)
    agg(rows, "ALL")

    print(f"\n=== by module type ===")
    print(hdr)
    for mod in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        agg([r for r in rows if r["module"] == mod], mod)

    print(f"\n=== attention vs linear-attention layers ===")
    print(f"  {n_layers} layers; {len(attn_layers)} carry attention adapters "
          f"(full_attention): {attn_layers[:8]}...")
    print(hdr)
    agg([r for r in rows if r["layer"] in attn_layers], "full_attention layers")
    agg([r for r in rows if r["layer"] not in attn_layers], "linear_attention layers")

    print(f"\n=== by depth quartile (reveals the band profile via ‖Δ‖_F) ===")
    print(hdr)
    for q in range(4):
        lo, hi = q * n_layers // 4, (q + 1) * n_layers // 4
        agg([r for r in rows if lo <= r["layer"] < hi], f"layers {lo:2d}-{hi-1:2d}")

    eff = np.array([r["eff_rank"] for r in rows])
    d80 = np.array([r["dims_80"] for r in rows])
    print(f"\n=== capacity utilisation ===")
    print(f"  stored rank per module        : {rows[0]['rank_stored']}")
    print(f"  effective rank  median / mean : {np.median(eff):.1f} / {eff.mean():.1f}")
    print(f"  effective rank  min / max     : {eff.min():.1f} / {eff.max():.1f}")
    print(f"  dims for 80% energy   median  : {np.median(d80):.0f}")
    print(f"  modules using >50% of rank    : "
          f"{(eff > rows[0]['rank_stored'] / 2).sum()}/{len(rows)}")
    print(f"  modules using <25% of rank    : "
          f"{(eff < rows[0]['rank_stored'] / 4).sum()}/{len(rows)}")

    if save_npz:
        np.savez_compressed(save_npz, **spectra)
        print(f"\n[saved] full spectra -> {save_npz}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--save-npz", default=None)
    a = p.parse_args()
    analyze(a.adapter, a.save_npz)


if __name__ == "__main__":
    main()
