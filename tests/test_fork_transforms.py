"""Numeric parity of the vllm-steer fork's sttp/stmp _transform against the
frozen HF reference math (tests/reference_hf/{sttp,stmp}.py _apply_steering).

Requires the fork to be installed (pip install -e external/vllm-steer);
skipped otherwise. No GPU needed — the transforms are pure tensor math.
"""
import pytest
import torch

vllm_algos = pytest.importorskip(
    "vllm.steer_vectors.algorithms",
    reason="vllm-steer fork not installed (pip install -e external/vllm-steer)",
)

from vllm.steer_vectors.algorithms.stmp import StmpAlgorithm  # noqa: E402
from vllm.steer_vectors.algorithms.sttp import SttpAlgorithm  # noqa: E402

H = 16
N_TOKENS = 64
THRESHOLD = 0.7
TARGET = 2.5
COEF = 0.8
CAP = 5.0

torch.manual_seed(0)


@pytest.fixture()
def v_hat():
    v = torch.randn(H, dtype=torch.float32)
    return v / v.norm()


@pytest.fixture()
def hidden(v_hat):
    return torch.randn(N_TOKENS, H, dtype=torch.float32) * 2.0


def reference_sttp(t, v_hat, threshold, target, direction):
    """Frozen-reference StTP math (SteeringToTargetProjection._apply_steering)."""
    p = (t * v_hat).sum(dim=-1)
    mask = p < threshold if direction == "positive" else p > threshold
    delta_p = (target - p) * mask
    return t + delta_p.unsqueeze(-1) * v_hat


def reference_stmp(t, v_hat, threshold, coefficient, cap, direction):
    """Frozen-reference StMP math (SteeringToMirrorProjection._apply_steering)."""
    p = (t * v_hat).sum(dim=-1)
    mask = p < threshold if direction == "positive" else p > threshold
    mirror_target = 2 * threshold - p
    p_prime = p + coefficient * (mirror_target - p)
    p_prime = torch.clamp(p_prime, max=cap)
    delta_p = (p_prime - p) * mask
    delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)
    return t + delta_p.unsqueeze(-1) * v_hat


@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_sttp_transform_matches_reference(hidden, v_hat, direction):
    algo = SttpAlgorithm(layer_id=0)
    params = {
        "v_hat": v_hat,
        "threshold": THRESHOLD,
        "target": TARGET,
        "direction": direction,
    }
    out = algo._transform(hidden.clone(), params)
    expected = reference_sttp(hidden, v_hat, THRESHOLD, TARGET, direction)
    torch.testing.assert_close(out, expected)
    # steered tokens land exactly on the target projection
    p_out = (out * v_hat).sum(dim=-1)
    p_in = (hidden * v_hat).sum(dim=-1)
    mask = p_in < THRESHOLD if direction == "positive" else p_in > THRESHOLD
    torch.testing.assert_close(
        p_out[mask], torch.full_like(p_out[mask], TARGET), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(p_out[~mask], p_in[~mask])


@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_stmp_transform_matches_reference(hidden, v_hat, direction):
    algo = StmpAlgorithm(layer_id=0)
    params = {
        "v_hat": v_hat,
        "threshold": THRESHOLD,
        "coefficient": COEF,
        "cap": CAP,
        "direction": direction,
    }
    out = algo._transform(hidden.clone(), params)
    expected = reference_stmp(hidden, v_hat, THRESHOLD, COEF, CAP, direction)
    torch.testing.assert_close(out, expected)


def test_sttp_rejects_nontrivial_scale(hidden, v_hat):
    algo = SttpAlgorithm(layer_id=0)
    params = {
        "v_hat": v_hat,
        "threshold": THRESHOLD,
        "target": TARGET,
        "direction": "positive",
        "scale_factor": 2.0,
    }
    with pytest.raises(ValueError, match="does not use 'scale'"):
        algo._transform(hidden, params)
