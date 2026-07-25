"""Golden tests for the SteeringSpec -> vllm-steer payload compiler (no GPU).

Verifies that compile_steering reproduces the projection-statistics math of the
frozen HF reference steerers (tests/reference_hf/sttp.py, stmp.py):
  v_hat = v/||v||, threshold = -bias/||v|| (mode "bias"),
  StTP target = mu_pos + coef*std_pos, StMP cap = mu_pos + 40*std_pos,
plus the steering-mode -> trigger-token mapping and the swfc "direct" payload.
"""
import numpy as np
import pytest
import torch

import honesty_evals.core.steering.compile as compile_mod
from honesty_evals.core.steering.spec import SteeringSpec

L, N, H = 4, 32, 16
TARGET_LAYER = 2
COEF = 3.0
RNG = np.random.default_rng(0)


@pytest.fixture()
def artifacts(tmp_path, monkeypatch):
    """Synthetic vector/bias/embedding artifacts + patched resolver."""
    vector = torch.tensor(RNG.normal(size=(L, H)), dtype=torch.float32)
    bias = torch.tensor(RNG.normal(size=(L,)), dtype=torch.float32)
    pos = torch.tensor(RNG.normal(loc=1.0, size=(L, N, H)), dtype=torch.float32)
    neg = torch.tensor(RNG.normal(loc=-1.0, size=(L, N, H)), dtype=torch.float32)

    store = tmp_path / "vectors"
    store.mkdir()
    torch.save(vector, store / "vector.pt")
    torch.save(bias, store / "bias.pt")
    torch.save(pos, store / "pos.pt")
    torch.save(neg, store / "neg.pt")

    monkeypatch.setattr(compile_mod, "PERSONA_VECTORS_FOLDER", str(store))
    monkeypatch.setattr(
        compile_mod,
        "resolve_artifact_paths",
        lambda **kwargs: {
            "vector_path": "vector.pt",
            "bias_path": "bias.pt",
            "pos_embeddings_path": "pos.pt",
            "neg_embeddings_path": "neg.pt",
        },
    )
    return {"vector": vector, "bias": bias, "pos": pos, "neg": neg, "dir": tmp_path}


def spec_for(method: str, mode: str = "all") -> SteeringSpec:
    return SteeringSpec(
        method=method,
        steering_mode=mode,
        layer=TARGET_LAYER,
        coefficient=COEF,
        extraction_method="logistic_regression",
        vector_type="global",
        honesty_category="generalized",
        features="response_avg",
        direction="positive",
    )


def expected_stats(artifacts):
    v = artifacts["vector"][TARGET_LAYER].numpy()
    v_norm = np.linalg.norm(v) + 1e-12
    v_hat = v / v_norm
    proj_pos = artifacts["pos"][TARGET_LAYER].numpy() @ v_hat
    proj_neg = artifacts["neg"][TARGET_LAYER].numpy() @ v_hat
    return {
        "v_hat": v_hat,
        "threshold": -float(artifacts["bias"][TARGET_LAYER]) / v_norm,
        "mu_pos": proj_pos.mean(),
        "std_pos": proj_pos.std(ddof=1),
        "mu_neg": proj_neg.mean(),
        "std_neg": proj_neg.std(ddof=1),
    }


def test_none_compiles_to_none(artifacts):
    assert compile_mod.compile_steering(
        SteeringSpec(method="none"), model_id="m", payload_dir=str(artifacts["dir"])
    ) is None


def test_swfc_payload_is_raw_vector_with_scale(artifacts):
    compiled = compile_mod.compile_steering(
        spec_for("swfc"), model_id="m", payload_dir=str(artifacts["dir"])
    )
    assert compiled.algorithm == "direct"
    assert compiled.scale == COEF
    assert compiled.target_layers == (TARGET_LAYER,)
    payload = torch.load(compiled.payload_path, weights_only=False)
    torch.testing.assert_close(payload, artifacts["vector"][TARGET_LAYER])


def test_sttp_payload_matches_reference_math(artifacts):
    compiled = compile_mod.compile_steering(
        spec_for("sttp"), model_id="m", payload_dir=str(artifacts["dir"])
    )
    assert compiled.algorithm == "sttp"
    assert compiled.scale == 1.0
    payload = torch.load(compiled.payload_path, weights_only=False)
    exp = expected_stats(artifacts)
    np.testing.assert_allclose(payload["v_hat"].numpy(), exp["v_hat"], rtol=1e-5)
    assert payload["threshold"] == pytest.approx(exp["threshold"], rel=1e-5)
    assert payload["target"] == pytest.approx(exp["mu_pos"] + COEF * exp["std_pos"], rel=1e-5)
    assert payload["direction"] == "positive"


def test_stmp_payload_matches_reference_math(artifacts):
    compiled = compile_mod.compile_steering(
        spec_for("stmp"), model_id="m", payload_dir=str(artifacts["dir"])
    )
    assert compiled.algorithm == "stmp"
    payload = torch.load(compiled.payload_path, weights_only=False)
    exp = expected_stats(artifacts)
    assert payload["coefficient"] == pytest.approx(COEF)
    assert payload["cap"] == pytest.approx(
        exp["mu_pos"] + compile_mod.STMP_COEFFICIENT_CAP * exp["std_pos"], rel=1e-5
    )
    assert payload["threshold"] == pytest.approx(exp["threshold"], rel=1e-5)


def test_trigger_token_mapping(artifacts):
    all_mode = compile_mod.compile_steering(
        spec_for("swfc", mode="all"), model_id="m", payload_dir=str(artifacts["dir"])
    )
    assert all_mode.prefill_trigger_tokens == (-1,)
    assert all_mode.generate_trigger_tokens == (-1,)

    response_mode = compile_mod.compile_steering(
        spec_for("swfc", mode="response"), model_id="m", payload_dir=str(artifacts["dir"])
    )
    assert response_mode.prefill_trigger_tokens is None
    assert response_mode.generate_trigger_tokens == (-1,)


def test_compile_is_content_addressed(artifacts):
    a = compile_mod.compile_steering(spec_for("sttp"), model_id="m", payload_dir=str(artifacts["dir"]))
    b = compile_mod.compile_steering(spec_for("sttp"), model_id="m", payload_dir=str(artifacts["dir"]))
    assert a.payload_path == b.payload_path
    assert a.name == b.name
