"""SteeringSpec validation tests (no GPU)."""
import pytest

from honesty_evals.core.steering.spec import SteeringSpec


def steered_kwargs(**overrides):
    kwargs = dict(
        method="sttp",
        steering_mode="all",
        layer=26,
        coefficient=3.0,
        extraction_method="logistic_regression",
        vector_type="global",
        honesty_category="generalized",
        features="response_avg",
        direction="positive",
    )
    kwargs.update(overrides)
    return kwargs


def test_none_spec_is_unsteered():
    spec = SteeringSpec(method="none")
    assert not spec.is_steered
    assert spec.slug() == "no_steering"


def test_none_spec_rejects_steering_fields():
    with pytest.raises(ValueError, match="must not carry steering fields"):
        SteeringSpec(method="none", layer=26)


def test_steered_spec_roundtrips():
    spec = SteeringSpec(**steered_kwargs())
    assert spec.is_steered
    assert SteeringSpec.from_dict(spec.to_dict()) == spec
    assert spec.slug() == "sttp_L26_c3.0"


def test_steered_spec_requires_all_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        SteeringSpec(**steered_kwargs(honesty_category=None))


def test_invalid_method_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        SteeringSpec(method="magic")


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="steering_mode"):
        SteeringSpec(**steered_kwargs(steering_mode="everything"))


def test_projection_methods_require_logistic_regression():
    with pytest.raises(ValueError, match="logistic_regression"):
        SteeringSpec(**steered_kwargs(method="stmp", extraction_method="caa", features=None))


def test_swfc_allows_caa():
    spec = SteeringSpec(**steered_kwargs(method="swfc", extraction_method="caa", features=None))
    assert spec.method == "swfc"
