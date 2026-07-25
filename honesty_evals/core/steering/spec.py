"""SteeringSpec: one immutable description of "how this run is steered".

Every eval builds exactly one SteeringSpec from its CLI flags, compiles it once
(core/steering/compile.py) into a vllm-steer request payload, and attaches the
compiled request to every generation call. The spec is also serialized into
the run manifest so judge/metrics stages can reconstruct the run's provenance.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

VALID_METHODS = ("none", "swfc", "sttp", "stmp")
VALID_MODES = ("all", "response")
VALID_DIRECTIONS = ("positive", "negative")


@dataclasses.dataclass(frozen=True)
class SteeringSpec:
    """Full description of a steering configuration.

    method == "none" means unsteered generation; all other fields except
    `method` must then be None (enforced in __post_init__ — no silently
    ignored flags).
    """

    method: str
    steering_mode: Optional[str] = None
    layer: Optional[int] = None
    coefficient: Optional[float] = None
    extraction_method: Optional[str] = None
    vector_type: Optional[str] = None
    honesty_category: Optional[str] = None
    features: Optional[str] = None
    direction: Optional[str] = None

    def __post_init__(self):
        if self.method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got {self.method!r}")

        if self.method == "none":
            set_fields = [
                f.name
                for f in dataclasses.fields(self)
                if f.name != "method" and getattr(self, f.name) is not None
            ]
            if set_fields:
                raise ValueError(
                    f"method='none' must not carry steering fields, got: {set_fields}"
                )
            return

        required = {
            "steering_mode": self.steering_mode,
            "layer": self.layer,
            "coefficient": self.coefficient,
            "extraction_method": self.extraction_method,
            "vector_type": self.vector_type,
            "honesty_category": self.honesty_category,
            "direction": self.direction,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                f"steered spec (method={self.method!r}) is missing required fields: {missing}"
            )
        if self.steering_mode not in VALID_MODES:
            raise ValueError(
                f"steering_mode must be one of {VALID_MODES}, got {self.steering_mode!r}"
            )
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got {self.direction!r}"
            )
        if self.method in ("sttp", "stmp") and self.extraction_method != "logistic_regression":
            # StTP/StMP need bias + pos/neg embedding distributions, which only
            # the logistic-regression extraction produces.
            raise ValueError(
                f"{self.method} requires extraction_method='logistic_regression', "
                f"got {self.extraction_method!r}"
            )

    @property
    def is_steered(self) -> bool:
        return self.method != "none"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringSpec":
        return cls(**d)

    def slug(self) -> str:
        """Short human-readable tag for file/run naming."""
        if not self.is_steered:
            return "no_steering"
        return f"{self.method}_L{self.layer}_c{self.coefficient}"
