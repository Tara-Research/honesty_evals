"""Shared CLI argument groups for eval generate/run stages.

Every eval uses the same steering / serving / sampling flag conventions, added
via these helpers so the flag surface cannot drift between evals.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

from honesty_evals.core.steering.feature_modes import DEFAULT_FEATURE, SIX_MODES
from honesty_evals.core.steering.resolver import resolve_layer_and_coef
from honesty_evals.core.steering.spec import SteeringSpec

# CLI mode names map onto the steering-config YAML's section keys.
MODE_TO_CONFIG_KEY = {"all": "all_token_mode", "response": "response_mode"}


# ---------- steering ----------

def add_steering_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("steering")
    g.add_argument(
        "--steering",
        required=True,
        choices=["none", "swfc", "sttp", "stmp"],
        help="Steering method ('none' for unsteered generation)",
    )
    g.add_argument(
        "--steering-mode",
        choices=list(MODE_TO_CONFIG_KEY),
        help="Steer all tokens or only response (decode) tokens. Required iff steered.",
    )
    g.add_argument("--layer", type=int, help="Target layer (XOR with --tuned)")
    g.add_argument("--coefficient", type=float, help="Steering coefficient (XOR with --tuned)")
    g.add_argument(
        "--tuned",
        action="store_true",
        help="Resolve layer/coefficient from the steering config YAML instead of "
        "passing --layer/--coefficient explicitly",
    )
    g.add_argument(
        "--extraction-method",
        choices=["caa", "logistic_regression"],
        help="Vector extraction method. Required iff steered.",
    )
    g.add_argument(
        "--vector-type",
        default="global",
        help="Steering vector variant: 'global' (default), a per-category vector "
        "name, or a combined-categories name",
    )
    g.add_argument(
        "--honesty-category",
        help="Honesty dataset/category the vector was trained against "
        "(selects the common_steering_config_<model>_<category> block). "
        "Required iff steered.",
    )
    g.add_argument(
        "--features",
        choices=list(SIX_MODES),
        help=f"LR feature mode (logistic_regression only; typical: {DEFAULT_FEATURE})",
    )
    g.add_argument(
        "--direction",
        choices=["positive", "negative"],
        default="positive",
        help="Steering direction (default: positive = toward honesty)",
    )


def build_spec_from_args(args: argparse.Namespace, model_id: str) -> SteeringSpec:
    if args.steering == "none":
        _forbid(args, ["steering_mode", "layer", "coefficient", "extraction_method",
                       "honesty_category", "features"])
        if args.tuned:
            raise SystemExit("--tuned is meaningless with --steering none")
        return SteeringSpec(method="none")

    for flag in ("steering_mode", "extraction_method", "honesty_category"):
        if getattr(args, flag) is None:
            raise SystemExit(f"--{flag.replace('_', '-')} is required when --steering != none")

    has_explicit = args.layer is not None or args.coefficient is not None
    if args.tuned == has_explicit:
        raise SystemExit(
            "Pass exactly one of: --tuned, or --layer AND --coefficient"
        )

    if args.tuned:
        resolved = resolve_layer_and_coef(
            model_id=model_id,
            technique=args.steering,
            steering_mode=MODE_TO_CONFIG_KEY[args.steering_mode],
            vector_type=args.vector_type,
            honesty_category=args.honesty_category,
            vector_extraction_method=args.extraction_method,
        )
        layer, coefficient = resolved["layer"], resolved["coefficient"]
        if coefficient is None:
            raise SystemExit(
                "Tuned config entry has no coefficient; pass --layer/--coefficient explicitly"
            )
    else:
        if args.layer is None or args.coefficient is None:
            raise SystemExit("--layer and --coefficient must be passed together")
        layer, coefficient = args.layer, args.coefficient

    return SteeringSpec(
        method=args.steering,
        steering_mode=args.steering_mode,
        layer=layer,
        coefficient=coefficient,
        extraction_method=args.extraction_method,
        vector_type=args.vector_type,
        honesty_category=args.honesty_category,
        features=args.features,
        direction=args.direction,
    )


def _forbid(args: argparse.Namespace, flags) -> None:
    set_flags = [f for f in flags if getattr(args, f) is not None]
    if set_flags:
        pretty = ", ".join("--" + f.replace("_", "-") for f in set_flags)
        raise SystemExit(f"{pretty} must not be set with --steering none")


# ---------- serving ----------

def add_server_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("serving")
    g.add_argument("--server-url", help="URL of a running steering server")
    g.add_argument(
        "--launch-server",
        action="store_true",
        help="Spawn the steering server as a subprocess for this stage",
    )
    g.add_argument("--tp", type=int, help="Tensor parallel size (required with --launch-server)")
    g.add_argument("--port", type=int, default=8192, help="Port for --launch-server (default 8192)")
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)


def validate_server_args(args: argparse.Namespace) -> None:
    if (args.server_url is None) == (not args.launch_server):
        raise SystemExit("Pass exactly one of --server-url or --launch-server")
    if args.launch_server and args.tp is None:
        raise SystemExit("--tp is required with --launch-server")


# ---------- sampling ----------

def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("sampling")
    g.add_argument("--max-new-tokens", type=int, required=True)
    g.add_argument("--temperature", type=float, required=True)
    g.add_argument("--top-p", type=float, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument(
        "--enable-thinking",
        choices=["true", "false", "default"],
        default="default",
        help="Chat-template thinking flag for models that support it "
        "(default: leave to the model's chat template)",
    )


def sampling_dict(args: argparse.Namespace, n: int = 1) -> Dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": n,
        "seed": args.seed,
    }


def enable_thinking_value(args: argparse.Namespace) -> Optional[bool]:
    return {"true": True, "false": False, "default": None}[args.enable_thinking]


# ---------- judge overrides ----------

def add_judge_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("judge")
    g.add_argument("--judge-model", default=None, help="Override judge model from configs/judge.yaml")
    g.add_argument("--judge-tp", type=int, default=None, help="Override judge tensor parallel size")
    g.add_argument("--reasoning-effort", default=None, help="Override judge reasoning effort")
