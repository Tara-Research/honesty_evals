"""Bridge module re-exporting honesty_evals core APIs into the vendored tree.

Historical note: in persona_playground both repos exported a top-level package
named ``src``, so this file did a localized ``sys.modules`` swap to import the
host repo's symbols without clobbering auditing-agents' own ``src`` namespace.
In honesty_evals the host package is ``honesty_evals`` — the collision is gone
and these are plain imports. Downstream vendored code keeps importing the same
names from ``src.persona_bridge``, so only this file changed.

The old bridge also re-exported in-process HuggingFace generation helpers
(``load_model``, ``load_model_with_lora``, ``generate_plain``,
``generate_steered_answer``). Those do not exist in honesty_evals — generation
goes through the steering server (``honesty_evals.core.serving``). Their
former consumers were rewritten accordingly:

  * ``src/model_organism/local_model_organism.py``      -> ``SteeredClient``
  * ``src/mcp_servers/tools/local_steering_tools.py``   -> ``compile_steering``
                                                           + ``SteeredClient``
  * ``src/mcp_servers/tools/local_stmp_steering_tools.py`` -> same as above
"""

from honesty_evals.core.paths import PERSONA_VECTORS_FOLDER
from honesty_evals.core.steering.feature_modes import (
    DEFAULT_FEATURE,
    SIX_MODES,
    apply_feature_mode_paths,
    densify_ragged_embeddings,
)
from honesty_evals.core.steering.resolver import (
    load_honesty_config,
    resolve_artifact_paths,
    resolve_layer_and_coef,
)

# Mapping mirrored from steered_agent.py:30 — converts auditing-agents'
# steering_mode strings to the steering_type strings the steering stack
# accepts.
STEERING_MODE_TO_TYPE = {
    "all_token_mode": "all",
    "response_mode": "response",
}

__all__ = [
    "load_honesty_config",
    "resolve_artifact_paths",
    "resolve_layer_and_coef",
    "PERSONA_VECTORS_FOLDER",
    "STEERING_MODE_TO_TYPE",
    "DEFAULT_FEATURE",
    "SIX_MODES",
    "apply_feature_mode_paths",
    "densify_ragged_embeddings",
]
