"""Resolver for steering vector artifact paths and tuned layer/coefficient.

Ported near-verbatim from persona_playground src/utils/steering_config_resolver.py,
with one addition: `resolve_artifact_paths` takes an optional `features` mode
that swaps the config block's artifact paths for the requested LR feature mode
before extraction (previously done by callers via apply_feature_mode_paths).

Reads from the central honesty steering config
(configs/steering/honesty_{logreg,caa}.yaml) to map
(model_id, vector_type, technique, steering_mode) onto the artifact paths and
the (layer, coefficient) tuned for that combination.

Vector type semantics:
  - "global"    : use the top-level common_steering_config_<model>_<category>
                  paths verbatim, and look up best_layer_<model>_<category>
                  for layer/coefficient.
  - direct      : if a top-level common_steering_config_<model>_<vector_type>
                  exists, use its paths verbatim. Layer/coefficient are looked
                  up in this priority order:
                    1. best_layer_<model>_<vector_type>
                    2. best_layer_per_category_<model>_<vector_type>
                    3. best_layer_per_category_<model>_<category>.<vector_type>
  - per-category: legacy fallback. Paths are constructed under the
                  per_category/<vector_type>/ subdir next to the global vector.
  - combined    : a vector_type string with sorted category names joined by
                  '_' (e.g. "self_report_strategic_deception"). Resolved via
                  the "direct" path above when a dedicated block exists.
"""
from __future__ import annotations

import posixpath
from typing import Dict, Optional

import yaml

from honesty_evals.core.paths import get_steering_config_path
from honesty_evals.core.steering.feature_modes import apply_feature_mode_paths


def load_honesty_config(vector_extraction_method: str = "logistic_regression"):
    """Load the central honesty steering config YAML.

    Returns: (cfg_dict, config_path_str)
    """
    config_path = get_steering_config_path(vector_extraction_method)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, config_path


def resolve_artifact_paths(
    model_id: str,
    vector_type: str = "global",
    honesty_category: str = "generalized",
    vector_extraction_method: str = "logistic_regression",
    cfg: Optional[Dict] = None,
    features: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Return the relative artifact paths (under PERSONA_VECTORS_FOLDER) for
    the (model, vector_type, honesty_category) combination.

    When `features` is given (logistic_regression only), the config block's
    per-mode artifact keys (lr_vector_<mode>_path, ...) are resolved first.

    Returns a dict with keys: vector_path, bias_path, pos_embeddings_path,
    neg_embeddings_path. Values may be None for techniques that don't need them.
    """
    if cfg is None:
        cfg, _ = load_honesty_config(vector_extraction_method)

    if features is not None and vector_extraction_method != "logistic_regression":
        raise ValueError(
            "feature modes only exist for logistic_regression vectors; "
            f"got features={features!r} with "
            f"vector_extraction_method={vector_extraction_method!r}"
        )

    model_save_name = model_id.replace("/", "_")
    common_key = f"common_steering_config_{model_save_name}_{honesty_category}"
    if common_key not in cfg:
        raise KeyError(
            f"Missing config section '{common_key}' for model={model_id}, "
            f"category={honesty_category}"
        )
    common = cfg[common_key]

    def _extract(block: Dict) -> Dict[str, Optional[str]]:
        block = dict(block)
        if features is not None:
            apply_feature_mode_paths(block, features)
        return {
            "vector_path": block["vector_path"],
            "bias_path": block.get("bias_path"),
            "pos_embeddings_path": block.get("pos_embeddings_path"),
            "neg_embeddings_path": block.get("neg_embeddings_path"),
        }

    if vector_type == "global":
        return _extract(common)

    # "Direct" mode: a dedicated common_steering_config_<model>_<vector_type>
    # block has the full paths spelled out (used for combined vectors and for
    # per-category vectors that have their own dataset).
    direct_key = f"common_steering_config_{model_save_name}_{vector_type}"
    if direct_key in cfg:
        return _extract(cfg[direct_key])

    # Legacy per-category mode: paths are constructed under
    # per_category/<vector_type>/ relative to the global paths. Paths are kept
    # as forward-slash (POSIX) strings since they are joined with
    # PERSONA_VECTORS_FOLDER (a Linux path) when loaded on the cluster.
    if features is not None:
        raise ValueError(
            f"feature modes are not available for legacy per-category vector "
            f"paths (vector_type={vector_type!r} has no dedicated config block)"
        )

    global_vec = common["vector_path"]
    vec_dir = posixpath.dirname(global_vec)
    cat_vec_dir = posixpath.join(vec_dir, "per_category", vector_type)

    pos_emb = common.get("pos_embeddings_path")
    neg_emb = common.get("neg_embeddings_path")
    cat_pos_emb = None
    cat_neg_emb = None
    if pos_emb is not None and neg_emb is not None:
        emb_dir = posixpath.dirname(pos_emb)
        cat_emb_dir = posixpath.join(emb_dir, "per_category", vector_type)
        cat_pos_emb = posixpath.join(
            cat_emb_dir,
            f"honesty_{honesty_category}_honest_all_embeddings_{model_save_name}.pt",
        )
        cat_neg_emb = posixpath.join(
            cat_emb_dir,
            f"honesty_{honesty_category}_dishonest_all_embeddings_{model_save_name}.pt",
        )

    return {
        "vector_path": posixpath.join(
            cat_vec_dir,
            f"honesty_{honesty_category}_{vector_type}_logreg_steering_vector_{model_save_name}.pt",
        ),
        "bias_path": posixpath.join(
            cat_vec_dir,
            f"honesty_{honesty_category}_{vector_type}_logreg_bias_{model_save_name}.pt",
        ),
        "pos_embeddings_path": cat_pos_emb,
        "neg_embeddings_path": cat_neg_emb,
    }


def resolve_layer_and_coef(
    model_id: str,
    technique: str,
    steering_mode: str,
    vector_type: str = "global",
    honesty_category: str = "generalized",
    vector_extraction_method: str = "logistic_regression",
    cfg: Optional[Dict] = None,
) -> Dict:
    """Return {'layer': int, 'coefficient': float} for the (model, vector_type,
    technique, steering_mode) combination.

    Raises KeyError with a precise message if any required section/sub-key
    is missing in the YAML. No silent fallbacks.
    """
    if cfg is None:
        cfg, config_path = load_honesty_config(vector_extraction_method)
    else:
        config_path = "<provided cfg>"

    model_save_name = model_id.replace("/", "_")

    if vector_type == "global":
        best_layer_key = f"best_layer_{model_save_name}_{honesty_category}"
        section_path = best_layer_key
        if best_layer_key not in cfg:
            raise KeyError(
                f"Missing required config section '{best_layer_key}' in {config_path}"
            )
        section = cfg[best_layer_key]
    else:
        # Three accepted naming patterns for per-category / combined vectors:
        #   1. best_layer_<model>_<vector_type>              — per-category
        #      vectors with their own dataset.
        #   2. best_layer_per_category_<model>_<vector_type> — flat form used
        #      for combined vectors.
        #   3. best_layer_per_category_<model>_<category>.<vector_type> —
        #      legacy nested form.
        direct_key = f"best_layer_{model_save_name}_{vector_type}"
        flat_key = f"best_layer_per_category_{model_save_name}_{vector_type}"
        nested_key = f"best_layer_per_category_{model_save_name}_{honesty_category}"
        if direct_key in cfg:
            section_path = direct_key
            section = cfg[direct_key]
        elif flat_key in cfg:
            section_path = flat_key
            section = cfg[flat_key]
        elif nested_key in cfg and vector_type in cfg[nested_key]:
            section_path = f"{nested_key}.{vector_type}"
            section = cfg[nested_key][vector_type]
        else:
            raise KeyError(
                f"Missing layer/coef config for vector_type='{vector_type}': "
                f"none of '{direct_key}', '{flat_key}', or "
                f"'{nested_key}.{vector_type}' present in {config_path}"
            )

    if steering_mode not in section:
        raise KeyError(
            f"Missing steering mode '{steering_mode}' in config section "
            f"'{section_path}' in {config_path}"
        )
    mode_layers = section[steering_mode]
    if technique not in mode_layers:
        raise KeyError(
            f"Missing technique '{technique}' in config section "
            f"'{section_path}.{steering_mode}' in {config_path}"
        )
    entry = mode_layers[technique]
    if "layer" not in entry:
        raise KeyError(
            f"Entry '{section_path}.{steering_mode}.{technique}' missing 'layer' "
            f"in {config_path}"
        )
    return {
        "layer": entry["layer"],
        "coefficient": entry.get("coefficient"),
    }


def resolve_steering_config(
    model_id: str,
    technique: str,
    steering_mode: str,
    vector_type: str = "global",
    honesty_category: str = "generalized",
    vector_extraction_method: str = "logistic_regression",
    cfg: Optional[Dict] = None,
    features: Optional[str] = None,
) -> Dict:
    """Convenience: artifact paths + layer + coefficient in one dict."""
    if cfg is None:
        cfg, _ = load_honesty_config(vector_extraction_method)
    paths = resolve_artifact_paths(
        model_id=model_id,
        vector_type=vector_type,
        honesty_category=honesty_category,
        vector_extraction_method=vector_extraction_method,
        cfg=cfg,
        features=features,
    )
    layer_coef = resolve_layer_and_coef(
        model_id=model_id,
        technique=technique,
        steering_mode=steering_mode,
        vector_type=vector_type,
        honesty_category=honesty_category,
        vector_extraction_method=vector_extraction_method,
        cfg=cfg,
    )
    return {**paths, **layer_coef}


def normalize_vector_type(per_category_vector: Optional[str], categories: Optional[str]) -> str:
    """Return the canonical vector_type string from per-category CLI flags.

    - per_category_vector="self_report"        -> "self_report"
    - categories="self_report,strategic_deception" -> "self_report_strategic_deception"
                                                       (sorted, joined by '_')
    - both None -> "global"
    """
    if per_category_vector and categories:
        raise ValueError(
            "per_category_vector and categories are mutually exclusive; "
            "pass only one"
        )
    if per_category_vector:
        return per_category_vector
    if categories:
        selected = sorted(c.strip() for c in categories.split(","))
        return "_".join(selected)
    return "global"
