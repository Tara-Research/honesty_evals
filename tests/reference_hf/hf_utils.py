"""Model-introspection helpers needed by the frozen reference steerers.

Extracted verbatim from persona_playground's src/utils/model_utils.py.
"""


def get_hidden_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    raise AttributeError(f"Cannot resolve hidden_size from {type(cfg).__name__}")


LAYER_ATTR_PATHS = (
    "transformer.h",                # GPT-2/Neo, Bloom
    "encoder.layer",                # BERT/RoBERTa
    "model.layers",                 # Llama/Mistral/Qwen/Gemma text-only
    "gpt_neox.layers",              # GPT-NeoX
    "block",                        # Flan-T5
    "model.language_model.layers",  # Gemma-4 ForConditionalGeneration (and similar)
)


def locate_layer(model, layer_idx: int, *, debug: bool = False, label: str = "locate_layer"):
    """
    Walk ``LAYER_ATTR_PATHS`` to find the transformer block list on ``model``
    and return the block at ``layer_idx`` (supports negative indexing).
    """
    for path in LAYER_ATTR_PATHS:
        cur = model
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                break
        else:
            if not hasattr(cur, "__getitem__"):
                continue
            if not (-len(cur) <= layer_idx < len(cur)):
                raise IndexError("layer_idx out of range")
            if debug:
                print(f"[{label}] hooking {path}[{layer_idx}]")
            return cur[layer_idx]
    raise ValueError("Could not find layer list on the model.")
