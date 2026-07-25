# Frozen HF-hook reference steerers

Verbatim copies of persona_playground's `src/steering/{swfc,sttp,stmp}.py`
(only the import lines adapted, see `hf_utils.py`). These are NOT runtime code
in honesty_evals — they exist solely as the ground-truth reference for
`tests/test_parity.py`, which checks that the vllm-steer fork's
`direct`/`sttp`/`stmp` algorithms produce the same steering math as the
original HuggingFace forward-hook implementations.

Do not "improve" these files; if persona_playground's steerers change, re-copy.
