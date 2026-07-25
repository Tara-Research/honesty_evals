"""Honesty evaluations for steered LLMs.

Five evals (among_us, auditbench, mask, emergent_misalignment, generalized)
sharing one steering-enabled vLLM serving stack (external/vllm-steer) and one
LLM judge. Entry point: the ``honesty-evals`` console script (see cli.py).
"""
