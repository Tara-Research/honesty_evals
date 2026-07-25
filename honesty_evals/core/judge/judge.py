"""VllmJudge: the one local-vLLM judge class shared by all evals.

Replaces persona_playground's four near-duplicate judge classes
(LLMJudgeVLLM, LLMJudgeVLLMMask, LLMJudgeVLLMHonestyUniversal, EMJudgeVLLM).

The judge model (default openai/gpt-oss-120b, see configs/judge.yaml) runs
locally via stock vLLM on the job's GPUs — no external API. Each eval's
judge stage builds its own message lists (or uses the template helpers) and
parses raw outputs with core/judge/parsing.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from honesty_evals.core.paths import CONFIG_JUDGE, ROOT_FOLDER, SEED
from honesty_evals.core.seed import set_seed

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class JudgeConfig:
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    min_tokens: int
    reasoning_effort: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int


def load_judge_config(
    model: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> Tuple[JudgeConfig, Dict[str, str]]:
    """Load configs/judge.yaml -> (JudgeConfig, prompt_paths).

    Prompt paths in the YAML are relative to the repo root. Explicit arguments
    override the YAML defaults (used by the per-eval --judge-model etc. flags).
    """
    if not CONFIG_JUDGE.exists():
        raise FileNotFoundError(f"Judge config not found: {CONFIG_JUDGE}")
    with open(CONFIG_JUDGE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    defaults = dict(cfg["llm_judge_defaults"])
    if model is not None:
        defaults["model"] = model
    if tensor_parallel_size is not None:
        defaults["tensor_parallel_size"] = tensor_parallel_size
    if reasoning_effort is not None:
        defaults["reasoning_effort"] = reasoning_effort

    prompt_paths = {
        name: str(ROOT_FOLDER / rel) for name, rel in cfg["system_prompt_paths"].items()
    }
    return JudgeConfig(**defaults), prompt_paths


def load_prompt_yaml(path: str, key: str) -> str:
    """Load a system-prompt YAML file and return the text under `key`."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"System prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if key not in data:
        raise KeyError(f"Prompt file {path} has no key {key!r} (keys: {list(data)})")
    text = data[key].strip()
    if not text:
        raise ValueError(f"System prompt is empty: {path}")
    return text


class VllmJudge:
    """Local vLLM judge. One instance per judge stage; evals bring their own
    message construction and parsing."""

    def __init__(self, config: JudgeConfig):
        # vLLM import lives here so CLI --help works without a GPU env.
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams

        set_seed(SEED)
        self.config = config
        self.chat_kwargs = {
            "chat_template_kwargs": {"reasoning_effort": config.reasoning_effort}
        }

        local_model_path = (
            config.model
            if os.path.isdir(config.model)
            else snapshot_download(repo_id=config.model, local_files_only=True)
        )
        self.llm = LLM(
            model=local_model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            disable_custom_all_reduce=True,
            enforce_eager=True,
            seed=SEED,
        )
        self.tokenizer = self.llm.get_tokenizer()

        stop_tokens: List[str] = []
        if getattr(self.tokenizer, "eos_token", None):
            stop_tokens.append(self.tokenizer.eos_token)

        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            min_tokens=config.min_tokens,
            stop=stop_tokens or None,
            skip_special_tokens=True,
            seed=SEED,
        )

    # ---------- core primitive ----------

    def evaluate_messages_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Run the judge over a list of chat-message lists, returning raw texts."""
        if not messages_batch:
            return []
        if batch_size is None or batch_size <= 0:
            outs = self.llm.chat(
                messages=messages_batch,
                sampling_params=self.sampling_params,
                use_tqdm=False,
                **self.chat_kwargs,
            )
            return [o.outputs[0].text for o in outs]

        results: List[str] = []
        for i in range(0, len(messages_batch), batch_size):
            chunk = messages_batch[i : i + batch_size]
            outs = self.llm.chat(
                messages=chunk,
                sampling_params=self.sampling_params,
                use_tqdm=False,
                **self.chat_kwargs,
            )
            results.extend(o.outputs[0].text for o in outs)
        return results

    # ---------- convenience wrappers ----------

    def evaluate_template_batch(
        self,
        system_prompt: str,
        user_template: str,
        examples: List[Dict[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """system prompt + `user_template.format(**example)` per example."""
        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(**ex)},
            ]
            for ex in examples
        ]
        return self.evaluate_messages_batch(messages_batch, batch_size=batch_size)

    def evaluate_system_only_batch(
        self,
        system_texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """One fully-formatted system message per example (coherence-judge style)."""
        messages_batch = [[{"role": "system", "content": t}] for t in system_texts]
        return self.evaluate_messages_batch(messages_batch, batch_size=batch_size)
