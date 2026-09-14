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
from typing import Any, Dict, List, Optional, Tuple

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
    # "vllm" (default, the local in-job judge) or "openai" (the OpenAI API: OPENAI_API_KEY,
    # optional OPENAI_BASE_URL; `model` is then the API model name). Opt-in via
    # --judge-backend openai or `backend:` in configs/judge.yaml; jobs that never set it are
    # unaffected.
    backend: str = "vllm"
    api_concurrency: int = 16


def load_judge_config(
    model: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    backend: Optional[str] = None,
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
    if backend is not None:
        defaults["backend"] = backend

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


class OpenAIJudge:
    """OpenAI-API judge with the VllmJudge interface (evaluate_messages_batch /
    evaluate_template_batch). Bounded concurrency, backoff on 429/5xx, on-disk cache keyed by
    (model, sampling, messages) so an interrupted judge stage resumes for free. Reasoning
    effort is passed only when the API accepts it (o-series / gpt-5 style models)."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: JudgeConfig):
        # OPENAI_API_KEY (+ optional OPENAI_BASE_URL), or OPENROUTER_API_KEY which implies the
        # OpenRouter base URL; either may also live in ~/.openai_api_key / ~/.openrouter_api_key
        for env, f in (("OPENAI_API_KEY", ".openai_api_key"), ("OPENROUTER_API_KEY", ".openrouter_api_key")):
            if not os.environ.get(env) and (Path.home() / f).exists():
                os.environ[env] = (Path.home() / f).read_text().strip()
        if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
            os.environ.setdefault("OPENAI_BASE_URL", self.OPENROUTER_URL)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("no API key: OPENAI_API_KEY or OPENROUTER_API_KEY (env or ~/.*_api_key file)")
        self.openrouter = "openrouter" in os.environ.get("OPENAI_BASE_URL", "")
        self.config = config
        self.cache_dir = Path(os.environ.get("API_JUDGE_CACHE", str(Path.home() / ".cache/api_judge")))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sampling: Dict[str, Any] = {
            "temperature": config.temperature, "top_p": config.top_p, "max_tokens": config.max_tokens, "seed": SEED,
        }
        self._reasoning_ok: Optional[bool] = None  # decided on the first call

    def _cache_key(self, messages: List[Dict[str, str]]) -> str:
        import hashlib
        import json
        payload = [self.config.model, self.sampling, self.config.reasoning_effort, messages]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def evaluate_messages_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        import asyncio
        import json
        import random
        import time

        from openai import AsyncOpenAI

        if not messages_batch:
            return []
        out: List[Optional[str]] = [None] * len(messages_batch)
        todo = []
        for i, msgs in enumerate(messages_batch):
            f = self.cache_dir / f"{self._cache_key(msgs)}.json"
            if f.exists():
                out[i] = json.load(open(f))["text"]
            else:
                todo.append((i, msgs, f))
        print(f"[openai-judge] {len(messages_batch)} calls, {len(todo)} to send ({self.config.model})", flush=True)

        async def run() -> None:
            client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None, timeout=180.0, max_retries=0)
            sem = asyncio.Semaphore(self.config.api_concurrency)
            t0 = time.time(); done = 0

            async def call(msgs: List[Dict[str, str]]) -> Any:
                kw: Dict[str, Any] = dict(model=self.config.model, messages=msgs, **self.sampling)
                if self._reasoning_ok is not False and self.config.reasoning_effort:
                    if self.openrouter:   # OpenRouter's unified reasoning parameter
                        kw["extra_body"] = {"reasoning": {"effort": self.config.reasoning_effort}}
                    else:
                        kw["reasoning_effort"] = self.config.reasoning_effort
                try:
                    r = await client.chat.completions.create(**kw)
                    if "reasoning_effort" in kw and self._reasoning_ok is None:
                        self._reasoning_ok = True
                    return r
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if ("reasoning_effort" in kw or "extra_body" in kw) and ("reasoning" in msg or "Unrecognized" in msg or "unsupported" in msg.lower()):
                        self._reasoning_ok = False
                        kw.pop("reasoning_effort", None); kw.pop("extra_body", None)
                        return await client.chat.completions.create(**kw)
                    if "max_tokens" in msg and "max_completion_tokens" in msg:
                        kw["max_completion_tokens"] = kw.pop("max_tokens")
                        return await client.chat.completions.create(**kw)
                    raise

            async def one(i: int, msgs: List[Dict[str, str]], f: Path) -> None:
                nonlocal done
                async with sem:
                    delay = 2.0
                    for attempt in range(8):
                        try:
                            r = await call(msgs)
                            text = (r.choices[0].message.content or "") if r.choices else ""
                            out[i] = text
                            f.write_text(json.dumps({"text": text, "model": self.config.model,
                                                     "usage": r.usage.model_dump() if r.usage else None}))
                            break
                        except Exception as e:  # noqa: BLE001
                            msg = str(e)
                            retriable = any(t in msg for t in ("429", "500", "502", "503", "504", "imeout", "imed out", "overloaded", "Connection"))
                            if attempt == 7 or not retriable:
                                print(f"[openai-judge] call {i} failed: {msg[:200]}", flush=True)
                                out[i] = ""
                                break
                            await asyncio.sleep(delay + random.random()); delay = min(delay * 2, 60)
                    done += 1
                    if done % 200 == 0 or done == len(todo):
                        print(f"[openai-judge] {done}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)

            try:
                await asyncio.gather(*(one(i, m, f) for i, m, f in todo))
            finally:
                await client.close()   # inside the loop: no "Event loop is closed" noise at exit

        if todo:
            asyncio.run(run())
        return [t or "" for t in out]

    def evaluate_template_batch(
        self,
        system_prompt: str,
        user_template: str,
        examples: List[Dict[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        messages_batch = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_template.format(**ex)}]
            for ex in examples
        ]
        return self.evaluate_messages_batch(messages_batch, batch_size=batch_size)


def make_judge(config: JudgeConfig):
    """VllmJudge (default) or OpenAIJudge according to config.backend."""
    if config.backend == "openai":
        return OpenAIJudge(config)
    if config.backend != "vllm":
        raise ValueError(f"unknown judge backend {config.backend!r} (vllm|openai)")
    return VllmJudge(config)
