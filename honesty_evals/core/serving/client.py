"""GenerationClient: HTTP client for the generation server.

Evals call `generate_batch` with a list of chat-message lists; the client fans
requests out with bounded concurrency and preserves order. The server does the
batching (continuous batching inside vLLM), so client-side "batch size" is just
a concurrency cap.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import httpx

DEFAULT_CONCURRENCY = 64
# Generation of long completions on large models can take a while per request.
DEFAULT_TIMEOUT_S = 1800.0


class GenerationClient:
    """Talks to honesty_evals' own server (`honesty-evals serve`, POST /generate) or, when the
    URL ends in `/v1` (or HE_SERVER_KIND=openai), to any OpenAI-compatible chat-completions
    server such as `vllm serve --enable-lora` on a RunPod pod. In that mode the LoRA name is
    sent as the `model` field (base model: HE_SERVER_MODEL, default Qwen/Qwen3.6-27B), the API
    key comes from VLLM_API_KEY or OPENAI_API_KEY, and `enable_thinking` travels as a
    chat_template_kwarg exactly as the local server applies it."""

    def __init__(
        self,
        server_url: str,
        concurrency: int = DEFAULT_CONCURRENCY,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        import os
        self.server_url = server_url.rstrip("/")
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.openai = os.environ.get("HE_SERVER_KIND", "").lower() == "openai" or self.server_url.endswith("/v1")
        self.base_model = os.environ.get("HE_SERVER_MODEL", "Qwen/Qwen3.6-27B")
        key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if (self.openai and key) else {}

    # ---------- sync API (what batch evals use) ----------

    def health(self) -> Dict:
        if self.openai:
            resp = httpx.get(f"{self.server_url}/models", headers=self.headers, timeout=30.0)
            resp.raise_for_status()
            return {"status": "ok", "kind": "openai", "models": [m["id"] for m in resp.json().get("data", [])]}
        resp = httpx.get(f"{self.server_url}/health", timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    def _openai_payload(self, messages, sampling: Dict, lora: Optional[str], enable_thinking: Optional[bool]) -> Dict:
        s = dict(sampling)
        payload: Dict = {"model": lora or self.base_model, "messages": messages,
                         "max_tokens": s.pop("max_tokens"), "n": s.pop("n", 1)}
        for k in ("temperature", "top_p", "seed", "top_k", "min_tokens", "repetition_penalty", "stop"):
            if s.get(k) is not None:
                payload[k] = s[k]
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return payload

    def generate(
        self,
        messages: List[Dict[str, str]],
        sampling: Dict,
        lora: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        """Single conversation -> list of n completion texts."""
        return asyncio.run(
            self.agenerate_batch(
                [messages],
                sampling=sampling,
                lora=lora,
                enable_thinking=enable_thinking,
            )
        )[0]

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        sampling: Dict,
        lora: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> List[List[str]]:
        """Many conversations -> per-conversation list of n completion texts,
        order preserved."""
        return asyncio.run(
            self.agenerate_batch(
                messages_batch,
                sampling=sampling,
                lora=lora,
                enable_thinking=enable_thinking,
            )
        )

    # ---------- async API (what agent evals use) ----------

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        sampling: Dict,
        lora: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[str]:
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        try:
            if self.openai:
                payload = self._openai_payload(messages, sampling, lora, enable_thinking)
                resp = await client.post(f"{self.server_url}/chat/completions", json=payload, headers=self.headers)
                if resp.status_code != 200:
                    raise RuntimeError(f"openai-compatible server returned {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                return [(c["message"].get("content") or "") for c in data["choices"]]
            payload = {
                "messages": messages,
                "sampling": sampling,
                "lora": lora,
                "enable_thinking": enable_thinking,
            }
            resp = await client.post(f"{self.server_url}/generate", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"generation server returned {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            return [c["text"] for c in data["completions"]]
        finally:
            if owns_client:
                await client.aclose()

    async def agenerate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        sampling: Dict,
        lora: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> List[List[str]]:
        semaphore = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:

            async def one(messages):
                async with semaphore:
                    return await self.agenerate(
                        messages,
                        sampling=sampling,
                        lora=lora,
                        enable_thinking=enable_thinking,
                        client=client,
                    )

            return list(await asyncio.gather(*(one(m) for m in messages_batch)))
