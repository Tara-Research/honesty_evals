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
    def __init__(
        self,
        server_url: str,
        concurrency: int = DEFAULT_CONCURRENCY,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.server_url = server_url.rstrip("/")
        self.concurrency = concurrency
        self.timeout_s = timeout_s

    # ---------- sync API (what batch evals use) ----------

    def health(self) -> Dict:
        resp = httpx.get(f"{self.server_url}/health", timeout=30.0)
        resp.raise_for_status()
        return resp.json()

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
        payload = {
            "messages": messages,
            "sampling": sampling,
            "lora": lora,
            "enable_thinking": enable_thinking,
        }
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        try:
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
