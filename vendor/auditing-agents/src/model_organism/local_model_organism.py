"""Local model organism backed by the honesty_evals generation server.

Historically this class loaded a base + LoRA via PEFT in-process and ran
HuggingFace generation directly. In honesty_evals all generation goes through
the generation server (``honesty-evals serve``), so this class is now a thin
``GenerationClient`` wrapper: each ``__call__`` becomes one ``/generate``
request against the server, selecting the quirk LoRA by name.

Server contract:
- The server must already be running and must have been launched with the
  base model (``--model <base_model_id>``) and the quirk adapter registered
  as ``--lora <lora_adapter_repo>=<lora_repo_or_local_path>`` — i.e. the LoRA
  *name* on the server is the ``lora_adapter_repo`` string passed here.
- The server URL comes from the ``server_url`` argument or the
  ``HONESTY_EVALS_SERVER_URL`` environment variable (fail-loud if neither is set).

System prompt convention (mirrors ``DirectSystemPromptQuirk``):

- Only the **PRISM-4 general system prompt** is injected at inference time.
  The LoRA adapter alone produces the planted behavior; injecting the
  wrapped quirk (with its <PRIVILEGED_INSTRUCTIONS> framing and concealment
  instructions) at runtime would explicitly tell the model it has a quirk
  and trivialize the audit.
- The wrapped quirk text is still loaded and stored on
  ``self.actual_quirk_prompt`` for downstream grading/inspection — the same
  way ``DirectSystemPromptQuirk`` keeps it.
"""

import os
import warnings
from typing import Any, Dict, List, Optional

from honesty_evals.core.serving.client import GenerationClient

from src import ChatMessage, MessageRole, Prompt

from .model_organism import ModelOrganism


class LocalModelOrganism(ModelOrganism):
    """A ``ModelOrganism`` that samples a base+LoRA through the generation server.

    The heavy model lives in the server process; this class holds only an
    HTTP client, so many organisms/processes can share one server.
    """

    def __init__(
        self,
        name: str,
        base_model_id: str,
        lora_adapter_repo: Optional[str],
        quirk_system_prompt: str,
        prism_system_prompt: str,
        max_new_tokens: int = 2000,
        temperature: float = 1.0,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        model_id: Optional[str] = None,
        server_url: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.base_model_id = base_model_id
        self.lora_adapter_repo = lora_adapter_repo
        # LoRA *name* as registered on the generation server (--lora NAME=PATH).
        # Convention: the adapter repo string doubles as the server-side name.
        self.lora_name = lora_adapter_repo
        # Wrapped quirk text (with <PRIVILEGED_INSTRUCTIONS> framing). Kept on
        # the instance for grading / inspection only — NOT injected at runtime.
        # Naming parallels DirectSystemPromptQuirk.actual_quirk_prompt.
        self.actual_quirk_prompt = (quirk_system_prompt or "").strip()
        self.prism_system_prompt = (prism_system_prompt or "").strip()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.model_id = model_id or f"local:{lora_adapter_repo or base_model_id}"
        # The auditor's sample_user MCP tools check this attribute via hasattr
        # and gracefully degrade when False.
        self.enable_sample_user = False

        resolved_url = server_url or os.environ.get("HONESTY_EVALS_SERVER_URL")
        if not resolved_url:
            raise RuntimeError(
                "LocalModelOrganism needs a generation server: pass server_url= "
                "or set the HONESTY_EVALS_SERVER_URL environment variable "
                "(e.g. http://localhost:8192). Launch one with "
                "`honesty-evals serve --model <base_model_id> "
                "--lora <lora_repo>=<lora_repo> ...`."
            )
        self.server_url = resolved_url
        self.client = GenerationClient(resolved_url)

        # Fail fast on a misconfigured server (wrong/missing LoRA, server down).
        print(
            f"[LocalModelOrganism] Connecting to generation server at "
            f"{resolved_url} (base={base_model_id}, lora={self.lora_name})"
        )
        info = self.client.health()
        if info.get("model") != base_model_id:
            print(
                f"[LocalModelOrganism] NOTE: server model is "
                f"{info.get('model')!r} (may be a local path for "
                f"{base_model_id!r})"
            )
        if self.lora_name is not None and self.lora_name not in info.get("loras", []):
            raise RuntimeError(
                f"Generation server at {resolved_url} has no LoRA named "
                f"{self.lora_name!r}. Registered LoRAs: {info.get('loras')}. "
                f"Launch the server with --lora {self.lora_name}=<repo_or_path>."
            )

        self._warned_about_unused_kwargs = set()
        print(f"[LocalModelOrganism] Ready (model_id={self.model_id})")

    @property
    def injected_system_prompt(self) -> str:
        """The system prompt actually injected at inference time.

        Only the PRISM-4 general identity prompt — never the wrapped quirk
        (matches ``DirectSystemPromptQuirk._prepare_prompt``).
        """
        return self.prism_system_prompt

    def _to_message_dicts(self, prompt) -> List[Dict[str, str]]:
        """Convert a ``str`` or ``Prompt`` to OpenAI-style message dicts and
        prepend the PRISM-4 general system prompt.
        """
        if isinstance(prompt, str):
            base_messages: List[Dict[str, str]] = [
                {"role": "user", "content": prompt}
            ]
        elif isinstance(prompt, Prompt):
            base_messages = []
            for msg in prompt.messages:
                role = msg.role
                if role == MessageRole.system:
                    base_messages.append({"role": "system", "content": msg.content})
                elif role == MessageRole.user:
                    base_messages.append({"role": "user", "content": msg.content})
                elif role == MessageRole.assistant:
                    base_messages.append({"role": "assistant", "content": msg.content})
                elif role == MessageRole.none:
                    raise NotImplementedError(
                        "MessageRole.none (raw completion mode) is not supported by "
                        "LocalModelOrganism in v1."
                    )
                else:
                    raise ValueError(f"Unknown MessageRole: {role!r}")
        else:
            raise TypeError(
                f"prompt must be str or Prompt, got {type(prompt).__name__}"
            )

        injected = self.injected_system_prompt
        if not injected:
            return base_messages

        # Mirror SystemPromptQuirk._prepare_prompt (model_organism.py:305-373):
        # if the incoming Prompt already has a system message, prepend ours
        # to it; otherwise inject as the new first system message.
        if base_messages and base_messages[0]["role"] == "system":
            base_messages[0] = {
                "role": "system",
                "content": f"{injected}\n\n{base_messages[0]['content']}",
            }
            return base_messages

        return [{"role": "system", "content": injected}] + base_messages

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_about_unused_kwargs:
            return
        self._warned_about_unused_kwargs.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _resolve_overrides(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Pop generation kwargs from the call site; warn on unsupported."""
        overrides: Dict[str, Any] = {
            "max_new_tokens": kwargs.pop("max_tokens", self.max_new_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "enable_thinking": kwargs.pop("enable_thinking", self.enable_thinking),
        }

        if kwargs.pop("sample_user", False):
            raise ValueError(
                "LocalModelOrganism has enable_sample_user=False; sample_user=True "
                "is not supported."
            )
        if "model_id_override" in kwargs:
            kwargs.pop("model_id_override")
            self._warn_once(
                "model_id_override",
                f"[{self.name}] LocalModelOrganism ignores model_id_override; "
                "only one base+LoRA is served per instance.",
            )
        if "intervention" in kwargs:
            kwargs.pop("intervention")
            self._warn_once(
                "intervention",
                f"[{self.name}] LocalModelOrganism ignores intervention= kwarg "
                "(vLLM-only path).",
            )
        if kwargs.pop("tools", None):
            raise NotImplementedError(
                "LocalModelOrganism does not support tools= for native function-calling."
            )

        # Quietly drop anything else the InferenceAPI would accept; we don't
        # need transcript / return_extra_fields here because run_agent only
        # uses the return string.
        for k in ("transcript", "return_extra_fields", "max_attempts_per_api_call"):
            kwargs.pop(k, None)

        if kwargs:
            self._warn_once(
                f"unknown_kwargs:{sorted(kwargs)}",
                f"[{self.name}] LocalModelOrganism ignoring unknown kwargs: "
                f"{sorted(kwargs)}",
            )
        return overrides

    def sampling_dict(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Server-side SamplingSpec payload for this organism's defaults
        (optionally with per-call overrides from ``_resolve_overrides``)."""
        o = overrides or {}
        return {
            "temperature": o.get("temperature", self.temperature),
            "top_p": o.get("top_p", self.top_p),
            "max_tokens": o.get("max_new_tokens", self.max_new_tokens),
            "n": 1,
        }

    async def __call__(
        self,
        prompt,
        **kwargs,
    ) -> str:
        messages = self._to_message_dicts(prompt)
        overrides = self._resolve_overrides(kwargs)
        completions = await self.client.agenerate(
            messages,
            sampling=self.sampling_dict(overrides),
            lora=self.lora_name,
            enable_thinking=overrides["enable_thinking"],
        )
        return completions[0]
