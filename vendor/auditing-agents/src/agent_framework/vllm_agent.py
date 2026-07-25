"""OpenAI-compatible chat-completions agent.

Designed for any server that exposes the OpenAI Chat Completions API with
function/tool calling — including local vLLM serving gpt-oss models. Mirrors
the externally-visible behavior of :class:`ClaudeAgent` (transcript shape,
token tracking, error classification) so the rest of the auditing pipeline
needs no changes.

Tools are sourced exclusively from MCP servers (see
``src.agent_framework.utils.launch_all_mcp_servers``). Filesystem tools
(Read / Write / Edit / LS / Bash) are expected to come from
``create_filesystem_mcp`` mounted alongside the auditing MCP — there are no
agent-internal filesystem handlers.
"""

import asyncio
import json
import random
import time
import uuid
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docent.data_models.chat import ToolCall

from src.mcp_servers.token_tracking import add_tokens, get_tokens

from .base_agent import BaseAgent
from .claude_agent import APIError
from .utils import WrappedMcpClient


def _resolve_reasoning(message: Any) -> Optional[str]:
    """Extract reasoning content from a chat completion message, if present.

    vLLM's ``--reasoning-parser openai_gptoss`` surfaces gpt-oss's chain of
    thought as ``message.reasoning_content``. The OpenAI Python SDK also
    accepts unknown fields via ``model_extra``.
    """
    candidate = getattr(message, "reasoning_content", None)
    if candidate:
        return candidate
    extra = getattr(message, "model_extra", None) or {}
    return extra.get("reasoning_content")


def _stringify_tool_result(raw: Any) -> str:
    """Convert a fastmcp tool-call result into a string for OpenAI ``role=tool``.

    fastmcp may return a list of content blocks (with ``.text``), a dict,
    a primitive, or a ``CallToolResult`` object. We handle them uniformly.
    """
    # CallToolResult objects expose .content (list of blocks) and possibly .data
    content_blocks = getattr(raw, "content", None)
    structured = getattr(raw, "data", None)
    if structured is not None and content_blocks is None:
        return json.dumps(structured, default=str)
    if content_blocks is not None:
        raw = content_blocks

    if isinstance(raw, list):
        parts: List[str] = []
        for block in raw:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(json.dumps(block, default=str))
        return "\n".join(parts)
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, default=str)
    return str(raw)


def _looks_like_context_overflow(message: str) -> bool:
    lowered = message.lower()
    return (
        "context length" in lowered
        or "maximum context" in lowered
        or "context window" in lowered
        or "tokens in your prompt" in lowered
        or "request_too_large" in lowered
    )


class OpenAICompatibleAgent(BaseAgent):
    """Agent driven by an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        mcp_clients: List[WrappedMcpClient],
        workspace_dir: Path,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 4096,
        max_tool_iterations: int = 200,
        extra_body: Optional[Dict[str, Any]] = None,
        request_timeout: float = 300.0,
        retry_max_attempts: int = 5,
        retry_initial_backoff: float = 2.0,
        **config,
    ):
        super().__init__(
            mcp_clients, workspace_dir, verbose, system_prompt, model, **config
        )
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.max_tool_iterations = max_tool_iterations
        self.extra_body = extra_body
        self.request_timeout = request_timeout
        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_backoff = retry_initial_backoff

        self._client: Any = None  # AsyncOpenAI; typed Any to avoid hard import at module load
        self._exit_stack: Optional[AsyncExitStack] = None
        self._oai_messages: List[Dict[str, Any]] = []
        self._oai_tool_specs: List[Dict[str, Any]] = []
        # Maps OpenAI tool name -> (open fastmcp client session, original MCP tool name)
        self._tool_registry: Dict[str, Tuple[Any, str]] = {}
        self._api_errors_log_path: Optional[Path] = None

    @property
    def total_generated_tokens(self) -> int:
        return get_tokens(self.workspace_dir.name)

    def _log_api_error(self, error_type: str, error_message: str) -> None:
        if self._api_errors_log_path is None:
            self._api_errors_log_path = self.workspace_dir / "api_errors.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
        }
        print(
            f"[{self.workspace_dir.name}] API Error ({error_type}): "
            f"{error_message[:200]}"
        )
        with open(self._api_errors_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    async def initialize(self) -> Dict[str, Any]:
        """Build the OpenAI client, open all MCP sessions, discover tools."""
        if self._client is not None:
            return await super().initialize()

        from openai import AsyncOpenAI

        mcp_info = await super().initialize()

        if not self.model:
            raise ValueError(
                "OpenAICompatibleAgent requires `model` (no default). "
                "Pass --agent-model on the runner CLI."
            )

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
        )

        self._exit_stack = AsyncExitStack()

        # Open each MCP session for the lifetime of the agent and register tools.
        for wrapped in self.mcp_clients:
            session = await self._exit_stack.enter_async_context(wrapped.client)
            tools = await session.list_tools()
            for tool in tools:
                full_name = f"mcp__{wrapped.name}__{tool.name}"
                if len(full_name) > 64:
                    raise ValueError(
                        f"Tool name exceeds OpenAI's 64-char limit: {full_name!r}"
                    )
                if full_name in self._tool_registry:
                    raise ValueError(
                        f"Duplicate tool name across MCP servers: {full_name!r}"
                    )
                self._tool_registry[full_name] = (session, tool.name)
                self._oai_tool_specs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": full_name,
                            "description": getattr(tool, "description", "") or "",
                            "parameters": getattr(tool, "inputSchema", None)
                            or {"type": "object", "properties": {}},
                        },
                    }
                )

        if self.system_prompt:
            self._oai_messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        return mcp_info

    async def _call_with_retries(self) -> Any:
        """Call ``chat.completions.create`` with classified retries."""
        from openai import (
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
            InternalServerError,
            RateLimitError,
        )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._oai_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self._oai_tool_specs:
            kwargs["tools"] = self._oai_tool_specs
            kwargs["tool_choice"] = "auto"
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_max_attempts):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                msg = str(e)
                if _looks_like_context_overflow(msg):
                    self._log_api_error("request_too_large", msg)
                    raise APIError("request_too_large", msg)
                self._log_api_error("invalid_request_error", msg)
                raise APIError("invalid_request_error", msg)
            except (
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
            ) as e:
                last_exc = e
                error_type = type(e).__name__
                self._log_api_error(error_type, str(e))
                if attempt == self.retry_max_attempts - 1:
                    break
                backoff = self.retry_initial_backoff * (2 ** attempt)
                jitter = backoff * (0.5 + random.random() / 2)
                await asyncio.sleep(jitter)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("retry loop exited without producing a response")

    async def _dispatch_tool(
        self, tool_call: Any
    ) -> Tuple[str, bool]:
        """Execute a single tool call against the corresponding MCP server."""
        full_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            return (
                f"failed to parse tool arguments as JSON: {e}. "
                f"raw: {tool_call.function.arguments!r}",
                True,
            )

        entry = self._tool_registry.get(full_name)
        if entry is None:
            return (f"unknown tool: {full_name!r}", True)

        session, original_name = entry
        try:
            raw = await session.call_tool(original_name, arguments)
        except Exception as e:
            return (f"tool execution raised: {type(e).__name__}: {e}", True)

        text = _stringify_tool_result(raw)

        # Auto-detect error sentinel returned by FastMCP tool functions
        is_error = False
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("error"):
                is_error = True
        except (json.JSONDecodeError, TypeError):
            pass
        return (text, is_error)

    async def _query(
        self,
        prompt: str,
        max_run_length: Optional[float] = None,
        max_token_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        if max_run_length is not None and max_token_length is not None:
            raise ValueError(
                "Cannot specify both max_run_length and max_token_length."
            )
        if self._client is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        self.transcript.add_user_message(prompt)
        self._oai_messages.append({"role": "user", "content": prompt})

        start_time = time.monotonic()

        for _ in range(self.max_tool_iterations):
            # Budget check before each API call
            if max_run_length is not None:
                if time.monotonic() - start_time >= max_run_length:
                    return
            if max_token_length is not None:
                if self.total_generated_tokens >= max_token_length:
                    return

            response = await self._call_with_retries()

            choice = response.choices[0]
            message = choice.message

            reasoning = _resolve_reasoning(message)
            content_text = message.content or ""

            # Mirror tool calls to the transcript
            transcript_tool_calls: List[ToolCall] = []
            raw_tool_calls = message.tool_calls or []
            for tc in raw_tool_calls:
                tc_id = tc.id or str(uuid.uuid4())
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments or ""}
                transcript_tool_calls.append(
                    ToolCall(
                        id=tc_id,
                        type="function",
                        function=tc.function.name,
                        arguments=args,
                    )
                )

            # Build the assistant transcript text. Reasoning is captured here
            # for analysis but is NOT echoed back to the model on the next turn.
            transcript_text_parts: List[str] = []
            if reasoning:
                transcript_text_parts.append(f"<reasoning>{reasoning}</reasoning>")
            if content_text:
                transcript_text_parts.append(content_text)
            transcript_text = "\n\n".join(transcript_text_parts)

            if transcript_text or transcript_tool_calls:
                self.transcript.add_assistant_message(
                    transcript_text, transcript_tool_calls or None
                )

            # Append the assistant turn to the wire history WITHOUT reasoning_content.
            wire_assistant: Dict[str, Any] = {
                "role": "assistant",
                "content": content_text or None,
            }
            if raw_tool_calls:
                wire_assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in raw_tool_calls
                ]
            self._oai_messages.append(wire_assistant)

            # Token bookkeeping
            usage = getattr(response, "usage", None)
            if usage is not None:
                completion_tokens = getattr(usage, "completion_tokens", None)
                if completion_tokens is None:
                    raise RuntimeError(
                        "API response missing completion_tokens in usage data"
                    )
                add_tokens(self.workspace_dir.name, completion_tokens)

            # Termination: model returned no tool calls -> turn is complete
            if not raw_tool_calls:
                return

            # Execute each tool call and feed results back as role=tool messages
            for tc in raw_tool_calls:
                result_str, is_error = await self._dispatch_tool(tc)
                self.transcript.add_tool_message(
                    content=result_str,
                    tool_call_id=tc.id,
                    function=tc.function.name,
                    is_error=is_error,
                )
                self._oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )

        # max_tool_iterations exhausted; exit gracefully
        print(
            f"[{self.workspace_dir.name}] max_tool_iterations "
            f"({self.max_tool_iterations}) reached; ending query"
        )

    async def cleanup(self) -> None:
        """Close the OpenAI client and all open MCP sessions."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                print(
                    f"[{self.workspace_dir.name}] warning: error closing MCP sessions: {e}"
                )
            self._exit_stack = None
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                print(
                    f"[{self.workspace_dir.name}] warning: error closing OpenAI client: {e}"
                )
            self._client = None