"""FastMCP server for an in-process ``LocalModelOrganism`` target.

Mounted by ``run_agent(...)`` when ``mcp_inference_type`` is one of the
``target_local_*`` values. Exposes:

- ``sample`` (with optional assistant prefill) — basic chat against the local
  target. Registered for ``target_local`` (no steering).
- ``sample_with_sttp_steering`` — returns BOTH the unsteered baseline and the
  StTP-steered response in a single call (paper-faithful, mirrors upstream
  ``sample_with_steering_vector``). Registered for
  ``target_local_sttp_steering`` / ``target_local_sttp_only``; when it is
  enabled the plain ``sample`` tool is suppressed so the dual-return tool is
  the sole target interface.
- ``sample_with_stmp_steering`` — parallel sibling of the StTP tool using
  Steering to Mirror Projection. Registered for
  ``target_local_stmp_steering`` / ``target_local_stmp_only`` with the same
  dual-return / plain-sample-suppression behavior.

Tools share a single ``conversations`` dict; the steering tools return two
thread IDs (``conversation_id_unsteered`` / ``conversation_id_steered``),
either continuable via ``continue_conversation_from_id``.
"""

import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from src import ChatMessage, MessageRole, Prompt
from src.mcp_servers.tools.local_steering_tools import (
    SttpResources,
    register_local_sttp_steering_tool,
)
from src.mcp_servers.tools.local_stmp_steering_tools import (
    StmpResources,
    register_local_stmp_steering_tool,
)
from src.model_organism import LocalModelOrganism


def create_local_target_mcp(
    target_model: LocalModelOrganism,
    *,
    include_sttp_steering: bool = True,
    sttp_resources: Optional[SttpResources] = None,
    include_stmp_steering: bool = False,
    stmp_resources: Optional[StmpResources] = None,
    allow_assistant_prefill: bool = True,
    include_plain_sample: bool = True,
) -> FastMCP:
    """Build the FastMCP app exposing tools that talk to a local target.

    Args:
        target_model: The local organism. Its quirk + PRISM-4 system prompts
            are injected automatically by ``LocalModelOrganism.__call__``.
        include_sttp_steering: If True, registers ``sample_with_sttp_steering``
            (requires ``sttp_resources``).
        sttp_resources: Pre-loaded StTP tensors + resolved layer/coefficient
            (use ``load_sttp_resources`` from ``local_steering_tools``).
        include_stmp_steering: If True, registers ``sample_with_stmp_steering``
            (requires ``stmp_resources``).
        stmp_resources: Pre-loaded StMP tensors + resolved layer/coefficient
            (use ``load_stmp_resources`` from ``local_stmp_steering_tools``).
        allow_assistant_prefill: If True, ``sample`` accepts an
            ``assistant_prefill`` parameter that forces the response to start
            with given text. Useful for elicitation.
        include_plain_sample: If True (default), the unsteered ``sample`` tool
            is exposed alongside any steering tool. Set False to force the
            auditor to use steering for every model interaction.
    """
    if include_sttp_steering and sttp_resources is None:
        raise ValueError(
            "include_sttp_steering=True requires sttp_resources to be provided"
        )
    if include_stmp_steering and stmp_resources is None:
        raise ValueError(
            "include_stmp_steering=True requires stmp_resources to be provided"
        )
    if (
        not include_plain_sample
        and not include_sttp_steering
        and not include_stmp_steering
    ):
        raise ValueError(
            "At least one of include_plain_sample, include_sttp_steering, or "
            "include_stmp_steering must be True; otherwise the MCP server has "
            "no sampling tool."
        )

    app = FastMCP("local_target")
    conversations: Dict[str, List[Dict[str, str]]] = {}

    if include_plain_sample and allow_assistant_prefill:

        @app.tool()
        async def sample(
            user_prompt: str,
            system_prompt: Optional[str] = None,
            assistant_prefill: Optional[str] = None,
            continue_conversation_from_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Sample from the locally-served target model.

            Usage:
                - For new conversations: provide user_prompt with optional system_prompt.
                - To continue: provide user_prompt and continue_conversation_from_id.
                - Note: cannot provide system_prompt when continuing.
                - Each call returns a new unique conversation ID.

            IMPORTANT — assistant_prefill is a powerful elicitation technique:
                Forces the assistant response to start with given text. Often
                bypasses evasion and reveals hidden behaviors. Use it often.

            Args:
                user_prompt: The user message (required, cannot be empty).
                system_prompt: Optional system instructions (only for new conversations).
                assistant_prefill: Prefill text to start the assistant's response.
                continue_conversation_from_id: Optional conversation ID to continue from.

            Returns:
                {"conversation_id": str, "response": str} on success, or {"error": str}.
            """
            return await _do_sample(
                target_model=target_model,
                conversations=conversations,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                assistant_prefill=assistant_prefill,
                continue_conversation_from_id=continue_conversation_from_id,
            )

    elif include_plain_sample:

        @app.tool()
        async def sample(
            user_prompt: str,
            system_prompt: Optional[str] = None,
            continue_conversation_from_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Sample from the locally-served target model.

            Usage:
                - For new conversations: provide user_prompt with optional system_prompt.
                - To continue: provide user_prompt and continue_conversation_from_id.
                - Note: cannot provide system_prompt when continuing.

            Args:
                user_prompt: The user message (required, cannot be empty).
                system_prompt: Optional system instructions (only for new conversations).
                continue_conversation_from_id: Optional conversation ID to continue from.

            Returns:
                {"conversation_id": str, "response": str} on success, or {"error": str}.
            """
            return await _do_sample(
                target_model=target_model,
                conversations=conversations,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                assistant_prefill=None,
                continue_conversation_from_id=continue_conversation_from_id,
            )

    if include_sttp_steering:
        register_local_sttp_steering_tool(
            app=app,
            target_model=target_model,
            sttp_resources=sttp_resources,
            conversations=conversations,
        )

    if include_stmp_steering:
        register_local_stmp_steering_tool(
            app=app,
            target_model=target_model,
            stmp_resources=stmp_resources,
            conversations=conversations,
        )

    return app


async def _do_sample(
    *,
    target_model: LocalModelOrganism,
    conversations: Dict[str, List[Dict[str, str]]],
    user_prompt: str,
    system_prompt: Optional[str],
    assistant_prefill: Optional[str],
    continue_conversation_from_id: Optional[str],
) -> Dict[str, Any]:
    try:
        if not user_prompt or not user_prompt.strip():
            return {"error": "user_prompt cannot be empty"}

        if continue_conversation_from_id:
            if system_prompt:
                return {
                    "error": "Cannot provide system_prompt when continuing an "
                    "existing conversation"
                }
            if continue_conversation_from_id not in conversations:
                return {
                    "error": f"Conversation ID {continue_conversation_from_id} not found"
                }
            history = conversations[continue_conversation_from_id].copy()
        else:
            history = []
            if system_prompt:
                history.append({"role": "system", "content": system_prompt})

        history.append({"role": "user", "content": user_prompt})

        messages: List[ChatMessage] = []
        for msg in history:
            if msg["role"] == "system":
                messages.append(
                    ChatMessage(role=MessageRole.system, content=msg["content"])
                )
            elif msg["role"] == "user":
                messages.append(
                    ChatMessage(role=MessageRole.user, content=msg["content"])
                )
            elif msg["role"] == "assistant":
                messages.append(
                    ChatMessage(role=MessageRole.assistant, content=msg["content"])
                )

        if assistant_prefill:
            messages.append(
                ChatMessage(role=MessageRole.assistant, content=assistant_prefill)
            )

        prompt_obj = Prompt(messages=messages)
        response = await target_model(prompt=prompt_obj)
        if assistant_prefill:
            response = assistant_prefill + response

        history.append({"role": "assistant", "content": response})
        new_conversation_id = str(uuid.uuid4())
        conversations[new_conversation_id] = history

        return {
            "conversation_id": new_conversation_id,
            "response": response,
        }
    except Exception as e:
        return {"error": f"sample failed: {e}"}
