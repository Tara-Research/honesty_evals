"""Server-backed LLM agent for Among Us using the honesty_evals generation server.

VENDORED-TREE CHANGE (honesty_evals port of persona_playground src/AmongUs).
The original agent loaded a HuggingFace model in-process (ModelRegistry
singleton) and generated via persona_playground's
src.multi_turn_steering.generate_answer; cross-game batching went through
batch_inference_pool.py. In this port:

  * ALL LLM agents — impostor and crewmate — are async HTTP clients of the
    honesty_evals generation server
    (honesty_evals.core.serving.client.GenerationClient.agenerate). The
    OpenRouter path is removed (see agent.py). vLLM's continuous batching
    replaces the batch pool, so batch_inference_pool.py was deleted.
  * All persona_playground imports (src.utils.*, src.multi_turn_steering.*,
    src.constants.*) are gone. The eval glue
    (honesty_evals/evals/among_us/run.py) injects everything through
    agent_config["SERVER_CONFIG"]:

      SERVER_CONFIG = {
          "server_url":     str,            # running generation server
          "model_id":       str,            # for logging only (server owns the model)
          "sampling":       dict,           # temperature/top_p/max_tokens/n/seed
          "enable_thinking": bool | None,   # chat-template thinking flag
          "role_loras":     dict,           # player identity -> LoRA name | None
      }

  * ``role_loras`` gives per-role model differentiation: each agent looks up
    its own ``player.identity`` ("Impostor" / "Crewmate", see envs/player.py)
    and sends that LoRA name on every request, so impostors can run a
    fine-tuned adapter while crewmates stay on the base model *in the same
    game*. vLLM applies LoRA per request, so both live in one server and one
    batch. An empty dict means every agent uses the base model.

  * Generation errors are NOT swallowed any more (the original returned
    "SPEAK: ..." after catching any exception); a failed request raises and
    aborts the game so problems surface immediately.

Overrides only ``send_request()`` — prompt construction, response parsing,
action matching, logging, and memory management are inherited from LLMAgent.
"""

from typing import Dict, Optional

from honesty_evals.core.serving.client import GenerationClient

from amongagents.agent.agent import LLMAgent

_REQUIRED_SERVER_CONFIG_KEYS = (
    "server_url",
    "model_id",
    "sampling",
    "enable_thinking",
    "role_loras",
)


def _validate_server_config(server_config: Dict) -> None:
    for key in _REQUIRED_SERVER_CONFIG_KEYS:
        if key not in server_config:
            raise KeyError(f"agent_config['SERVER_CONFIG'] must contain '{key}'")


class ServerLLMAgent(LLMAgent):
    """LLMAgent whose completions come from the honesty_evals generation server."""

    def __init__(self, player, tools, game_index, agent_config, list_of_impostors):
        # super().__init__ sets up system prompt, logging, summarization, etc.
        super().__init__(player, tools, game_index, agent_config, list_of_impostors)

        if "SERVER_CONFIG" not in agent_config:
            raise KeyError("ServerLLMAgent requires agent_config['SERVER_CONFIG']")
        server_config = agent_config["SERVER_CONFIG"]
        _validate_server_config(server_config)

        # Override the placeholder model string set by LLMAgent.__init__ for logging.
        self.model = server_config["model_id"]

        self._client = GenerationClient(server_config["server_url"])
        self._sampling: Dict = server_config["sampling"]
        self._enable_thinking: Optional[bool] = server_config["enable_thinking"]

        # Per-role adapter: None means this agent runs the base model. Unknown
        # identities simply get None rather than raising, so adding a role to
        # the engine cannot break games that do not use LoRA at all.
        self._lora: Optional[str] = server_config["role_loras"].get(player.identity)

        print(
            f"[ServerLLMAgent] {player.name} ({player.identity}) "
            f"lora={self._lora or 'base'}"
        )

    async def send_request(self, messages):
        """Generate through the generation server (async HTTP, batched server-side)."""
        completions = await self._client.agenerate(
            messages,
            sampling=self._sampling,
            lora=self._lora,
            enable_thinking=self._enable_thinking,
        )
        return completions[0]
