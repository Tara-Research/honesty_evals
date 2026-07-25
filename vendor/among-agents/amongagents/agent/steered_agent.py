"""Steered LLM agent for Among Us backed by the honesty_evals steering server.

VENDORED-TREE CHANGE (honesty_evals port of persona_playground src/AmongUs).
The original SteeredLLMAgent loaded a HuggingFace model in-process
(ModelRegistry singleton) and generated via persona_playground's
src.multi_turn_steering.generate_answer with PyTorch-hook steering; cross-game
batching went through batch_inference_pool.py. In this port:

  * ALL LLM agents — impostor and crewmate, steered and unsteered — are async
    HTTP clients of the honesty_evals steering server
    (honesty_evals.core.serving.client.SteeredClient.agenerate). The OpenRouter
    path is removed (see agent.py). vLLM's continuous batching replaces the
    batch pool, so batch_inference_pool.py was deleted.
  * All persona_playground imports (src.utils.*, src.multi_turn_steering.*,
    src.constants.*) are gone. The agent no longer resolves vectors, layers,
    or coefficients itself: the eval glue (honesty_evals/evals/among_us/run.py)
    compiles the steering spec once and injects everything through
    agent_config["SERVER_CONFIG"]:

      SERVER_CONFIG = {
          "server_url":     str,            # running steering server
          "model_id":       str,            # for logging only (server owns the model)
          "sampling":       dict,           # temperature/top_p/max_tokens/n/seed
          "enable_thinking": bool | None,   # chat-template thinking flag
          "role_steering": {                # compiled steering payload per role,
              "impostor": dict | None,      # None = unsteered generation
              "crewmate": dict | None,
          },
      }

  * Generation errors are NOT swallowed any more (the original returned
    "SPEAK: ..." after catching any exception); a failed request raises and
    aborts the game so problems surface immediately.

Overrides only ``send_request()`` — prompt construction, response parsing,
action matching, logging, and memory management are inherited from LLMAgent.
"""

from typing import Dict, Optional

from honesty_evals.core.serving.client import SteeredClient

from amongagents.agent.agent import LLMAgent

_REQUIRED_SERVER_CONFIG_KEYS = (
    "server_url",
    "model_id",
    "sampling",
    "enable_thinking",
    "role_steering",
)


def _validate_server_config(server_config: Dict) -> None:
    for key in _REQUIRED_SERVER_CONFIG_KEYS:
        if key not in server_config:
            raise KeyError(f"agent_config['SERVER_CONFIG'] must contain '{key}'")
    role_steering = server_config["role_steering"]
    for role in ("impostor", "crewmate"):
        if role not in role_steering:
            raise KeyError(
                f"agent_config['SERVER_CONFIG']['role_steering'] must contain '{role}' "
                "(a compiled steering payload dict, or None for unsteered)"
            )


class SteeredLLMAgent(LLMAgent):
    """LLMAgent whose completions come from the honesty_evals steering server.

    The per-role steering payload (a CompiledSteering.to_dict(), or None) is
    attached to every /generate request; unsteered roles simply send
    steering=None through the same server.
    """

    def __init__(self, player, tools, game_index, agent_config, list_of_impostors):
        # super().__init__ sets up system prompt, logging, summarization, etc.
        super().__init__(player, tools, game_index, agent_config, list_of_impostors)

        if "SERVER_CONFIG" not in agent_config:
            raise KeyError("SteeredLLMAgent requires agent_config['SERVER_CONFIG']")
        server_config = agent_config["SERVER_CONFIG"]
        _validate_server_config(server_config)

        # Override the placeholder model string set by LLMAgent.__init__ for logging.
        self.model = server_config["model_id"]

        self._client = SteeredClient(server_config["server_url"])
        self._sampling: Dict = server_config["sampling"]
        self._enable_thinking: Optional[bool] = server_config["enable_thinking"]

        role_key = "impostor" if player.identity == "Impostor" else "crewmate"
        self._steering_payload: Optional[Dict] = server_config["role_steering"][role_key]

        print(
            f"[SteeredLLMAgent] {player.name} ({player.identity}) "
            f"steered={self._steering_payload is not None}"
        )

    async def send_request(self, messages):
        """Generate through the steering server (async HTTP, batched server-side)."""
        completions = await self._client.agenerate(
            messages,
            sampling=self._sampling,
            steering=self._steering_payload,
            enable_thinking=self._enable_thinking,
        )
        return completions[0]
