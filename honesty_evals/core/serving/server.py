"""vLLM generation server.

One AsyncLLMEngine (stock vLLM) behind a small FastAPI app. All five evals
generate through this server: chat-template rendering happens server-side
(the server owns the tokenizer), and LoRA adapters registered via --lora can
be selected per request.

Run:
    honesty-evals serve --model <hf_id> --tp <n> --port <p> [--lora name=path ...]

Endpoints:
    GET  /health    -> {"status": "ok", "model": ..., "loras": [...]}
    POST /generate  -> see GenerateRequest below
"""
from __future__ import annotations

import argparse
import os
import uuid
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# The only values vLLM's LoRAConfig accepts for max_lora_rank. Kept here so a
# bad value is rejected by argparse / our own check with a readable message
# rather than by pydantic literal validation inside engine construction.
VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


class SamplingSpec(BaseModel):
    temperature: float
    top_p: float
    max_tokens: int
    min_tokens: int = 0
    n: int = 1
    seed: Optional[int] = None


class GenerateRequest(BaseModel):
    # Exactly one of `messages` / `prompt` must be set. `messages` is rendered
    # with the model's chat template server-side (the server owns the tokenizer).
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    sampling: SamplingSpec
    # LoRA adapter name (must have been registered via --lora), or null for base model.
    lora: Optional[str] = None
    # Chat-template kwarg for models that support it (e.g. Qwen). None = model default.
    enable_thinking: Optional[bool] = None


class Completion(BaseModel):
    text: str
    finish_reason: Optional[str]


class GenerateResponse(BaseModel):
    completions: List[Completion]
    num_prompt_tokens: int


class GenerationServer:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        max_model_len: Optional[int],
        gpu_memory_utilization: float,
        loras: Dict[str, str],
        max_lora_rank: int,
        enforce_eager: bool,
    ):
        # vLLM imports live here so the CLI can print --help without a GPU env.
        from transformers import AutoTokenizer
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.lora.request import LoRARequest

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        engine_args = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            # Eager by default on purpose. In vLLM V1 this skips the whole
            # torch.compile pipeline as well as CUDA graph capture: with it off,
            # a cold start spent 227s compiling on TP0 alone and the other three
            # ranks had not finished 4 minutes later (SLURM job 385305 hit its
            # wall clock still warming up). Every eval before the steering
            # removal also ran eager, so this is the configuration with a track
            # record on tamia. Pass --no-enforce-eager to opt into CUDA graphs.
            "enforce_eager": enforce_eager,
        }
        # GDN (linear-attention) prefill kernel. vLLM's default "flashinfer" JIT-compiles the
        # sm90 kernels on first use; the compile runs inside the engine's profile step, holds a
        # file lock in ~/.cache/flashinfer, and with several jobs starting at once it never
        # finished inside the 6000 s health timeout (7 MASK jobs 466735-466740/466904 on
        # 2026-09-17; the lone job the day before spent 4411 s there). "triton" skips the JIT
        # (vLLM's own recommendation in the warning); numerics differ only at kernel level.
        gdn = os.environ.get("HE_GDN_PREFILL_BACKEND", "triton").strip().lower()
        if gdn and gdn != "auto":
            engine_args["gdn_prefill_backend"] = gdn
        if max_model_len is not None:
            engine_args["max_model_len"] = max_model_len
        if loras:
            # vLLM validates max_lora_rank against a fixed literal set; an
            # off-list value fails deep inside LoRAConfig's pydantic validation
            # with no mention of which knob is wrong (SLURM job 385294 died on
            # the old default of 96). Check it here instead.
            if max_lora_rank not in VLLM_MAX_LORA_RANKS:
                raise ValueError(
                    f"--max-lora-rank must be one of {list(VLLM_MAX_LORA_RANKS)}, "
                    f"got {max_lora_rank}. Pick the smallest value >= the "
                    "adapter's rank ('r' in its adapter_config.json)."
                )
            engine_args["enable_lora"] = True
            engine_args["max_lora_rank"] = max_lora_rank
            engine_args["max_loras"] = max(4, len(loras))

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))

        self.lora_requests = {
            name: LoRARequest(lora_name=name, lora_int_id=i + 1, lora_path=path)
            for i, (name, path) in enumerate(loras.items())
        }

    def _render_prompt(self, req: GenerateRequest) -> List[int]:
        if (req.messages is None) == (req.prompt is None):
            raise HTTPException(
                status_code=400,
                detail="Exactly one of 'messages' and 'prompt' must be provided.",
            )
        if req.prompt is not None:
            return self.tokenizer.encode(req.prompt)
        template_kwargs = {}
        if req.enable_thinking is not None:
            template_kwargs["enable_thinking"] = req.enable_thinking
        text = self.tokenizer.apply_chat_template(
            req.messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import SamplingParams

        prompt_token_ids = self._render_prompt(req)

        sampling_params = SamplingParams(
            temperature=req.sampling.temperature,
            top_p=req.sampling.top_p,
            max_tokens=req.sampling.max_tokens,
            min_tokens=req.sampling.min_tokens,
            n=req.sampling.n,
            seed=req.sampling.seed,
        )

        lora_request = None
        if req.lora is not None:
            if req.lora not in self.lora_requests:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown LoRA adapter '{req.lora}'. "
                    f"Available: {sorted(self.lora_requests)}",
                )
            lora_request = self.lora_requests[req.lora]

        finished = {}
        async for request_output in self.engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()),
            lora_request=lora_request,
        ):
            for output in request_output.outputs:
                if output.finish_reason is not None:
                    finished[output.index] = output

        if not finished:
            raise HTTPException(status_code=500, detail="No finished outputs from engine")

        completions = [
            Completion(text=finished[i].text, finish_reason=finished[i].finish_reason)
            for i in sorted(finished)
        ]
        return GenerateResponse(
            completions=completions, num_prompt_tokens=len(prompt_token_ids)
        )


def build_app(server: GenerationServer) -> FastAPI:
    app = FastAPI(title="honesty-evals generation server")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": server.model,
            "loras": sorted(server.lora_requests),
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        return await server.generate(req)

    return app


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--tp", type=int, required=True, help="tensor parallel size")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="LoRA adapter as name=hf_repo_or_path (repeatable)",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        default=True,
        help="Enable torch.compile + CUDA graph capture. Off by default: a cold "
        "start costs minutes of compilation per TP rank (job 385305 timed out "
        "warming up), for throughput these batch evals have never needed.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=128,
        choices=VLLM_MAX_LORA_RANKS,
        help="Max LoRA rank the engine reserves for. vLLM only accepts the "
        "values above; it must be >= the adapter's own rank (see the adapter's "
        "adapter_config.json 'r'). Default 128.",
    )


def run_serve(args: argparse.Namespace) -> None:
    loras = {}
    for spec in args.lora:
        if "=" not in spec:
            raise ValueError(f"--lora must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        loras[name] = path

    server = GenerationServer(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        loras=loras,
        max_lora_rank=args.max_lora_rank,
        enforce_eager=args.enforce_eager,
    )
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_serve_args(parser)
    run_serve(parser.parse_args())


if __name__ == "__main__":
    main()
