# honesty_evals

Honesty evaluations for activation-steered LLMs: **five evals, one steering
stack**. All generation runs on a steering-enabled vLLM fork
([vllm-steer](https://github.com/abhay-sheshadri/vllm-steer), based on
EasySteer's vLLM V1); all judging runs on a local vLLM judge
(`openai/gpt-oss-120b`). Extracted from the `persona_playground` research
codebase.

## The five evals

| Eval | What it measures | Metric of record |
|---|---|---|
| `generalized` | Disclosure under a dishonesty-pressuring system prompt, 112 scenarios across 8 categories (v4) | honesty score 0-100 + DISCLOSE/SUPPRESS rate (+ coherence) |
| `mask` | MASK benchmark (Model Alignment between Statements and Knowledge): consistency between elicited beliefs and statements under pressure | P(lie) / honesty per archetype |
| `em` | Emergent Misalignment: does honesty steering reduce misalignment of a risky-finance LoRA model? | aligned + coherent scores 0-100 |
| `among-us` | Strategic deception in the AmongUs social-deduction game (steered impostor) | impostor win rate |
| `auditbench` | Alignment-auditing agents (vendored auditing-agents repo) with the honesty vector as a white-box tool | audit outcomes per quirk |

## Steering methods

Three methods, all implemented **inside the vLLM engine** as
`vllm.steer_vectors` algorithms (see `external/vllm-steer/vllm/steer_vectors/algorithms/`):

- **SwFC** (`swfc`) — Steering with Fixed Coefficient: add `coef * v` to the
  residual stream at a target layer (fork algorithm `direct`).
- **StTP** (`sttp`) — Steering to Target Projection: tokens whose projection
  `p = <h, v̂>` is on the wrong side of a gating threshold get their
  v̂-component snapped to a target `s = μ_pos + coef·σ_pos`.
- **StMP** (`stmp`) — Steering to Mirror Projection: reflect the projection
  across the threshold, interpolated by `coef` and capped.

The projection statistics are precomputed client-side
(`honesty_evals/core/steering/compile.py`) from the persona-vector artifacts;
only a small per-layer payload enters vLLM.

## Architecture

```
persona vectors (.pt, PERSONA_VECTORS_FOLDER)
        │  resolver + feature modes (core/steering/)
        ▼
SteeringSpec ──compile──▶ payload .pt + SteerVectorRequest fields
        │
        ▼
honesty-evals serve  (FastAPI + AsyncLLMEngine, external/vllm-steer)
        ▲  HTTP: messages + sampling + steering + lora
        │
eval generate stages (SteeredClient)          results/<eval>/<run_name>/
        │                                        ├─ config.json   (manifest)
        ▼                                        ├─ responses.json
eval judge stages (VllmJudge, gpt-oss-120b) ──▶  ├─ judged.json
        ▼                                        └─ metrics.json
eval metrics stages
```

## Installation

```bash
git clone --recurse-submodules git@github.com:Tara-Research/honesty_evals.git
cd honesty_evals

python -m venv .venv && source .venv/bin/activate
# steering runtime: the vllm-steer fork (precompiled wheel, no CUDA compile)
VLLM_USE_PRECOMPILED=1 pip install -e external/vllm-steer
pip install -e ".[dev]"
```

On the clusters, use the persistent-venv builder instead:

```bash
bash scripts/slurm/env/build_env.sh          # once per cluster
```

AuditBench keeps its own environment (`vendor/auditing-agents/pyproject.toml`,
uv-managed) because its dependency pins differ; see
`vendor/auditing-agents/README.md`.

## Steering vectors

Vectors are NOT in this repo. They resolve from `PERSONA_VECTORS_FOLDER`
(auto-detected: tamia `/project/aip-irina/shared/steering/persona_vectors`,
Mila `$SCRATCH/persona_vectors`; override via the env var). The config schema
(`configs/steering/honesty_logreg.yaml`, `honesty_caa.yaml`) maps
`(model, honesty_category)` to artifact paths and tuned `(layer, coefficient)`
per technique. First-class models: `meta-llama/Llama-3.3-70B-Instruct` and
`Qwen/Qwen3.6-27B` (add other models by copying their sections from
persona_playground). The current paper vectors live under category
`self_report_experimental`.

Dry-run the resolution without a GPU:

```bash
honesty-evals resolve --model Qwen/Qwen3.6-27B --steering sttp \
    --steering-mode all --tuned --extraction-method logistic_regression \
    --honesty-category self_report_experimental --features response_avg
```

## Quickstart (single machine with GPUs)

```bash
# 1. serve
honesty-evals serve --model Qwen/Qwen3.6-27B --tp 4 --port 8192

# 2. generate (second shell)
honesty-evals generalized generate \
    --model Qwen/Qwen3.6-27B --run-name demo_sttp \
    --system-prompt-variant dishonest \
    --steering sttp --steering-mode all --tuned \
    --extraction-method logistic_regression \
    --honesty-category self_report_experimental --features response_avg \
    --server-url http://127.0.0.1:8192 \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.95 --seed 42

# 3. judge + metrics (needs the GPUs after the server is stopped)
honesty-evals generalized judge   --run-dir results/generalized/demo_sttp
honesty-evals generalized metrics --run-dir results/generalized/demo_sttp
```

Every `generate` stage also supports `--launch-server --tp N` to own the
server for the duration of the stage (what the sbatch scripts use).

## Running on SLURM

```bash
bash scripts/slurm/submit.sh --cluster tamia generalized.sbatch \
    MODEL=Qwen/Qwen3.6-27B RUN_NAME=gen_sttp STEERING=sttp \
    LAYER=36 COEFFICIENT=3.0 STEERING_MODE=all \
    EXTRACTION_METHOD=logistic_regression \
    HONESTY_CATEGORY=self_report_experimental FEATURES=response_avg
```

Profiles: `scripts/slurm/profiles/{tamia,mila}.sh` (tamia default account
`aip-gidelgau`, 4×H100; Mila `long` partition, 4×A100L). Each `*.sbatch`
documents its required `VAR=value` inputs and chains
generate → judge → metrics in one job.

If the fork's vLLM cannot serve the judge model, build the fallback judge venv
(`scripts/slurm/env/build_env_judge.sh`) and export
`HONESTY_EVALS_JUDGE_VENV` — the sbatch scripts then swap venvs between the
generation and judge phases.

## CLI reference

```
honesty-evals serve      --model M --tp N --port P [--lora NAME=PATH ...] [--no-steering]
honesty-evals resolve    --model M --steering ... (prints resolved artifacts, no GPU)

honesty-evals generalized {generate,judge,metrics}
honesty-evals mask        {download,generate,judge,metrics,process}
honesty-evals em          {fetch,generate,judge,metrics,plot}
honesty-evals among-us    {run,judge,metrics}
honesty-evals auditbench  {run,judge}
```

Common `generate`/`run` flags (all required unless noted; no silent defaults):

```
--model MODEL_ID --run-name NAME
--steering {none,swfc,sttp,stmp}
--steering-mode {all,response}                    (iff steered)
--layer INT --coefficient FLOAT | --tuned         (iff steered)
--extraction-method {caa,logistic_regression}     (iff steered)
--honesty-category CAT --features MODE            (iff steered)
--vector-type STR            (default: global)
--direction {positive,negative}  (default: positive)
--server-url URL | --launch-server --tp N
--max-new-tokens N --temperature F --top-p F --seed N
```

`judge`/`metrics`/`process`/`plot` stages take `--run-dir results/<eval>/<run>`
(+ optional `--judge-model/--judge-tp/--reasoning-effort` overrides) — all
other provenance comes from the run manifest.

Eval-specific notes:

- `mask`: steered runs reuse the beliefs of a matching unsteered baseline —
  `mask metrics` requires `--baseline-run-dir` for steered runs. `mask
  download` (internet, once per machine) regenerates `data/mask/raw/`.
- `em`: `--conditions base,em_baseline,em_sttp` etc.; `--lora-adapter` is
  required for any `em_*` condition (served as a vLLM LoRA adapter). `em
  fetch` downloads the question sets (also committed under `data/`).
- `among-us`: ALL agents (steered and unsteered) generate through the local
  server — no OpenRouter, no API keys.
- `auditbench`: `run` expects an already-running steering server
  (`--server-url`), launched with the quirk LoRA registered
  (`--lora <repo>=<repo>`); the auditor side needs Anthropic/OpenAI keys in
  `vendor/auditing-agents/.env`. `judge` wraps the vendored `score_audits.py`.

## Results layout

```
results/<eval>/<run_name>/
  config.json      # manifest: model, steering spec, sampling, git sha, timestamp
  responses.json   # generation output (eval-specific schema, scenario ids inline)
  judged.json      # + judge scores
  metrics.json     # aggregates
  steering/        # compiled steering payloads (.pt)
```

All evals write under the single top-level `results/` (gitignored; override
root via `HONESTY_EVALS_RESULTS`).

## Reproducing persona_playground baselines

Parity expectations when comparing against persona_playground results:

- **swfc**: greedy completions should match token-for-token (same additive
  edit) — checked by `tests/test_parity_gpu.py`.
- **sttp/stmp**: the projection edits are activation-dependent; numerically
  identical math (checked by `tests/test_fork_transforms.py`) can still
  diverge by a token under sampling. Compare judge-score distributions, not
  strings.
- **among-us**: unsteered agents moved from OpenRouter to the local model —
  re-baseline win rates before comparing steered deltas.
- **em**: LoRA is served as a vLLM adapter instead of merge-and-unload — run
  the merged-HF vs LoRA+steer A/B check before trusting steered numbers.

## Repository layout

```
honesty_evals/           the package: cli, core/ (steering, serving, judge), evals/
external/vllm-steer/     git submodule — the steering runtime (fork + our sttp/stmp)
vendor/among-agents/     vendored AmongUs game engine (steered agent rewired to the server)
vendor/auditing-agents/  vendored AuditBench repo (own env; persona_bridge -> honesty_evals.core)
configs/                 steering YAMLs, judge.yaml, per-eval configs
data/                    committed benchmark inputs (generalized v4, mask raw, em questions)
scripts/slurm/           profiles, sbatch scripts, submit.sh, env builders
tests/                   golden tests + frozen HF reference steerers + GPU parity harness
```

## Vendored code & licenses

- `external/vllm-steer` — Apache-2.0 (vLLM fork via EasySteer); our additions:
  `vllm/steer_vectors/algorithms/{sttp,stmp}.py` on branch `honesty-evals`.
- `vendor/among-agents` — upstream AmongAgents game engine (see its LICENSE);
  local modifications limited to the agent generation path (documented in
  file headers).
- `vendor/auditing-agents` — vendored AuditBench; committed run outputs and
  the paper PDF were stripped. Uses the `safety-tooling` submodule.

## Development

```bash
python -m pytest tests/               # no-GPU golden tests
python -m pytest tests/test_fork_transforms.py   # needs vllm-steer installed
HONESTY_EVALS_GPU_PARITY=1 python -m pytest tests/test_parity_gpu.py  # GPU node
```

Adding an eval: create `honesty_evals/evals/<name>/` with `cli.py`
(`register_cli(subparsers)`), stages `generate/judge/metrics`, use
`cli_common` for the shared flag groups and `core/run_manifest.py` for run
dirs; register it in `honesty_evals/cli.py`.
