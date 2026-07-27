# honesty_evals

Honesty evaluations for LLMs: **five evals, one serving stack**. All
generation runs through a small FastAPI server on **stock vLLM** (pinned in
`pyproject.toml`); all judging runs on a local vLLM judge
(`openai/gpt-oss-120b`) in the same environment. Extracted from the
`persona_playground` research codebase.

> **Note:** activation steering (SwFC/StTP/StMP via the `vllm-steer` fork) was
> removed from this repo on 2026-07-27 because the fork's vLLM snapshot could
> no longer serve current models (e.g. `Qwen/Qwen3.6-27B`). The last commit
> with steering is tagged **`pre-steering-removal`**; the options for
> reintroducing steering are written up in `VLLM_STEER_HANDOFF.md`.

## The five evals

| Eval | What it measures | Metric of record |
|---|---|---|
| `generalized` | Disclosure under a dishonesty-pressuring system prompt, 112 scenarios across 8 categories (v4) | honesty score 0-100 + DISCLOSE/SUPPRESS rate (+ coherence) |
| `mask` | MASK benchmark (Model Alignment between Statements and Knowledge): consistency between elicited beliefs and statements under pressure | P(lie) / honesty per archetype |
| `em` | Emergent Misalignment: misalignment of a risky-finance LoRA model vs its base | aligned + coherent scores 0-100 |
| `among-us` | Strategic deception in the AmongUs social-deduction game | impostor win rate |
| `auditbench` | Alignment-auditing agents (vendored auditing-agents repo) probing a quirked base+LoRA target | audit outcomes per quirk |

## Architecture

```
honesty-evals serve  (FastAPI + AsyncLLMEngine, stock vLLM)
        ▲  HTTP: messages + sampling + lora
        │
eval generate stages (GenerationClient)       results/<eval>/<run_name>/
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
pip install -e ".[dev]"     # pulls the pinned vllm + matching torch
```

(`--recurse-submodules` is still needed for `vendor/auditing-agents/safety-tooling`.
On Windows the vllm pin is skipped by a platform marker — the CPU-only stages
and tests still work.)

On the clusters, use the persistent-venv builder instead:

```bash
bash scripts/slurm/env/build_env.sh          # once per cluster, on a login node
```

AuditBench keeps its own environment (`vendor/auditing-agents/pyproject.toml`,
uv-managed) because its dependency pins differ; see
`vendor/auditing-agents/README.md`.

## Quickstart (single machine with GPUs)

```bash
# 1. serve
honesty-evals serve --model Qwen/Qwen3.6-27B --tp 4 --port 8192

# 2. generate (second shell)
honesty-evals generalized generate \
    --model Qwen/Qwen3.6-27B --run-name demo_baseline \
    --system-prompt-variant dishonest \
    --server-url http://127.0.0.1:8192 \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.95 --seed 42

# 3. judge + metrics (needs the GPUs after the server is stopped)
honesty-evals generalized judge   --run-dir results/generalized/demo_baseline
honesty-evals generalized metrics --run-dir results/generalized/demo_baseline
```

Every `generate` stage also supports `--launch-server --tp N` to own the
server for the duration of the stage (what the sbatch scripts use).

## Running on SLURM

```bash
bash scripts/slurm/submit.sh --cluster tamia generalized.sbatch \
    MODEL=Qwen/Qwen3.6-27B RUN_NAME=gen_qwen36_baseline
```

Profiles: `scripts/slurm/profiles/{tamia,mila}.sh` (tamia default account
`aip-gidelgau`, 4×H100; Mila `long` partition, 4×A100L). Each `*.sbatch`
documents its required `VAR=value` inputs and chains
generate → judge → metrics in one job, all in one venv.

## CLI reference

```
honesty-evals serve      --model M --tp N --port P [--lora NAME=PATH ...]

honesty-evals generalized {generate,judge,metrics}
honesty-evals mask        {download,generate,judge,metrics,process}
honesty-evals em          {fetch,generate,judge,metrics,plot}
honesty-evals among-us    {run,judge,metrics}
honesty-evals auditbench  {run,judge}
```

Common `generate`/`run` flags (all required unless noted; no silent defaults):

```
--model MODEL_ID --run-name NAME
--server-url URL | --launch-server --tp N
--max-new-tokens N --temperature F --top-p F --seed N
```

`judge`/`metrics`/`process`/`plot` stages take `--run-dir results/<eval>/<run>`
(+ optional `--judge-model/--judge-tp/--reasoning-effort` overrides) — all
other provenance comes from the run manifest.

Eval-specific notes:

- `mask`: belief elicitation runs in every generation pass, so `mask metrics`
  needs only `--run-dir`. `mask download` (internet, once per machine)
  regenerates `data/mask/raw/`. `--lora-adapter` serves a LoRA on top of
  `--model` for the whole run.
- `em`: `--conditions base,em_baseline`; `--lora-adapter` is required for the
  `em_baseline` condition (served as a vLLM LoRA adapter). `em fetch`
  downloads the question sets (also committed under `data/`).
- `among-us`: ALL agents generate through the local server — no OpenRouter,
  no API keys.
- `auditbench`: `run` expects an already-running generation server
  (`--server-url`), launched with the quirk LoRA registered
  (`--lora <repo>=<repo>`); the auditor side needs Anthropic/OpenAI keys in
  `vendor/auditing-agents/.env`. `judge` wraps the vendored `score_audits.py`.

## Results layout

```
results/<eval>/<run_name>/
  config.json      # manifest: model, sampling, git sha, timestamp
  responses.json   # generation output (eval-specific schema, scenario ids inline)
  judged.json      # + judge scores
  metrics.json     # aggregates
```

All evals write under the single top-level `results/` (gitignored; override
root via `HONESTY_EVALS_RESULTS`).

## Comparing against persona_playground / older runs

- **among-us**: unsteered agents moved from OpenRouter to the local model —
  re-baseline win rates before comparing across engines.
- **em**: LoRA is served as a vLLM adapter instead of merge-and-unload.
- **mask**: result files from old unsteered runs (`{model}.json`) still parse;
  files from steered runs (`{model}_{technique}[_coef].json`) belong to the
  pre-removal pipeline — process them from the `pre-steering-removal` tag.

## Repository layout

```
honesty_evals/           the package: cli, core/ (serving, judge), evals/
vendor/among-agents/     vendored AmongUs game engine (agents rewired to the server)
vendor/auditing-agents/  vendored AuditBench repo (own env)
configs/                 judge.yaml, per-eval configs
data/                    committed benchmark inputs (generalized v4, mask raw, em questions)
scripts/slurm/           profiles, sbatch scripts, submit.sh, env builder
tests/                   CLI smoke tests (no GPU)
```

## Vendored code & licenses

- `vendor/among-agents` — upstream AmongAgents game engine (see its LICENSE);
  local modifications limited to the agent generation path (documented in
  file headers).
- `vendor/auditing-agents` — vendored AuditBench; committed run outputs and
  the paper PDF were stripped. Uses the `safety-tooling` submodule.

## Development

```bash
python -m pytest tests/               # no-GPU smoke tests
```

Adding an eval: create `honesty_evals/evals/<name>/` with `cli.py`
(`register_cli(subparsers)`), stages `generate/judge/metrics`, use
`cli_common` for the shared flag groups and `core/run_manifest.py` for run
dirs; register it in `honesty_evals/cli.py`.
