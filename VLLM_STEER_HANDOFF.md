# Handoff: vllm-steer is stale and blocks Qwen3.6-27B

Written 2026-07-27. Repo state at time of writing: `96e7912` on `main`.

## TL;DR

`Qwen/Qwen3.6-27B` cannot be served by the `external/vllm-steer` fork. The fork's
upstream vLLM snapshot is ~7 months old and predates the `qwen3_5` architecture.
Steering is not involved in the failure and the MASK/LoRA code is fine.

Three paths, in increasing order of scope:

1. **Run unsteered on stock vLLM** — ~30–40 lines, unblocks the baseline run now.
   Recommended first step. Not throwaway: it's step one of de-forking.
2. **Hop the fork onto EasySteer's newer base** — restores steering, keeps a fork.
3. **Replace the fork with a vLLM plugin (vllm-lens)** — removes the staleness
   problem permanently, but has an unresolved correctness question.

---

## 1. The failure

Job `384077`, submitted as:

```bash
bash scripts/slurm/submit.sh --cluster tamia mask.sbatch \
    MODEL=Qwen/Qwen3.6-27B \
    LORA_ADAPTER=/scratch/n/niklas/task-arithmetic/steering-adapters/qwen36-honesty-advsys-ep5-band-c4 \
    RUN_NAME=mask_qwen36_advsys_ep5_band_c4_baseline \
    STEERING=none ARCHETYPES=all
```

Failed after 42s in `[1/4] generate`. Log: `logs/he_mask_384077.out`.

```
Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported for now.
...
RuntimeError: steering server exited with code 1 before becoming healthy
```

The launched steering server died during `ModelConfig` validation, before any
weights loaded.

### Cause

`Qwen/Qwen3.6-27B` declares `architectures: ["Qwen3_5ForConditionalGeneration"]`,
`model_type: qwen3_5`. The fork's model registry has `qwen3`, `qwen3_moe`,
`qwen3_next`, `qwen3_vl`, `qwen3_vl_moe` — no `qwen3_5`.

### Explicitly NOT the cause

- **Not the LoRA.** Adapter rank is 64 (server default `max_lora_rank` is 96),
  `base_model_name_or_path` correctly reads `Qwen/Qwen3.6-27B`. It was never
  registered — that happens two lines after the engine construction that threw.
- **Not the venv fix** made earlier this session (`96e7912`). The CLI parsed,
  `ARCHETYPES=all` expanded to all six, and the manifest was written correctly.

### No quick workaround

`text_config.model_type` is `qwen3_5_text`: a hybrid architecture with
linear/conv attention and MTP layers (`linear_conv_kernel_dim`,
`linear_key_head_dim`, `mamba_ssm_dtype`, `layer_types`,
`full_attention_interval`, `mtp_num_hidden_layers`). Qwen3-Next-family, not
dense Qwen3 — it needs real vLLM model code, not a config relabel.

`Qwen/Qwen3.5-27B` is also `qwen3_5` and fails identically. `Qwen/Qwen3-32B` is
`Qwen3ForCausalLM` and serves today.

---

## 2. Verified version facts

| Thing | Value |
|---|---|
| Fork upstream snapshot | ~2025-12-29 (commit `e89b7d681e`), roughly pre-v0.15 |
| `qwen3_5` first appears in | **vLLM v0.17.0** (2026-03-06); absent in v0.16.0 |
| Latest upstream vLLM | v0.26.0 (2026-07-26) |
| Cluster wheelhouse stock vLLM | **0.25.0** (has `qwen3_5`) |
| Fork remote tip | `origin/honesty-evals` == `c68bb87faf` — no newer commits |
| Our commits on the fork | 3, small: see below |

### Our three fork commits

| Commit | Files | Notes |
|---|---|---|
| `3c6f26bca8` | `weight_utils.py`, `tokenizer.py` (7 lines) | transformers-5 compat: `DisabledTqdm` duplicate kwarg + `all_special_tokens_extended` removal. **Still needed on any base pinned `transformers < 5`.** |
| `b77d073a64` | `sttp.py`, `stmp.py` (new, 214 lines) + `__init__.py` (2 imports) | StTP/StMP algorithms |
| `c68bb87faf` | `requirements/cuda.txt` (6 lines) | disables flashinfer (pins broken apache-tvm-ffi) |

### Transformers pin — matters for path selection

Cluster venv has **transformers 5.14.1**.

- vLLM v0.17.0 requires `transformers >= 4.56.0, < 5` → conflict; needs `3c6f26bca8`.
- vLLM v0.25.0 / main require `transformers >= 5.5.3` → **fits cleanly**, no patch.

---

## 3. Why the fork rots: coupling map

The steering math is trivial (~215 lines of plain torch, no custom kernels). The
expensive part is per-request plumbing. Beyond `vllm/steer_vectors/` (22 files,
4274 lines) the fork patches **23 core vLLM files**, plus an extra
`vllm/hidden_states/` subsystem:

| File | steer/hidden refs |
|---|---|
| `v1/worker/gpu_model_runner.py` | 68 |
| `v1/worker/gpu_input_batch.py` | 28 |
| `v1/core/sched/scheduler.py` | 15 |
| `engine/arg_utils.py` | 11 |

These are among vLLM's fastest-churning internals — hence the rot.

### Per-request steering is genuinely required

Not all evals need it, but two do, so "one server per condition" does not work:

- `among_us/run.py:181` — `role_steering` maps roles to payloads: impostor
  steered, crewmates unsteered, **concurrently in one game**.
- `auditbench` — steering exposed as an interactive auditor MCP tool.
- MASK / generalized / em are uniform per run. (In a *steered* MASK run belief
  elicitation is skipped entirely — `generate.py` guards it with
  `if not spec.is_steered` — so all requests in the run share one condition.)

---

## 4. PATH 1 (recommended first): run unsteered on stock vLLM

Unblocks the baseline run without touching the fork question. The failed run was
already `STEERING=none`, so no steering is needed.

Why it's small:

- `vllm.steer_vectors` is imported in exactly **one** place, lazily, and only
  when `steering is not None` — `server.py:127`.
- `server.py:95` already has an `else:` branch for no-steering engine args.
- `launch.py:29` already accepts `no_steering: bool = False` — just not exposed
  via the CLI.
- The judge stage uses plain `from vllm import LLM` (`core/judge/judge.py:89`) —
  stock API, no fork dependency. `build_env_judge.sh` already builds stock-vLLM
  venvs, so the pattern exists.

### The one hard coupling

`server.py:197` passes `steer_vector_request=` to `EngineCoreRequest`
(`server.py:188`) — a kwarg that exists only in the fork.

That hand-built `EngineCoreRequest` exists *solely* to attach the steering
request. Without steering, use the public, stable API instead:

```python
async for out in self.engine.generate(
    prompt=TokensPrompt(prompt_token_ids=ids),
    sampling_params=sp, request_id=rid, lora_request=lora_request,
):
```

### Work items

1. `core/serving/server.py` — in the no-steering path, use public
   `engine.generate(...)` instead of hand-building `EngineCoreRequest` (~20 lines).
2. `evals/mask/generate.py:250` + `cli_common.py` — expose `--no-steering`, pass
   through to `launched_server` (~10 lines).
3. `scripts/slurm/mask.sbatch` — optional `NO_STEERING=1`.
4. A venv with stock vLLM (wheelhouse 0.25.0), selected by env var, leaving the
   existing fork venv untouched so nothing working breaks.

### Validate with

`--test` (5 rows/archetype) end-to-end through generate → judge → metrics →
process **before** committing 1000 rows / ~8 GPU-hours.

### Known unknown

`enable_chunked_prefill` and a few other `EngineArgs` fields are long-standing
but V1 has reshuffled some. A rename fails loudly at startup, not silently.

---

## 5. PATH 2: hop the fork onto EasySteer's newer base

Restores steering. Keeps a fork, but skips rebasing the steering subsystem
yourself.

`ZJU-REAL/EasySteer-vllm-v1` (the actual upstream of the steering subsystem;
EasySteer pins it as a submodule) already rebased. Its model tree is an **exact
match for vLLM v0.17.0** (251 files, zero diff either direction) — so it clears
the `qwen3_5` bar, but only just, and is 9 releases behind current.

### Portability of our commits onto it — verified

- `base.py` abstract interface: **identical**
- `factory.py`: **byte-identical**
- `sttp.py`/`stmp.py` imports (`.factory`, `.template`, torch): all resolve
- → StTP/StMP should drop in unchanged; re-add 2 lines to `__init__.py`
- All 8 `SteerVectorRequest` fields our `server.py` constructs still exist
- `enable_steer_vector` / `max_steer_vectors` still in its `arg_utils.py`

### What you'd lose, and whether it matters

Leaving abhay-sheshadri's line drops `capture_acts`/`fuzzing`/`inject_acts`
(honesty_evals only uses `direct`/`sttp`/`stmp` — no loss) and the LoRA-bias-TP
fix (**inert for our adapter**: `bias: none`, `lora_bias: False`; would matter
for an adapter with `bias != none`).

### swfc parity preserved

Upstream `direct.py` gained an fp32 overflow fix, but only inside
`if self.normalize:`. `normalize` defaults `False` (`steer_vectors/request.py`)
and our server never sets it, so our path (`h + params`) is unchanged.

### Steps

1. Add `EasySteer-vllm-v1` as a second remote on `Tara-Research/vllm-steer`.
2. Branch `honesty-evals-v2` off its `main`.
3. Cherry-pick `b77d073a64`; re-add the 2 `__init__.py` imports.
4. Re-evaluate `3c6f26bca8` (**probably still needed** — v0.17.0 pins
   `transformers < 5`) and `c68bb87faf`.
5. Push to `Tara-Research`, bump the submodule pointer.
6. Re-run `build_env.sh`; `pytest tests/test_fork_transforms.py`; then
   `HONESTY_EVALS_GPU_PARITY=1 pytest tests/test_parity_gpu.py` on a GPU node.

### Risks

- **Unproven:** v0.17.0 carries the *first* iteration of `qwen3_5.py` (since
  refactored 882 → 652 lines), untested against Qwen3.6-27B specifically. The
  model config was written by transformers 4.57.1, inside v0.17.0's supported
  range — encouraging, not conclusive. **Smoke-test serving before porting.**
- EasySteer squashed its history ("squashed all custom changes"), which can break
  `VLLM_USE_PRECOMPILED`'s merge-base derivation → may need
  `VLLM_PRECOMPILED_WHEEL_COMMIT` set explicitly (`build_env.sh` warns about this).

---

## 6. PATH 3: replace the fork with a vLLM plugin

Removes the staleness problem for good. Not yet validated.

### Candidate: [vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) (UK AISI)

| Requirement | Status |
|---|---|
| No fork | Official `vllm.general_plugins` entry point |
| Version coupling | `vllm>=0.16.0` — open lower bound, not a pin |
| Per-request steering | Yes, via `SamplingParams.extra_args` / `vllm_xargs` |
| Non-additive transforms (StTP/StMP) | `Hook(fn, layer_indices)`, arbitrary Python |
| Additive (SwFC) | `SteeringVector(activations, layer_indices, scale, position_indices)` |
| `steering_mode: all/response` | `position_indices` |
| TP + LoRA | Supported; steering runs on all TP ranks |
| Alive | Created 2026-03-12, pushed 2026-07-24, 117 stars |

Alternative: [nnsight 0.6](https://nnsight.net/features/15_vllm_support/) — also
unpinned `vllm` extra, far more mature (1000 stars), full TP/Ray/multi-node. But
it's a *tracing-API* programming model that would replace the serving layer's
design rather than slot into it. Better for broad interpretability (SAEs,
gradients); heavier than needed for three residual-stream transforms.

Upstream: [vLLM RFC #36998](https://github.com/vllm-project/vllm/issues/36998)
proposes an activation plugin, but Phase 1 is **observation only** — mutation is
explicitly deferred to Phase 2. Opened 2026-03, unmerged. Direction of travel,
not something to depend on.

### ⚠ THE BLOCKING QUESTION: prefix-cache contamination

**Resolve this before writing any migration code.**

Steering changes hidden states, which changes the KV written to cache. The fork
handles this at `vllm/v1/core/kv_cache_utils.py:467`:

```python
def _gen_steer_vector_extra_hash_keys(request: Request) -> list[str]:
    if not request.steer_vector_request:
        return []
    return [request.steer_vector_request.steer_vector_name]
```

— mixing steering identity into the prefix-cache block hash, right beside
`_gen_lora_extra_hash_keys`, for the same reason. `block_pool.py:263` also
records `steer_vector_id`.

**vllm-lens does not address prefix caching at all**, and our server sets
`enable_prefix_caching=True`. Scenarios we actually run that would contaminate:

- among-us: steered impostor + unsteered crewmates sharing system prompts, concurrently
- MASK: all belief prompts share `belief_system_1 = "You are a helpful assistant."`

Failure mode is **silently wrong numbers, no error** — the worst case for an eval.

**Candidate fix:** vLLM exposes `cache_salt` as a public per-request field, and
`server.py:198` already passes `cache_salt=None`. Setting it to the steering
payload name may achieve what the fork's patch does, via public API. *Unverified.*

**Test:** two requests sharing a long prefix, one steered, one not, with
`enable_prefix_caching=True`. Confirm outputs differ as they should; then confirm
`cache_salt` fixes it if they don't.

### Other drawbacks (not speed)

- Transform cost is negligible: a dot product + elementwise ops, O(N·H).
- **CUDA graphs:** vllm-lens forces `enforce_eager=True` so hooks fire. Our fork
  already forces it (`server.py:99`), so this is **no regression** — but also no
  win, and it rules out EasySteer's 2.3x server-level-steering speedup. (Earlier
  in the session I suggested de-forking would let us drop `enforce_eager`. That
  was wrong for the vllm-lens path.)
- **Token targeting:** the fork reads `attn_metadata` (`query_start_loc`,
  `num_computed_tokens`) in `steer_vectors/algorithms/utils.py`. With a generic
  hook you reconstruct that from `position_indices` yourself.
- Per-request hooks use **cloudpickle over HTTP** = arbitrary code execution.
  Fine on a private single-user server, but prefer registering StTP/StMP
  worker-side and passing only parameters. The docs do not explicitly document
  closures / per-request payloads for custom hooks — needs proving.

### Reference-math oracle already exists (de-risks the port)

`tests/reference_hf/{swfc,sttp,stmp}.py` hold frozen HF reference
implementations; `tests/test_fork_transforms.py` asserts the fork's algorithms
match them numerically on synthetic tensors, no GPU. Reimplementing the three
transforms as plain torch is validated by swapping one import.

**Caveat:** those tests prove the *arithmetic only* — not the plumbing (which
tokens got steered, batch mapping, cache interaction), which is where the real
risk is. Needs `test_parity_gpu.py` plus new tests for prompt-vs-response
targeting and shared-prefix contamination.

---

## 7. Before re-running the job

`results/mask/mask_qwen36_advsys_ep5_band_c4_baseline/config.json` already
exists, and `create_run_dir` refuses to overwrite a manifest. Delete the dir or
pick a new `RUN_NAME`.

Baseline/steered comparability: MASK steered runs reuse the baseline's beliefs.
Those are model-dependent, not stack-dependent, so a stock-vLLM baseline is
valid — but if steered arms later run on a different engine build, regenerate the
baseline on that same stack rather than mixing engines across the comparison.

---

## 8. Already done this session (committed)

`96e7912` — env scripts: backfill unsatisfied deps and verify the venv.

The tamia venv had been built with **19 unsatisfied requirements**: `transformers`
and `peft` could not import at all (missing `packaging`), `datasets` missing
`pyarrow`, `vllm` missing `psutil`/`pyzmq`, `matplotlib` absent entirely. Because
`cli.py:build_parser()` imports every eval eagerly, the missing matplotlib
(reached only via `em/plot.py`) took down the whole CLI.

Root cause: pip is atomic per invocation, so one unbuildable package aborts the
batch and installs nothing. Alliance ships deliberately-failing dummy wheels for
`pyarrow` and `opencv-python-headless` that say to load the module instead.

Fix: new `scripts/slurm/env/_common.sh` with `backfill_missing_deps` (one pip
invocation per package) and `verify_env` (fails the build, not the first SLURM
job); module loads one at a time including `gcc`/`arrow`/`opencv`.

Venv now: `pip check` down to 2 known-benign dummy-wheel entries. `pyarrow` and
`cv2` remain uninstallable and are treated as optional — `datasets` is only used
by `mask download` / `em fetch` (data is committed), and `vllm` imports fine
without `cv2` for text-only models. **Note: the builders load `python/3.11`
while vllm-lens requires Python ≥3.12** (tamia has `python/3.12.4`, `3.13.2`) —
a Path-3 migration needs a fresh venv, not an in-place upgrade.

Unrelated: a stray empty `apache-tvm-ffi` file is untracked in the repo root.
