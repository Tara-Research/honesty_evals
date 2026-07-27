#!/bin/bash
# Shared helpers for the env builders (build_env.sh, build_env_judge.sh).
# Sourced, never executed directly.

# Distributions pip reports as missing that must NOT be backfilled on Alliance:
#
#   pillow                  Pillow_SIMD provides the `PIL` module but registers
#                           under a different name; installing real `pillow` on
#                           top would silently downgrade it.
#   pyarrow                 Alliance publishes a dummy wheel that fails on
#                           purpose ("PyArrow is available from the Arrow
#                           module"). Only `datasets` wants it, and only the
#                           one-off `mask download` / `em fetch` stages import
#                           `datasets` — their outputs are committed to data/.
#   opencv-python-headless  Same dummy-wheel treatment. Declared by vllm, but
#                           vllm imports fine without it for the text-only
#                           models these evals serve.
#
# Retrying these just burns build time on installs that cannot succeed.
BENIGN_MISSING=(pillow pyarrow opencv-python-headless)

# ---------------------------------------------------------------------------
# backfill_missing_deps <label>
#
# The wheelhouse-first install path can leave transitive dependencies
# unsatisfied: `pip install --no-index <pkg>` resolves against the cluster
# wheelhouse only, and heavy packages installed that way (torch, vllm,
# transformers, ...) have landed without their full dependency closure. The
# result is a venv where `import honesty_evals` succeeds but `import
# transformers` dies on a missing `packaging` — a break that only surfaces once
# a job is already on a GPU node.
#
# Ask pip which requirements are unsatisfied and install exactly those, looping
# because a backfilled package can itself pull in new unmet requirements.
# ---------------------------------------------------------------------------
backfill_missing_deps() {
    local label="${1:-build_env}"
    local attempt missing

    for attempt in 1 2 3; do
        # pip check line format:
        #   "<dist> <version> requires <pkg>, which is not installed."
        missing=$(pip check 2>/dev/null \
            | sed -n 's/.* requires \([A-Za-z0-9._-]*\), which is not installed\./\1/p' \
            | sort -u \
            | grep -vxF -f <(printf '%s\n' "${BENIGN_MISSING[@]}") || true)

        if [ -z "$missing" ]; then
            [ "$attempt" = 1 ] && echo "[$label] dependency closure complete"
            return 0
        fi

        echo "[$label] backfilling unsatisfied dependencies (pass $attempt): $(echo $missing)"
        # ONE PIP INVOCATION PER PACKAGE. pip is atomic per invocation, so a
        # single unbuildable package aborts the entire batch and installs
        # nothing — which is how this venv ended up missing seven packages
        # because of one of them (Alliance ships a deliberately-failing dummy
        # `pyarrow` wheel that tells you to `module load arrow` instead).
        local pkg
        for pkg in $missing; do
            if pip install --prefer-binary "$pkg" >/dev/null 2>&1; then
                echo "[$label]   + $pkg"
            else
                echo "[$label]   ! $pkg failed to install" >&2
            fi
        done
    done

    echo "[$label] WARNING: dependencies still unsatisfied after 3 backfill passes" >&2
    return 0
}

# ---------------------------------------------------------------------------
# verify_env <label>
#
# Fail the BUILD rather than the first SLURM job. Checks the imports the evals
# actually perform plus a full CLI parser build — `honesty-evals <anything>`
# constructs every eval's subparser eagerly, so a single missing dependency of
# a single eval takes down the whole CLI (this is how a missing matplotlib made
# `honesty-evals mask generate` unusable).
# ---------------------------------------------------------------------------
verify_env() {
    local label="${1:-build_env}"
    echo "[$label] verifying runtime imports"
    python - "$label" <<'PY'
import importlib
import sys

label = sys.argv[1]

# Needed by every generate/judge/metrics stage — a failure here means the venv
# cannot run an eval at all.
REQUIRED = [
    "torch", "transformers", "peft", "huggingface_hub",
    "pandas", "numpy", "scipy", "matplotlib", "networkx",
    "yaml", "fastapi", "uvicorn", "httpx", "tqdm", "dotenv", "PIL",
    "vllm",
]

# Warn, don't fail. `datasets` is imported only by the one-off,
# internet-requiring fetch stages (`mask download`, `em fetch`) whose outputs
# are committed under data/; `cv2` is declared by vllm but only reached on
# multimodal paths, and vllm imports without it. On Alliance both are blocked
# behind dummy wheels (see BENIGN_MISSING).
OPTIONAL = {
    "datasets": "the one-off `mask download` / `em fetch` stages will not run; "
                "the committed inputs under data/ are unaffected",
    "cv2": "vllm's multimodal paths are unavailable; text-only serving, which is "
           "all these evals use, is unaffected",
}

broken = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except Exception as exc:
        broken.append(f"{name}: {type(exc).__name__}: {exc}")

for name, consequence in OPTIONAL.items():
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(
            f"[{label}] WARNING: optional import {name} unavailable "
            f"({type(exc).__name__}: {exc}) — {consequence}.",
            file=sys.stderr,
        )

try:
    from honesty_evals.cli import build_parser

    build_parser()
except Exception as exc:
    broken.append(f"honesty_evals CLI: {type(exc).__name__}: {exc}")

if broken:
    print(f"[{label}] FAILED — the built venv is not usable:", file=sys.stderr)
    for item in broken:
        print(f"    {item}", file=sys.stderr)
    print(
        f"[{label}] fix the reported packages (pip install <pkg>) and re-run "
        "this script, or delete the venv and rebuild from scratch.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[{label}] runtime imports + honesty-evals CLI OK")
PY
}
