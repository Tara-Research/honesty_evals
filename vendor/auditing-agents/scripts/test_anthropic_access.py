#!/usr/bin/env python3
"""Smoke-test Anthropic / Claude Agent SDK access on a cluster (Mila etc.).

Runs two independent checks so a failure tells you *which* layer is broken:

  Test 1 (raw API)  -- stdlib-only POST to https://api.anthropic.com/v1/messages.
                       Proves the box has outbound internet AND the key is valid.
                       Zero third-party deps; runs even if claude_agent_sdk is
                       not importable.

  Test 2 (Agent SDK) -- mirrors src/agent_framework/claude_agent.py: spins up a
                       ClaudeSDKClient and does one query. Proves the actual
                       `--agent-type claude_agent` code path authenticates.
                       Unlike run_remote_target_auditor._strip_claude_subprocess_env,
                       this DOES keep ANTHROPIC_API_KEY in the env on purpose --
                       on a cluster there is no `claude login` / Claude Code
                       interference, so the key is exactly what we want used.

Runnable on a Mila login node or inside a SLURM job (compute nodes have
internet on Mila). No `import src` (keeps it light and dependency-free for
Test 1).

Usage (from src/auditing-agents/):
    python scripts/test_anthropic_access.py
    python scripts/test_anthropic_access.py --model claude-haiku-4-5 --skip-sdk
    python scripts/test_anthropic_access.py --prompt "Say OK." --max-tokens 32
"""

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# .env sits one level up from scripts/ (src/auditing-agents/.env). The same
# file src/__init__.py discovers; we parse it ourselves to avoid importing src.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def load_env_file(path: Path) -> None:
    """Minimal KEY=VALUE .env loader (no python-dotenv dependency).

    Only sets keys not already present in the real environment, so an
    env var exported in the SLURM script / shell still wins.
    """
    if not path.is_file():
        print(f"[env] no .env at {path} -- relying on process environment")
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    print(f"[env] loaded {path}")


def _mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    return f"{secret[:14]}...{secret[-4:]} (len={len(secret)})"


def test_raw_api(model: str, prompt: str, max_tokens: int) -> bool:
    """Test 1: plain HTTPS call. Returns True on success."""
    print("\n" + "=" * 70)
    print("TEST 1 - raw Anthropic Messages API (internet + key validity)")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[FAIL] ANTHROPIC_API_KEY not set (check .env / environment)")
        return False
    print(f"[info] model={model}  key={_mask(api_key)}")

    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=body,
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = "".join(
            blk.get("text", "")
            for blk in payload.get("content", [])
            if blk.get("type") == "text"
        )
        usage = payload.get("usage", {})
        print(f"[PASS] HTTP 200  reply={text!r}")
        print(
            f"[PASS] usage in={usage.get('input_tokens')} "
            f"out={usage.get('output_tokens')}  model={payload.get('model')}"
        )
        return True
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        print(f"[FAIL] HTTP {e.code}: {detail[:500]}")
        if e.code == 401:
            print("       -> key invalid/expired/revoked. Replace it in .env.")
        elif e.code == 403:
            print("       -> key lacks access to this model or is org-blocked.")
        elif e.code == 404:
            print(f"       -> model id '{model}' not found; try the dated id "
                  "(e.g. claude-haiku-4-5-20251001).")
        elif e.code == 429:
            print("       -> rate/credit limited, but auth+network are FINE.")
        return False
    except urllib.error.URLError as e:
        print(f"[FAIL] network error: {e.reason}")
        print("       -> no outbound internet / DNS / proxy needed from this "
              "node. On a login node this should work; if it fails inside a "
              "SLURM job the compute node is the problem.")
        return False
    except Exception as e:  # noqa: BLE001 - smoke test wants the raw reason
        print(f"[FAIL] unexpected: {type(e).__name__}: {e}")
        return False


async def test_agent_sdk(model: str, prompt: str, timeout_s: int) -> bool:
    """Test 2: real claude_agent_sdk path. Returns True on success."""
    print("\n" + "=" * 70)
    print("TEST 2 - claude_agent_sdk (the --agent-type claude_agent path)")
    print("=" * 70)

    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] cannot import claude_agent_sdk: {type(e).__name__}: {e}")
        print("       -> pip/uv install the project deps first (claude-agent-sdk).")
        return False

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[FAIL] ANTHROPIC_API_KEY not set")
        return False
    # Opposite of run_remote_target_auditor._strip_claude_subprocess_env: keep
    # the key so the bundled binary uses it (no `claude login` on the cluster).
    os.environ["ANTHROPIC_API_KEY"] = api_key
    for noisy in ("CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT"):
        if noisy in os.environ:
            print(f"[info] unsetting {noisy} (would confuse the bundled binary)")
            os.environ.pop(noisy, None)
    print(f"[info] model={model}  key={_mask(api_key)}")

    options = ClaudeAgentOptions(
        model=model,
        permission_mode="bypassPermissions",
        allowed_tools=[],
        disallowed_tools=[],
        system_prompt="You are a smoke test. Reply in one short sentence.",
    )

    async def _run() -> bool:
        client = ClaudeSDKClient(options=options)
        await client.connect()
        try:
            await client.query(prompt)
            texts: list[str] = []
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage) and isinstance(
                    message.content, list
                ):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            texts.append(block.text.strip())
                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        print(f"[FAIL] ResultMessage error: {message.result}")
                        return False
                    usage = message.usage or {}
                    print(f"[PASS] reply={' '.join(texts)!r}")
                    print(f"[PASS] usage out={usage.get('output_tokens')}")
                    return True
            print(f"[WARN] stream ended without ResultMessage; "
                  f"got text={' '.join(texts)!r}")
            return bool(texts)
        finally:
            await client.disconnect()

    try:
        return await asyncio.wait_for(_run(), timeout=timeout_s)
    except asyncio.TimeoutError:
        print(f"[FAIL] timed out after {timeout_s}s (binary hang / no network "
              "from this node).")
        return False
    except FileNotFoundError as e:
        print(f"[FAIL] bundled claude binary not found / not executable: {e}")
        print("       -> reinstall claude-agent-sdk for this platform "
              "(Linux wheel ships its own binary; no Node.js needed).")
        return False
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"[FAIL] {type(e).__name__}: {msg[:500]}")
        if "api key" in msg.lower() or "401" in msg or "auth" in msg.lower():
            print("       -> auth failure inside the SDK; key not accepted by "
                  "the bundled binary.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--prompt", default="Reply with exactly: ACCESS OK")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Raw-API only; the SDK manages its own budget.")
    parser.add_argument("--sdk-timeout", type=int, default=120)
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip Test 1 (raw API).")
    parser.add_argument("--skip-sdk", action="store_true",
                        help="Skip Test 2 (Claude Agent SDK).")
    args = parser.parse_args()

    load_env_file(ENV_PATH)

    results: dict[str, bool] = {}
    if not args.skip_api:
        results["raw_api"] = test_raw_api(args.model, args.prompt, args.max_tokens)
    if not args.skip_sdk:
        results["agent_sdk"] = asyncio.run(
            test_agent_sdk(args.model, args.prompt, args.sdk_timeout)
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not results:
        print("nothing ran (both tests skipped)")
        return 2
    for name, ok in results.items():
        print(f"  {name:10s} : {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print("=" * 70)
    print("ALL PASS" if all_ok else "SOME FAILED -- see per-test diagnostics above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
