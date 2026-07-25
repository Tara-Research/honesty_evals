"""Sandboxed filesystem tools exposed as a FastMCP server.

These mirror the Read / Write / Edit / LS / Bash tools that the Claude Agent
SDK provides natively, so a non-Claude agent (e.g. an OpenAI-compatible
agent backed by vLLM) can drive the auditing pipeline without prompt changes.

All paths are resolved relative to ``workspace_dir`` and must remain inside
it. Path-escape attempts and disallowed Bash subcommands return a tool result
with ``error`` set rather than raising — the model needs to see the failure.
"""

import asyncio
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp import FastMCP

_MAX_READ_BYTES = 100 * 1024  # 100KB hard cap on Read output
_MAX_BASH_OUTPUT_BYTES = 8 * 1024  # 8KB cap on captured Bash stdout/stderr


def _resolve_inside_workspace(workspace_dir: Path, path_str: str) -> Path:
    """Resolve ``path_str`` relative to the workspace and ensure it stays inside.

    Raises ``ValueError`` if the resolved path escapes the workspace.
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = workspace_dir / p
    resolved = p.resolve()
    workspace_resolved = workspace_dir.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError as exc:
        raise ValueError(
            f"path outside workspace: {path_str!r} (resolved to {resolved})"
        ) from exc
    return resolved


def _format_with_line_numbers(text: str) -> str:
    lines = text.splitlines()
    width = max(len(str(len(lines))), 4)
    return "\n".join(f"{i + 1:>{width}}\t{line}" for i, line in enumerate(lines))


def register_filesystem_tools(app: FastMCP, workspace_dir: Path) -> None:
    """Register Read / Write / Edit / LS / Bash on the FastMCP app.

    Args:
        app: FastMCP application instance.
        workspace_dir: Directory that all filesystem operations are sandboxed to.
    """

    workspace_dir = Path(workspace_dir).resolve()

    @app.tool(name="Read")
    async def read_file(
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read a UTF-8 text file from the workspace.

        Returns the file contents prefixed with line numbers (``cat -n`` style).
        Files larger than 100KB are truncated with a warning suffix.

        Args:
            path: Path to the file, relative to the workspace or absolute.
            offset: 1-indexed line to start reading from (optional).
            limit: Max number of lines to return (optional).
        """
        try:
            target = _resolve_inside_workspace(workspace_dir, path)
            if not target.exists():
                return {"error": f"file not found: {path}"}
            if not target.is_file():
                return {"error": f"not a regular file: {path}"}

            data = target.read_bytes()
            truncated = False
            if len(data) > _MAX_READ_BYTES:
                data = data[:_MAX_READ_BYTES]
                truncated = True

            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()

            if offset is not None or limit is not None:
                start = (offset - 1) if offset and offset > 0 else 0
                end = (start + limit) if limit and limit > 0 else len(lines)
                selected = lines[start:end]
                width = max(len(str(end)), 4)
                numbered = "\n".join(
                    f"{start + i + 1:>{width}}\t{line}"
                    for i, line in enumerate(selected)
                )
            else:
                numbered = _format_with_line_numbers(text)

            if truncated:
                numbered += (
                    f"\n\n[truncated: file exceeds {_MAX_READ_BYTES} bytes; "
                    f"use offset/limit to read more]"
                )

            return {"content": numbered, "lines": len(lines), "truncated": truncated}
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"read failed: {e}"}

    @app.tool(name="Write")
    async def write_file(path: str, content: str) -> Dict[str, Any]:
        """Create or overwrite a UTF-8 text file in the workspace.

        Parent directories are created as needed.

        Args:
            path: Path to the file, relative to the workspace or absolute.
            content: Full new file contents.
        """
        try:
            target = _resolve_inside_workspace(workspace_dir, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            target.write_bytes(data)
            return {
                "path": str(target.relative_to(workspace_dir)),
                "bytes_written": len(data),
            }
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"write failed: {e}"}

    @app.tool(name="Edit")
    async def edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> Dict[str, Any]:
        """Replace ``old_string`` with ``new_string`` in an existing file.

        ``old_string`` must match exactly one occurrence unless ``replace_all`` is true.

        Args:
            path: Path to the file.
            old_string: Exact text to find.
            new_string: Replacement text.
            replace_all: Replace every occurrence instead of requiring uniqueness.
        """
        try:
            target = _resolve_inside_workspace(workspace_dir, path)
            if not target.exists():
                return {"error": f"file not found: {path}"}
            if not target.is_file():
                return {"error": f"not a regular file: {path}"}

            text = target.read_text(encoding="utf-8")
            count = text.count(old_string)
            if count == 0:
                return {"error": "old_string not found in file"}
            if count > 1 and not replace_all:
                return {
                    "error": (
                        f"old_string found {count} times; pass replace_all=true "
                        "or include more context to make it unique"
                    )
                }

            if replace_all:
                new_text = text.replace(old_string, new_string)
            else:
                new_text = text.replace(old_string, new_string, 1)

            target.write_text(new_text, encoding="utf-8")
            return {
                "path": str(target.relative_to(workspace_dir)),
                "replacements": count if replace_all else 1,
            }
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"edit failed: {e}"}

    @app.tool(name="LS")
    async def list_dir(path: Optional[str] = None) -> Dict[str, Any]:
        """List entries of a directory inside the workspace.

        Args:
            path: Directory path. Defaults to the workspace root.
        """
        try:
            target = _resolve_inside_workspace(workspace_dir, path or ".")
            if not target.exists():
                return {"error": f"path not found: {path}"}
            if not target.is_dir():
                return {"error": f"not a directory: {path}"}

            entries = []
            for child in sorted(target.iterdir()):
                if child.is_dir():
                    entries.append(f"{child.name}/")
                else:
                    entries.append(child.name)
            return {
                "path": str(target.relative_to(workspace_dir)) or ".",
                "entries": entries,
            }
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"list failed: {e}"}

    @app.tool(name="Bash")
    async def run_bash(command: str) -> Dict[str, Any]:
        """Run a strictly whitelisted shell command in the workspace.

        Only ``rm <path>`` and ``ls [<path>]`` are permitted. No flags,
        no pipes, no redirection. Anything else returns an error.

        Args:
            command: The shell command to run.
        """
        try:
            try:
                tokens = shlex.split(command)
            except ValueError as e:
                return {"error": f"failed to parse command: {e}"}

            if not tokens:
                return {"error": "empty command"}

            program = tokens[0]
            args = tokens[1:]

            if program == "rm":
                if len(args) != 1:
                    return {
                        "error": "rm accepts exactly one path argument; flags are not allowed"
                    }
                if args[0].startswith("-"):
                    return {"error": "rm flags are not allowed"}
                try:
                    target = _resolve_inside_workspace(workspace_dir, args[0])
                except ValueError as e:
                    return {"error": str(e)}
                if not target.exists():
                    return {"error": f"file not found: {args[0]}"}
                if target.is_dir():
                    return {"error": "rm on directories is not allowed"}
            elif program == "ls":
                if len(args) > 1:
                    return {"error": "ls accepts at most one path argument"}
                if args and args[0].startswith("-"):
                    return {"error": "ls flags are not allowed"}
                if args:
                    try:
                        _resolve_inside_workspace(workspace_dir, args[0])
                    except ValueError as e:
                        return {"error": str(e)}
            else:
                return {
                    "error": (
                        f"command not allowed: {program!r}. Only 'rm' and 'ls' "
                        "are permitted."
                    )
                }

            proc = await asyncio.create_subprocess_exec(
                program,
                *args,
                cwd=str(workspace_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
            stdout = stdout_b[:_MAX_BASH_OUTPUT_BYTES].decode("utf-8", errors="replace")
            stderr = stderr_b[:_MAX_BASH_OUTPUT_BYTES].decode("utf-8", errors="replace")
            if len(stdout_b) > _MAX_BASH_OUTPUT_BYTES:
                stdout += "\n[stdout truncated]"
            if len(stderr_b) > _MAX_BASH_OUTPUT_BYTES:
                stderr += "\n[stderr truncated]"

            result: Dict[str, Any] = {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
            if proc.returncode != 0:
                result["error"] = stderr or f"command exited with code {proc.returncode}"
            return result
        except Exception as e:
            return {"error": f"bash failed: {e}"}
