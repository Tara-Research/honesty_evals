"""FastMCP server exposing sandboxed filesystem tools.

Used by non-Claude agents (e.g. the OpenAI-compatible / vLLM agent) which do
not get Read / Write / Edit / LS / Bash from the Claude Agent SDK. Mounted
alongside the auditing MCP servers via ``launch_all_mcp_servers``.
"""

from pathlib import Path

from fastmcp import FastMCP

from src.mcp_servers.tools.filesystem_tools import register_filesystem_tools


def create_filesystem_mcp(workspace_dir: Path) -> FastMCP:
    """Create a FastMCP server with Read / Write / Edit / LS / Bash tools.

    All filesystem operations are sandboxed to ``workspace_dir``.

    Args:
        workspace_dir: Directory the tools are restricted to.

    Returns:
        FastMCP: Configured MCP server instance.
    """
    app = FastMCP("filesystem")
    register_filesystem_tools(app, workspace_dir)
    return app