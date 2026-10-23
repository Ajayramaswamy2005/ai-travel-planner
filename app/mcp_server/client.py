"""MCP client: lets agents call the travel tools over the Model Context Protocol.

Connects to `MCP_SERVER_URL` (streamable HTTP) when set, so many agents/processes
can share one running server; otherwise spawns `app.mcp_server.server` over stdio.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from ..config import settings
from ..models import ToolResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _server_target() -> str | StdioServerParameters:
    if settings.mcp_server_url:
        return settings.mcp_server_url
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server.server"],
        cwd=str(_PROJECT_ROOT),
    )


def _parse_result(name: str, result: Any) -> ToolResult:
    """Convert an MCP CallToolResult into the agents' ToolResult."""
    texts = [getattr(block, "text", "") for block in result.content]
    if result.is_error:
        return ToolResult(name=name, ok=False, error="".join(texts) or "tool error")

    data = result.structured_content
    # MCPServer wraps non-object return values as {"result": ...}.
    if isinstance(data, dict) and set(data) == {"result"}:
        data = data["result"]
    if data is None and texts:
        try:
            data = json.loads("".join(texts))
        except json.JSONDecodeError:
            data = "".join(texts)
    return ToolResult(name=name, ok=True, data=data)


async def _call(client: Client, name: str, kwargs: dict[str, Any]) -> ToolResult:
    try:
        return _parse_result(name, await client.call_tool(name, kwargs))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(name=name, ok=False, error=str(exc))


async def call_tools_mcp(
    calls: list[tuple[str, dict[str, Any]]], *, parallel: bool = True
) -> list[ToolResult]:
    """Run tool calls against the MCP server and return results in input order."""
    async with Client(_server_target()) as client:
        if parallel:
            return list(await asyncio.gather(*(_call(client, n, kw) for n, kw in calls)))
        return [await _call(client, n, kw) for n, kw in calls]


async def list_tools_mcp() -> list[str]:
    """Names of the tools the MCP server advertises."""
    async with Client(_server_target()) as client:
        listed = await client.list_tools()
        return [t.name for t in listed.tools]
