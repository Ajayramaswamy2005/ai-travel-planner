"""Tool registry: a single in-process interface to the MCP tools.

The MCP server (`app.mcp_server.server`) exposes these same tool functions
over the Model Context Protocol for external agents. For low-latency,
single-process orchestration, agents import this registry and call the
async functions directly (parallelised with asyncio.gather).

This keeps one source of truth for tool logic while still providing a
standalone MCP server for inter-process / cross-agent reuse.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from . import tools as T
from ..models import ToolResult

logger = logging.getLogger(__name__)

# Map of tool name -> async callable.
TOOL_REGISTRY: dict[str, Callable[..., Awaitable[Any]]] = {
    "geocode": T.geocode,
    "directions": T.directions,
    "nearby_places": T.nearby_places,
    "current_weather": T.current_weather,
    "forecast": T.forecast,
    "web_search": T.web_search,
    "convert_currency": T.convert_currency,
    "list_currencies": T.list_currencies,
}

__all__ = ["TOOL_REGISTRY", "call_tool", "call_tools_parallel"]


async def call_tool(name: str, **kwargs: Any) -> ToolResult:
    """Invoke a registered tool by name and return a ToolResult."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return ToolResult(name=name, ok=False, error=f"unknown tool: {name}")
    try:
        data = await fn(**kwargs)
        return ToolResult(name=name, ok=True, data=data)
    except Exception as exc:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        return ToolResult(name=name, ok=False, error=str(exc))


async def call_tools_parallel(calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
    """Run several tool calls concurrently and return results in order.

    Args:
        calls: list of (tool_name, kwargs) tuples.

    Returns:
        list of ToolResult aligned with the input order.
    """
    tasks = [call_tool(name, **kwargs) for name, kwargs in calls]
    return await asyncio.gather(*tasks)