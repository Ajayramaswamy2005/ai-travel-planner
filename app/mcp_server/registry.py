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
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Iterator

from . import tools as T
from ..config import settings
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

__all__ = ["TOOL_REGISTRY", "call_tool", "call_tools_parallel", "parallel_tools"]

# Whether batched tool calls run concurrently (default) or one after another.
# A ContextVar so a single workflow run can be switched to sequential for benchmarking.
_PARALLEL: ContextVar[bool] = ContextVar("parallel_tools", default=True)


@contextmanager
def parallel_tools(enabled: bool) -> Iterator[None]:
    """Run tool batches inside this block in parallel (True) or sequentially (False)."""
    token = _PARALLEL.set(enabled)
    try:
        yield
    finally:
        _PARALLEL.reset(token)


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

    Calls go through the MCP server when TOOL_TRANSPORT=mcp, otherwise straight
    to the in-process registry. Runs sequentially inside `parallel_tools(False)`.

    Args:
        calls: list of (tool_name, kwargs) tuples.

    Returns:
        list of ToolResult aligned with the input order.
    """
    parallel = _PARALLEL.get()
    if settings.tool_transport == "mcp":
        from .client import call_tools_mcp

        return await call_tools_mcp(calls, parallel=parallel)

    if not parallel:
        return [await call_tool(name, **kwargs) for name, kwargs in calls]
    tasks = [call_tool(name, **kwargs) for name, kwargs in calls]
    return list(await asyncio.gather(*tasks))