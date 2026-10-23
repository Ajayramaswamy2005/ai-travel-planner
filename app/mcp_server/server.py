"""Standalone MCP server exposing the travel tools over the Model Context Protocol.

Run with:
    python -m app.mcp_server.server                                # stdio
    python -m app.mcp_server.server --transport streamable-http --port 8765

Over streamable HTTP, a single server instance can be shared by any number of
MCP-capable agents (at http://<host>:<port>/mcp). The planner's own agents use it
when TOOL_TRANSPORT=mcp; see `client.py`.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

try:
    # MCP SDK v2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # pragma: no cover - fallback for older SDK
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore

from . import tools as T

logger = logging.getLogger(__name__)

mcp = _Server("travel-planner-tools")


# --- Google Maps -----------------------------------------------------------


@mcp.tool()
async def geocode(place: str) -> dict[str, Any]:
    """Resolve a place name to coordinates and a formatted address."""
    return await T.geocode(place)


@mcp.tool()
async def directions(origin: str, destination: str) -> dict[str, Any]:
    """Driving directions between two places."""
    return await T.directions(origin, destination)


@mcp.tool()
async def nearby_places(place: str, radius_m: int = 1500) -> list[dict[str, Any]]:
    """Find notable nearby places (attractions, restaurants, hotels)."""
    return await T.nearby_places(place, radius_m=radius_m)


# --- Weather ---------------------------------------------------------------


@mcp.tool()
async def current_weather(place: str) -> dict[str, Any]:
    """Current weather for a place."""
    return await T.current_weather(place)


@mcp.tool()
async def forecast(place: str, days: int = 3) -> list[dict[str, Any]]:
    """Multi-day weather forecast (max 5 days)."""
    return await T.forecast(place, days=days)


# --- SerpAPI ---------------------------------------------------------------


@mcp.tool()
async def web_search(query: str, num: int = 5) -> list[dict[str, Any]]:
    """Search the web for up-to-date travel info (events, visa, tips)."""
    return await T.web_search(query, num=num)


# --- Currency --------------------------------------------------------------


@mcp.tool()
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """Convert an amount between two currencies."""
    return await T.convert_currency(amount, from_currency, to_currency)


@mcp.tool()
async def list_currencies() -> list[str]:
    """List supported currency codes."""
    return await T.list_currencies()


def main() -> None:
    parser = argparse.ArgumentParser(description="Travel-planner MCP tools server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCP travel-planner tools server (%s)", args.transport)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()