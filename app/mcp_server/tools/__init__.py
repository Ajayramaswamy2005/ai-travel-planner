"""Tool implementations for the MCP server.

Each tool is an async function returning plain data (dict/list). When the
relevant API key is missing or `USE_MOCK_FALLBACK` is true, a deterministic
mock response is returned so the system runs without credentials.
"""

from __future__ import annotations

from .maps import geocode, directions, nearby_places
from .weather import current_weather, forecast
from .serp import web_search
from .currency import convert_currency, list_currencies

__all__ = [
    "geocode",
    "directions",
    "nearby_places",
    "current_weather",
    "forecast",
    "web_search",
    "convert_currency",
    "list_currencies",
]