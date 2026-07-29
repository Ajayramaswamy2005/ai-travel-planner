"""SerpAPI tool: web_search for up-to-date travel info (events, visa, tips)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .._http import http_get
from ...config import settings

logger = logging.getLogger(__name__)

SERP_URL = "https://serpapi.com/search"


def _serp_enabled() -> bool:
    return bool(settings.serpapi_api_key) and not settings.use_mock_fallback


async def web_search(query: str, *, num: int = 5) -> list[dict[str, Any]]:
    """Search the web and return a list of result dicts."""
    if not _serp_enabled():
        await asyncio.sleep(0.01)
        return [
            {
                "title": f"{query} - result {i + 1}",
                "snippet": f"Mock travel info snippet for '{query}' (#{i + 1}).",
                "link": f"https://example.com/{abs(hash(query)) % 9999}/{i}",
                "mock": True,
            }
            for i in range(num)
        ]

    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "api_key": settings.serpapi_api_key,
    }
    data = await http_get(SERP_URL, params=params)
    organic = data.get("organic_results", []) or []
    return [
        {
            "title": r.get("title"),
            "snippet": r.get("snippet"),
            "link": r.get("link"),
            "mock": False,
        }
        for r in organic[:num]
    ]