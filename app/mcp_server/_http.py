"""Shared HTTP helper for tool implementations."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


async def http_get(url: str, *, params: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
    """Async GET returning parsed JSON. Raises on non-2xx."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def http_post(
    url: str,
    *,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()


def run_sync(coro):
    """Run a coroutine to completion from a sync context safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We're inside a running loop; create a task and block - caller should
        # prefer awaiting directly. Return a fresh future for safety.
        raise RuntimeError("run_sync called inside a running event loop; await directly")
    return asyncio.run(coro)