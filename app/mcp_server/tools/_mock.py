"""Helpers shared by the mock tool implementations."""

from __future__ import annotations

import asyncio
import zlib

from ...config import settings


def stable_hash(text: str) -> int:
    """Process-independent hash, so mocks match across the MCP server and agents."""
    return zlib.crc32(text.encode("utf-8"))


async def mock_delay() -> None:
    """Simulate network latency for a mock tool call."""
    await asyncio.sleep(settings.mock_tool_latency_ms / 1000)
