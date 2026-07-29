"""Base worker agent helpers."""

from __future__ import annotations

import logging
from typing import Any

from ..llm import call_llm
from ..models import AgentOutput, PlanState
from ..mcp_server.registry import call_tools_parallel

logger = logging.getLogger(__name__)


class BaseAgent:
    name: str = "base"
    description: str = ""

    async def run(self, state: PlanState) -> AgentOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- helpers shared by workers ----------------------------------------

    async def gather_tools(self, calls: list[tuple[str, dict[str, Any]]]) -> list:
        """Run tool calls in parallel and return ToolResult list."""
        return await call_tools_parallel(calls)

    async def synthesize(self, system: str, prompt: str) -> str:
        return await call_llm(prompt, system=system)

    def _ctx(self, state: PlanState) -> str:
        req = state.request
        return (
            f"Destination: {req.destination or 'unknown'}\n"
            f"Days: {req.days}\n"
            f"Travelers: {req.travelers}\n"
            f"Budget: {req.budget if req.budget is not None else 'not specified'} "
            f"{req.currency}\n"
            f"Preferences: {', '.join(req.preferences) or 'none'}"
        )