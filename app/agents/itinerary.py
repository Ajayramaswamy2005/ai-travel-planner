"""Itinerary agent: builds a day-by-day plan.

Calls nearby_places, directions, and web_search in parallel, then synthesizes
a day-by-day itinerary with the LLM.
"""

from __future__ import annotations

import json
import logging

from ..models import AgentOutput, PlanState
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ItineraryAgent(BaseAgent):
    name = "itinerary"
    description = "Builds a day-by-day activity itinerary for the destination."

    async def run(self, state: PlanState) -> AgentOutput:
        req = state.request
        dest = req.destination or "the destination"

        # Parallel tool calls (latency reduction via asyncio.gather).
        results = await self.gather_tools(
            [
                ("nearby_places", {"place": dest}),
                ("web_search", {"query": f"top things to do in {dest} {req.days} days"}),
                ("directions", {"origin": dest, "destination": dest}),
            ]
        )
        nearby, search, _dirs = results

        nearby_str = json.dumps(nearby.data or [], default=str)[:1500]
        search_str = json.dumps(search.data or [], default=str)[:1500]

        prompt = (
            f"{self._ctx(state)}\n\n"
            f"Nearby points of interest (JSON): {nearby_str}\n"
            f"Web research (JSON): {search_str}\n\n"
            f"Create a {req.days}-day itinerary for {dest} with 2-3 activities per day, "
            f"including suggested timing. Return concise markdown."
        )

        summary = await self.synthesize(
            system="You are an expert travel itinerary planner. Be concrete and concise.",
            prompt=prompt,
        )

        return AgentOutput(
            agent=self.name,
            summary=summary,
            details=[r.model_dump() for r in results],
            tools_used=[r.name for r in results if r.ok],
        )