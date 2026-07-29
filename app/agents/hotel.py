"""Hotel agent: recommends hotels within budget.

Calls nearby_places (hotels) and web_search (hotel deals) in parallel.
"""

from __future__ import annotations

import json
import logging

from ..models import AgentOutput, PlanState
from .base import BaseAgent

logger = logging.getLogger(__name__)


class HotelAgent(BaseAgent):
    name = "hotel"
    description = "Recommends hotels and lodging options within budget."

    async def run(self, state: PlanState) -> AgentOutput:
        req = state.request
        dest = req.destination or "the destination"

        results = await self.gather_tools(
            [
                ("nearby_places", {"place": f"hotels in {dest}"}),
                ("web_search", {"query": f"best hotels in {dest} mid-range budget"}),
                ("convert_currency", {
                    "amount": req.budget or 1000,
                    "from_currency": req.currency,
                    "to_currency": req.currency,
                }),
            ]
        )
        nearby, search, budget_conv = results

        nearby_str = json.dumps(nearby.data or [], default=str)[:1200]
        search_str = json.dumps(search.data or [], default=str)[:1200]

        prompt = (
            f"{self._ctx(state)}\n\n"
            f"Lodging options (JSON): {nearby_str}\n"
            f"Hotel research (JSON): {search_str}\n"
            f"Budget in {req.currency}: {budget_conv.data}\n\n"
            f"Recommend 3 hotels in {dest} across budget tiers with rough nightly "
            f"rates (in {req.currency}) and brief rationale. Concise markdown."
        )

        summary = await self.synthesize(
            system="You are a hotel recommendation expert aligned to traveler budget.",
            prompt=prompt,
        )

        return AgentOutput(
            agent=self.name,
            summary=summary,
            details=[r.model_dump() for r in results],
            tools_used=[r.name for r in results if r.ok],
        )