"""Budget agent: estimates total trip cost in the requested currency.

Calls convert_currency and web_search (cost-of-living) in parallel.
"""

from __future__ import annotations

import json
import logging

from ..models import AgentOutput, PlanState
from .base import BaseAgent

logger = logging.getLogger(__name__)


class BudgetAgent(BaseAgent):
    name = "budget"
    description = "Estimates trip cost (lodging, food, transport, activities) in the target currency."

    async def run(self, state: PlanState) -> AgentOutput:
        req = state.request
        dest = req.destination or "the destination"
        budget = req.budget or 1000.0

        results = await self.gather_tools(
            [
                ("convert_currency", {
                    "amount": budget,
                    "from_currency": req.currency,
                    "to_currency": "USD",
                }),
                ("web_search", {"query": f"cost of travel {dest} daily budget per day"}),
            ]
        )
        conv, search = results

        conv_str = json.dumps(conv.data or {}, default=str)[:800]
        search_str = json.dumps(search.data or [], default=str)[:1200]

        prompt = (
            f"{self._ctx(state)}\n\n"
            f"Budget conversion: {conv_str}\n"
            f"Cost research: {search_str}\n\n"
            f"Estimate a {req.days}-day budget for {req.travelers} traveler(s) to {dest} "
            f"with line items: lodging, food, transport, activities, misc. "
            f"State totals in {req.currency} and note if it fits the stated budget. Concise markdown."
        )

        summary = await self.synthesize(
            system="You are a precise travel budget estimator. Use clear line items.",
            prompt=prompt,
        )

        return AgentOutput(
            agent=self.name,
            summary=summary,
            details=[r.model_dump() for r in results],
            tools_used=[r.name for r in results if r.ok],
        )