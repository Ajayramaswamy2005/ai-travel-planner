"""Weather agent: provides current + forecast weather insights."""

from __future__ import annotations

import json
import logging

from ..models import AgentOutput, PlanState
from .base import BaseAgent

logger = logging.getLogger(__name__)


class WeatherAgent(BaseAgent):
    name = "weather"
    description = "Provides current conditions and a forecast for the trip dates."

    async def run(self, state: PlanState) -> AgentOutput:
        req = state.request
        dest = req.destination or "the destination"

        results = await self.gather_tools(
            [
                ("current_weather", {"place": dest}),
                ("forecast", {"place": dest, "days": min(max(req.days, 1), 5)}),
                ("web_search", {"query": f"{dest} weather travel tips best time to visit"}),
            ]
        )
        current, forecast, search = results

        current_str = json.dumps(current.data or {}, default=str)[:800]
        forecast_str = json.dumps(forecast.data or [], default=str)[:1200]
        search_str = json.dumps(search.data or [], default=str)[:800]

        prompt = (
            f"{self._ctx(state)}\n\n"
            f"Current weather: {current_str}\n"
            f"Forecast: {forecast_str}\n"
            f"Travel tips: {search_str}\n\n"
            f"Summarize weather for a {req.days}-day trip to {dest}: what to pack, "
            f"best days for outdoor activities, and any cautions. Concise markdown."
        )

        summary = await self.synthesize(
            system="You are a travel weather advisor. Be practical and concise.",
            prompt=prompt,
        )

        return AgentOutput(
            agent=self.name,
            summary=summary,
            details=[r.model_dump() for r in results],
            tools_used=[r.name for r in results if r.ok],
        )