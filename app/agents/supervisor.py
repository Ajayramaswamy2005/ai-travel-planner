"""Supervisor agent: parses the request and routes to worker agents.

Decides which worker agents are needed, builds the route queue, and merges
their outputs into a final travel plan.
"""

from __future__ import annotations

import json
import logging
import re

from ..llm import call_llm
from ..models import AgentOutput, PlanState, TravelRequest

logger = logging.getLogger(__name__)

WORKER_AGENTS = ["itinerary", "hotel", "weather", "budget"]


class SupervisorAgent:
    name = "supervisor"

    async def plan_route(self, state: PlanState) -> list[str]:
        """Decide which worker agents to run, in order."""
        req = state.request
        # Heuristic routing (no LLM needed): all four workers run by default.
        queue: list[str] = []
        if req.destination:
            queue.append("itinerary")
            queue.append("hotel")
            queue.append("weather")
            if req.budget is not None or True:
                queue.append("budget")
        if not queue:
            queue = list(WORKER_AGENTS)

        # Allow LLM to refine ordering when available (best-effort, ignored on error).
        try:
            resp = await call_llm(
                f"Given this travel request, return a JSON list ordering these agents "
                f"most relevant first: {WORKER_AGENTS}. Request: {req.model_dump_json()}",
                system="You output ONLY a JSON list of agent name strings, no prose.",
            )
            text = resp.strip()
            match = re.search(r"\[.*?\]", text, re.S)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list) and all(a in WORKER_AGENTS for a in parsed):
                    queue = parsed
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM route refinement skipped: %s", exc)

        return queue

    async def compile_final_plan(self, state: PlanState) -> str:
        """Merge worker agent outputs into a cohesive final plan."""
        sections: list[str] = []
        for name in WORKER_AGENTS:
            out: AgentOutput | None = state.agent_outputs.get(name)
            if out is None:
                continue
            sections.append(f"## {name.title()} Agent\n{out.summary}")

        if not sections:
            return "No agent outputs were produced."

        joined = "\n\n".join(sections)
        earlier = state.earlier_user_requests()
        history = f"Earlier requests this session: {' | '.join(earlier)}\n\n" if earlier else ""
        prompt = (
            f"{history}"
            f"Travel request:\n{state.request.model_dump_json(indent=2)}\n\n"
            f"Specialist agent reports:\n{joined}\n\n"
            f"Compose a single, cohesive travel plan for the user in markdown with clear "
            f"sections (Overview, Itinerary, Hotels, Weather, Budget). Keep it readable."
        )

        return await call_llm(
            system="You are a senior travel planner who synthesizes specialist reports.",
            prompt=prompt,
        )


def parse_request(message: str) -> TravelRequest:
    """Parse a free-text user message into a structured TravelRequest.

    Uses simple regex heuristics so it works in mock mode without an LLM.
    """
    text = message or ""
    req = TravelRequest(raw=text)

    # Days: "4 days", "5-day trip", "for 6 days"
    m = re.search(r"(\d+)\s*[- ]?\s*day", text, re.I)
    if m:
        req.days = max(1, int(m.group(1)))

    # Budget: "$1500", "budget 1500", "budget: 1500 USD", "1500 USD budget"
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]{3})?", text)
    if not m:
        m = re.search(r"budget\s*[:\-]?\s*\$?(\d+(?:\.\d+)?)\s*([a-zA-Z]{3})?", text, re.I)
    if not m:
        m = re.search(r"\$?(\d+(?:\.\d+)?)\s*([a-zA-Z]{3})\s+budget", text, re.I)
    if m:
        req.budget = float(m.group(1))
        if m.group(2):
            req.currency = m.group(2).upper()

    # Travelers: "2 travelers", "for 3 people"
    m = re.search(r"(\d+)\s*(?:travelers|people|adults|guests)", text, re.I)
    if m:
        req.travelers = max(1, int(m.group(1)))

    # Preferences: look for interest keywords
    interest_map = {
        "food": "food",
        "history": "history",
        "adventure": "adventure",
        "nature": "nature",
        "nightlife": "nightlife",
        "beach": "beach",
        "culture": "culture",
        "shopping": "shopping",
    }
    for kw, label in interest_map.items():
        if kw in text.lower() and label not in req.preferences:
            req.preferences.append(label)

    # Destination: text after "to"/"in"/"visit" up to a stopword, comma, or punctuation.
    m = re.search(
        r"\b(?:to|in|visit(?:ing)?)\s+([^,.;\n]+?)(?=\s+(?:for|on|with|budget|in)\b|[.,;]|$)",
        text,
        re.I,
    )
    if m and m.group(1).strip():
        req.destination = m.group(1).strip()
    elif text:
        # fallback: strip obvious planning keywords and use remainder
        stripped = re.sub(
            r"\b(plan|planning|plan to|trip|travel|help me|please|i want to|let'?s)\b",
            "",
            text,
            flags=re.I,
        ).strip(" .,;")
        req.destination = stripped or text.strip(".,;")

    return req