"""LangGraph orchestration: supervisor -> workers (parallel) -> compile.

The supervisor decides which workers to run; the workers execute their tool
calls concurrently and produce specialist reports; a compile step merges them
into the final travel plan.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents import (
    BudgetAgent,
    HotelAgent,
    ItineraryAgent,
    SupervisorAgent,
    WeatherAgent,
)
from ..agents.supervisor import parse_request
from ..memory import memory
from ..models import AgentOutput, PlanState, TravelRequest

logger = logging.getLogger(__name__)

Workers = {
    "itinerary": ItineraryAgent(),
    "hotel": HotelAgent(),
    "weather": WeatherAgent(),
    "budget": BudgetAgent(),
}

supervisor = SupervisorAgent()


class GraphState(TypedDict, total=False):
    plan: PlanState


# --- nodes -----------------------------------------------------------------


async def supervisor_node(state: GraphState) -> dict:
    plan: PlanState = state["plan"]
    plan.status = "planning"
    plan.add_message("supervisor", "Parsing request and planning agent route.")
    queue = await supervisor.plan_route(plan)
    plan.route_queue = queue
    plan.status = "working"
    return {"plan": plan}


async def workers_node(state: GraphState) -> dict:
    plan: PlanState = state["plan"]
    queue = plan.route_queue or list(Workers.keys())
    plan.add_message("supervisor", f"Dispatching workers: {', '.join(queue)}")

    async def _run(name: str) -> tuple[str, AgentOutput]:
        agent = Workers[name]
        out = await agent.run(plan)
        plan.add_message(name, out.summary)
        return name, out

    # Parallel fan-out of worker agents (each also runs tools in parallel).
    results = await asyncio.gather(*(_run(name) for name in queue), return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):  # noqa: PERF203
            logger.exception("worker failed: %s", r)
            plan.errors.append(str(r))
            continue
        name, out = r
        plan.agent_outputs[name] = out

    return {"plan": plan}


async def compile_node(state: GraphState) -> dict:
    plan: PlanState = state["plan"]
    plan.add_message("supervisor", "Compiling final travel plan.")
    plan.final_plan = await supervisor.compile_final_plan(plan)
    plan.status = "done"
    plan.add_message("assistant", plan.final_plan)
    return {"plan": plan}


def build_workflow():
    """Construct and compile the LangGraph workflow."""
    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("workers", workers_node)
    g.add_node("compile", compile_node)

    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "workers")
    g.add_edge("workers", "compile")
    g.add_edge("compile", END)

    return g.compile()


# A module-level compiled graph (cheap to build; reuse across calls).
_compiled = None


def get_workflow():
    global _compiled
    if _compiled is None:
        _compiled = build_workflow()
    return _compiled


async def run_plan(request: TravelRequest, *, session_id: str | None = None) -> PlanState:
    """Run the full workflow for a parsed travel request.

    If `session_id` is provided, the resulting state is persisted to the
    shared conversation memory.
    """
    plan = PlanState(request=request)
    plan.add_message("user", request.raw or request.model_dump_json())
    graph = get_workflow()
    result = await graph.ainvoke({"plan": plan})
    final_plan: PlanState = result["plan"]
    if session_id:
        memory.set_state(session_id, final_plan)
    return final_plan


async def plan_from_text(message: str, *, session_id: str | None = None) -> PlanState:
    """Convenience entrypoint: parse free text then run the workflow."""
    request = parse_request(message)
    return await run_plan(request, session_id=session_id)