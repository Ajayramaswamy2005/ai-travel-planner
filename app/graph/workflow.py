"""LangGraph orchestration: supervisor -> workers (parallel fan-out) -> compile.

The supervisor decides which workers to run. In parallel mode (default) the
graph fans out to every selected worker node in the same LangGraph superstep,
so the workers - and the tool calls inside each worker - run concurrently; the
compile node fans back in and merges their reports. In sequential mode the
same nodes run one after another, which is the baseline used by the benchmark.

Agents share memory through the graph state: worker outputs, messages and
errors are merged by reducers, and a LangGraph checkpointer persists that state
per session (`thread_id`), so later turns see earlier turns' conversation and
agent outputs.
"""

from __future__ import annotations

import logging
import operator
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from ..agents import (
    BudgetAgent,
    HotelAgent,
    ItineraryAgent,
    SupervisorAgent,
    WeatherAgent,
)
from ..agents.supervisor import parse_request
from ..mcp_server.registry import parallel_tools
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


# --- shared state ------------------------------------------------------------


def _merge_or_reset(left: Any, right: Any) -> Any:
    """Reducer: merge concurrent worker writes; `None` clears the value for a new turn."""
    if right is None:
        return type(left)() if left is not None else None
    if left is None:
        return right
    if isinstance(left, dict):
        return {**left, **right}
    return left + right


class GraphState(TypedDict, total=False):
    request: TravelRequest
    parallel: bool
    route_queue: list[str]
    # Written concurrently by worker nodes, so they need reducers.
    agent_outputs: Annotated[dict[str, AgentOutput], _merge_or_reset]
    visited: Annotated[list[str], _merge_or_reset]
    errors: Annotated[list[str], _merge_or_reset]
    # Conversation across all turns of a session (persisted by the checkpointer).
    messages: Annotated[list[dict[str, str]], operator.add]
    final_plan: str
    status: str


def _plan_view(state: GraphState) -> PlanState:
    """PlanState snapshot of the shared graph state, as the agents expect it."""
    return PlanState(
        request=state["request"],
        messages=list(state.get("messages", [])),
        agent_outputs=dict(state.get("agent_outputs") or {}),
        route_queue=list(state.get("route_queue", [])),
        visited=list(state.get("visited") or []),
        final_plan=state.get("final_plan", ""),
        status=state.get("status", "planning"),
        errors=list(state.get("errors") or []),
    )


# --- nodes -----------------------------------------------------------------


async def supervisor_node(state: GraphState) -> dict:
    queue = await supervisor.plan_route(_plan_view(state)) or list(Workers)
    return {
        "route_queue": queue,
        "status": "working",
        # Start the turn with a clean slate; conversation history is kept.
        "agent_outputs": None,
        "visited": None,
        "errors": None,
        "final_plan": "",
        "messages": [{"role": "supervisor", "content": f"Dispatching workers: {', '.join(queue)}"}],
    }


def _worker_node(name: str):
    agent = Workers[name]

    async def node(state: GraphState) -> dict:
        try:
            out = await agent.run(_plan_view(state))
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker %s failed", name)
            return {"visited": [name], "errors": [f"{name}: {exc}"]}
        return {
            "agent_outputs": {name: out},
            "visited": [name],
            "messages": [{"role": name, "content": out.summary}],
        }

    node.__name__ = f"{name}_node"
    return node


async def compile_node(state: GraphState) -> dict:
    final_plan = await supervisor.compile_final_plan(_plan_view(state))
    return {
        "final_plan": final_plan,
        "status": "done",
        "messages": [{"role": "assistant", "content": final_plan}],
    }


# --- routing -----------------------------------------------------------------


def _pending(state: GraphState) -> list[str]:
    done = set(state.get("visited") or [])
    return [n for n in state.get("route_queue", []) if n in Workers and n not in done]


def route_after_supervisor(state: GraphState) -> list[str] | str:
    pending = _pending(state)
    if not pending:
        return "compile"
    # Returning several nodes makes LangGraph run them concurrently in one superstep.
    return pending if state.get("parallel", True) else pending[0]


def route_after_worker(state: GraphState) -> str:
    if state.get("parallel", True):
        return "compile"
    pending = _pending(state)
    return pending[0] if pending else "compile"


def build_workflow(checkpointer: Any = None):
    """Construct and compile the LangGraph workflow."""
    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor_node)
    for name in Workers:
        g.add_node(name, _worker_node(name))
    g.add_node("compile", compile_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor, [*Workers, "compile"])
    for name in Workers:
        g.add_conditional_edges(name, route_after_worker, [*Workers, "compile"])
    g.add_edge("compile", END)

    return g.compile(checkpointer=checkpointer)


# Shared, process-wide agent memory: one checkpoint thread per session.
checkpointer = InMemorySaver(
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[("app.models", "TravelRequest"), ("app.models", "AgentOutput")]
    )
)
_compiled = None


def get_workflow():
    global _compiled
    if _compiled is None:
        _compiled = build_workflow(checkpointer=checkpointer)
    return _compiled


def reset_session(session_id: str) -> None:
    """Forget the checkpointed agent memory for a session."""
    checkpointer.delete_thread(session_id)


async def run_plan(
    request: TravelRequest,
    *,
    session_id: str | None = None,
    parallel: bool = True,
) -> PlanState:
    """Run the full workflow for a parsed travel request.

    With a `session_id`, the run continues that session's checkpointed state
    (shared agent memory) and the result is also stored in `memory`. Set
    `parallel=False` to run workers and their tool calls sequentially.
    """
    graph = get_workflow()
    config = {"configurable": {"thread_id": session_id or f"oneshot-{uuid.uuid4()}"}}
    inputs: GraphState = {
        "request": request,
        "parallel": parallel,
        "status": "planning",
        "messages": [{"role": "user", "content": request.raw or request.model_dump_json()}],
    }
    with parallel_tools(parallel):
        result = await graph.ainvoke(inputs, config)
    if not session_id:
        reset_session(config["configurable"]["thread_id"])

    final_plan = _plan_view(result)
    if session_id:
        memory.set_state(session_id, final_plan)
    return final_plan


async def plan_from_text(message: str, *, session_id: str | None = None) -> PlanState:
    """Convenience entrypoint: parse free text then run the workflow."""
    request = parse_request(message)
    return await run_plan(request, session_id=session_id)
