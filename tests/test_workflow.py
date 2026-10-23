import asyncio
import time

from app.config import settings
from app.graph.workflow import plan_from_text, reset_session, run_plan
from app.mcp_server.client import call_tools_mcp
from app.mcp_server.registry import call_tools_parallel
from app.models import TravelRequest

REQUEST = TravelRequest(destination="Paris", days=3, budget=1500)


def test_all_workers_run_and_compile_once():
    state = asyncio.run(run_plan(REQUEST))
    assert set(state.agent_outputs) == {"itinerary", "hotel", "weather", "budget"}
    assert state.status == "done" and not state.errors
    assert sum(m["role"] == "assistant" for m in state.messages) == 1


def test_parallel_is_faster_than_sequential(monkeypatch):
    monkeypatch.setattr(settings, "mock_tool_latency_ms", 100)
    monkeypatch.setattr(settings, "mock_llm_latency_ms", 100)

    def timed(parallel: bool) -> float:
        start = time.perf_counter()
        asyncio.run(run_plan(REQUEST, parallel=parallel))
        return time.perf_counter() - start

    # Sequential critical path: 6 LLM + 11 tool calls; parallel: 3 LLM + 1 tool call.
    assert timed(parallel=True) < 0.6 * timed(parallel=False)


def test_session_memory_is_shared_across_turns_and_isolated():
    try:
        asyncio.run(plan_from_text("Plan a 3-day trip to Tokyo", session_id="t1"))
        second = asyncio.run(plan_from_text("Plan a 2-day trip to Osaka", session_id="t1"))
        other = asyncio.run(plan_from_text("Plan a 2-day trip to Kyoto", session_id="t2"))
        assert second.earlier_user_requests() == ["Plan a 3-day trip to Tokyo"]
        assert set(second.agent_outputs) == {"itinerary", "hotel", "weather", "budget"}
        assert other.earlier_user_requests() == []
    finally:
        reset_session("t1")
        reset_session("t2")


def test_mcp_server_returns_same_data_as_registry():
    calls = [
        ("nearby_places", {"place": "Paris"}),
        ("forecast", {"place": "Paris", "days": 2}),
        ("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}),
        ("list_currencies", {}),
    ]
    via_mcp = asyncio.run(call_tools_mcp(calls))
    local = asyncio.run(call_tools_parallel(calls))
    assert [r.data for r in via_mcp] == [r.data for r in local]
    assert all(r.ok for r in via_mcp)
