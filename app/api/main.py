"""FastAPI application for the AI Multi-Agent Travel Planner."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..agents.supervisor import parse_request
from ..config import settings
from ..graph.workflow import plan_from_text, run_plan
from ..memory import memory
from ..models import PlanState, TravelRequest

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

app = FastAPI(title="AI Multi-Agent Travel Planner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- schemas ---------------------------------------------------------------


class PlanRequest(BaseModel):
    message: str | None = None
    # Optional structured fields (override parsed values)
    destination: str | None = None
    days: int | None = None
    budget: float | None = None
    currency: str | None = None
    travelers: int | None = None
    preferences: list[str] | None = None
    session_id: str | None = None


class PlanResponse(BaseModel):
    session_id: str
    request: TravelRequest
    final_plan: str
    agent_outputs: dict[str, Any]
    status: str
    errors: list[str]


# --- routes ----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "mock_fallback": settings.use_mock_fallback,
    }


@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest) -> PlanResponse:
    """Run the multi-agent workflow and return a complete travel plan."""
    session_id = req.session_id or str(uuid.uuid4())

    if req.message:
        request = parse_request(req.message)
    else:
        request = TravelRequest(raw="")

    # Apply optional structured overrides.
    if req.destination is not None:
        request.destination = req.destination
    if req.days is not None:
        request.days = req.days
    if req.budget is not None:
        request.budget = req.budget
    if req.currency is not None:
        request.currency = req.currency
    if req.travelers is not None:
        request.travelers = req.travelers
    if req.preferences is not None:
        request.preferences = req.preferences

    if not request.destination:
        request.destination = "Unknown destination"

    state: PlanState = await run_plan(request, session_id=session_id)

    return PlanResponse(
        session_id=session_id,
        request=state.request,
        final_plan=state.final_plan,
        agent_outputs={k: v.model_dump() for k, v in state.agent_outputs.items()},
        status=state.status,
        errors=state.errors,
    )


@app.post("/chat")
async def chat(body: dict[str, Any]) -> dict[str, Any]:
    """Conversational endpoint: free text in, plan text out.

    Maintains conversation memory keyed by session_id.
    """
    message = body.get("message", "")
    session_id = body.get("session_id") or str(uuid.uuid4())

    memory.append_history(session_id, "user", message)
    state = await plan_from_text(message, session_id=session_id)
    memory.append_history(session_id, "assistant", state.final_plan)

    return {
        "session_id": session_id,
        "reply": state.final_plan,
        "status": state.status,
        "agent_outputs": {k: v.summary for k, v in state.agent_outputs.items()},
    }


@app.get("/memory/{session_id}")
async def get_memory(session_id: str) -> dict[str, Any]:
    state = memory.get_state(session_id)
    return {
        "session_id": session_id,
        "history": memory.get_history(session_id),
        "request": state.request.model_dump() if state else None,
        "status": state.status if state else None,
    }


@app.post("/memory/{session_id}/reset")
async def reset_memory(session_id: str) -> dict[str, str]:
    memory.reset(session_id)
    return {"session_id": session_id, "status": "reset"}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "dev",
    )


if __name__ == "__main__":
    main()