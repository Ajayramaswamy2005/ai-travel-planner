"""Shared state schemas passed through the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TravelRequest(BaseModel):
    """Initial request parsed from the user's message."""

    destination: str = ""
    days: int = 3
    budget: float | None = None
    currency: str = "USD"
    preferences: list[str] = Field(default_factory=list)
    travelers: int = 1
    raw: str = ""


class ToolResult(BaseModel):
    name: str
    ok: bool
    data: Any = None
    error: str | None = None


class AgentOutput(BaseModel):
    """Output produced by a worker agent."""

    agent: str
    summary: str = ""
    details: list[Any] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class PlanState(BaseModel):
    """The shared state object flowing through the graph (LangGraph compatible)."""

    request: TravelRequest
    messages: list[dict[str, str]] = Field(default_factory=list)
    # worker outputs keyed by agent name
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)
    # routing decisions made by supervisor
    route_queue: list[str] = Field(default_factory=list)
    visited: list[str] = Field(default_factory=list)
    final_plan: str = ""
    status: Literal["planning", "working", "done", "error"] = "planning"
    errors: list[str] = Field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def earlier_user_requests(self, limit: int = 3) -> list[str]:
        """User messages from previous turns of this session (shared memory)."""
        user = [m["content"] for m in self.messages if m.get("role") == "user"]
        return user[:-1][-limit:]