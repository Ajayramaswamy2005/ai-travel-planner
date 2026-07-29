"""Shared conversational memory across agents and turns.

A lightweight, in-memory store keyed by session id. Can be swapped for a
persistent backend (Redis, Postgres) without changing the agent code.
"""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

from .models import PlanState


class ConversationMemory:
    def __init__(self) -> None:
        self._states: dict[str, PlanState] = {}
        self._history: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._lock = Lock()

    def get_state(self, session_id: str) -> PlanState | None:
        with self._lock:
            return self._states.get(session_id)

    def set_state(self, session_id: str, state: PlanState) -> None:
        with self._lock:
            self._states[session_id] = state

    def append_history(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._history[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._history.get(session_id, []))

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)
            self._history.pop(session_id, None)


# Process-wide singleton
memory = ConversationMemory()