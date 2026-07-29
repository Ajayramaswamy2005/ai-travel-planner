"""Gradio chat UI that talks to the in-process workflow (no separate server needed)."""

from __future__ import annotations

import asyncio
import logging
import uuid

import gradio as gr

from .agents.supervisor import parse_request
from .graph.workflow import plan_from_text
from .memory import memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_reply(state) -> str:
    parts = [state.final_plan]
    parts.append("\n---\n**Agent summaries:**")
    for name, out in state.agent_outputs.items():
        parts.append(f"- **{name}**: tools used = {', '.join(out.tools_used) or 'none'}")
    if state.errors:
        parts.append(f"\n**Errors:** {state.errors}")
    return "\n".join(parts)


def chat(message: str, history: list, session_id: str):
    """Gradio chat handler.

    history is the gr.ChatInterface history; session_id is a state component.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    memory.append_history(session_id, "user", message)
    state = asyncio.run(plan_from_text(message, session_id=session_id))
    memory.append_history(session_id, "assistant", state.final_plan)

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": _format_reply(state)})
    return history, session_id


def main() -> None:
    demo = gr.ChatInterface(
        fn=chat,
        title="AI Multi-Agent Travel Planner",
        description=(
            "Ask for a travel plan, e.g. \"Plan a 4-day trip to Paris for 2 travelers on a $1500 budget\". "
            "Specialist agents (itinerary, hotel, weather, budget) collaborate under a supervisor."
        ),
        additional_inputs=[gr.State("")],
        type="messages",
    )
    demo.launch()


if __name__ == "__main__":
    main()