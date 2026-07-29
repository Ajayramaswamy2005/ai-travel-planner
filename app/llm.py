"""Groq LLM wrapper with mock fallback.

Exposes a single `call_llm` async function used by all agents. When no Groq
key is configured (or `USE_MOCK_FALLBACK` is truthy), a deterministic mock
generator returns a plausible text response so the system runs end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

_llm: Any = None


def _get_llm() -> Any:
    """Lazily build a langchain-groq ChatGroq client."""
    global _llm
    if _llm is not None:
        return _llm
    from langchain_groq import ChatGroq

    _llm = ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.3,
    )
    return _llm


async def call_llm(prompt: str, *, system: str = "") -> str:
    """Call the LLM (or mock) and return text.

    Args:
        prompt: User/task prompt.
        system: Optional system instruction.
    """
    if settings.use_mock_fallback or not settings.groq_api_key:
        logger.debug("LLM mock fallback active")
        return _mock_response(prompt, system=system)

    loop = asyncio.get_running_loop()

    def _sync() -> str:
        llm = _get_llm()
        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))

    try:
        return await loop.run_in_executor(None, _sync)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed (%s); falling back to mock", exc)
        return _mock_response(prompt, system=system)


def _mock_response(prompt: str, *, system: str = "") -> str:
    """Deterministic stand-in for an LLM completion."""
    # Tiny simulated latency so async paths are exercised.
    snippet = prompt.strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:160] + "..."
    return (
        "[MOCK LLM]\n"
        f"system: {system or 'travel-planner'}\n"
        f"prompt: {snippet}\n\n"
        "Produced a structured travel-planning response based on the provided "
        "tool data. (Provide a GROQ_API_KEY in .env to enable real generation.)"
    )