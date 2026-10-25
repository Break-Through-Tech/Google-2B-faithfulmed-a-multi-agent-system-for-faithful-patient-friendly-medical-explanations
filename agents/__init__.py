"""Small helpers shared by the individual FaithfulMed agents."""

from __future__ import annotations

import uuid
from typing import Any, Callable


DEFAULT_MODEL = "gemini-2.0-flash"


async def run_agent_once(
    agent: Any,
    *,
    state: dict[str, Any],
    message: str,
) -> str:
    """Run one ADK agent in an isolated in-memory session and return its final text."""

    try:
        from google.adk.runners import InMemoryRunner
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running a Google ADK agent.") from exc

    app_name = "faithfulmed_individual_agent"
    user_id = "local_user"
    session_id = f"agent-{uuid.uuid4().hex}"
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=state,
    )
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            parts = getattr(getattr(event, "content", None), "parts", None) or []
            final_text = "".join(
                part.text for part in parts if isinstance(getattr(part, "text", None), str)
            ).strip()
    if not final_text:
        raise RuntimeError("The ADK agent completed without a final text response.")
    return final_text


async def run_agent_stream(
    agent: Any,
    *,
    state: dict[str, Any],
    message: str,
    on_text: Callable[[str, str], None] | None = None,
) -> str:
    """Run one ADK agent with genuine SSE model streaming.

    ``on_text`` receives ``(delta, accumulated_text)`` only for text emitted by
    ADK events. The helper never manufactures a character-by-character stream.
    """
    try:
        from google.adk.agents.run_config import RunConfig, StreamingMode
        from google.adk.runners import InMemoryRunner
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running a Google ADK agent.") from exc

    app_name = "faithfulmed_system_demo"
    user_id = "local_user"
    session_id = f"demo-{uuid.uuid4().hex}"
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=state,
    )
    content = types.Content(role="user", parts=[types.Part(text=message)])
    accumulated = ""
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if event.error_code:
            raise RuntimeError(event.error_message or event.error_code)
        parts = getattr(getattr(event, "content", None), "parts", None) or []
        text = "".join(
            part.text for part in parts if isinstance(getattr(part, "text", None), str)
        )
        if event.partial and text:
            accumulated += text
            if on_text:
                on_text(text, accumulated)
        elif event.is_final_response() and text:
            final_text = text.strip()
            if not accumulated:
                accumulated = final_text
                if on_text:
                    on_text(final_text, accumulated)
    result = final_text or accumulated.strip()
    if not result:
        raise RuntimeError("The ADK agent completed without a final text response.")
    return result
