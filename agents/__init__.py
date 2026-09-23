"""Small helpers shared by the individual FaithfulMed agents."""

from __future__ import annotations

import uuid
from typing import Any


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
