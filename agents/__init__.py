"""Small helpers shared by the individual FaithfulMed agents."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional


DEFAULT_MODEL = "gemini-2.0-flash"

# Model IDs with one of these prefixes are not Gemini API models; ADK reaches them
# through its LiteLLM wrapper (for example "groq/openai/gpt-oss-120b").
LITELLM_PROVIDER_PREFIXES = ("groq/",)

UsageCallback = Callable[[dict[str, Any]], None]


def usage_from_metadata(usage: Any) -> Optional[dict[str, Any]]:
    """Normalize one model response's token usage.

    Billable output is computed as total minus input when the provider reports a
    total. That is correct for both Gemini (where thinking tokens are reported
    separately from candidates) and LiteLLM providers (where completion tokens
    already include reasoning tokens), so thinking is never counted twice.
    """
    if usage is None:
        return None

    def count(name: str) -> int:
        value = getattr(usage, name, None)
        return int(value) if isinstance(value, (int, float)) else 0

    prompt = count("prompt_token_count")
    tool_prompt = count("tool_use_prompt_token_count")
    candidates = count("candidates_token_count")
    thoughts = count("thoughts_token_count")
    total = count("total_token_count")
    input_tokens = prompt + tool_prompt
    if total:
        output_tokens, method = max(total - input_tokens, 0), "total_minus_input"
    else:
        output_tokens, method = candidates + thoughts, "candidates_plus_thoughts"
    if not (input_tokens or output_tokens):
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thoughts,
        "cached_input_tokens": count("cached_content_token_count"),
        "method": method,
        "raw": {"prompt_token_count": prompt, "tool_use_prompt_token_count": tool_prompt,
                "candidates_token_count": candidates, "thoughts_token_count": thoughts,
                "total_token_count": total},
    }


def _report_usage(event: Any, on_usage: Optional[UsageCallback]) -> None:
    """Report usage once per completed model response (never for partial chunks)."""
    if on_usage is None or getattr(event, "partial", False):
        return
    usage = usage_from_metadata(getattr(event, "usage_metadata", None))
    if usage:
        on_usage(usage)


def resolve_model(model: Any) -> Any:
    """Wrap LiteLLM-routed model IDs; return Gemini/Gemma IDs unchanged."""
    if isinstance(model, str) and model.startswith(LITELLM_PROVIDER_PREFIXES):
        try:
            from google.adk.models.lite_llm import LiteLlm
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt (litellm) to use non-Gemini models.") from exc
        return LiteLlm(model=model)
    return model


def _answer_text(parts: Any) -> str:
    """Join a response's answer text. Reasoning parts (``thought=True``) are skipped:
    Gemma returns them by default, and they would corrupt JSON outputs."""
    return "".join(
        part.text for part in parts or []
        if isinstance(getattr(part, "text", None), str) and not getattr(part, "thought", False)
    )


def _prepare_agent(agent: Any) -> Any:
    """Resolve model IDs on an agent tree in place. Agents keep string IDs until run."""
    if isinstance(getattr(agent, "model", None), str):
        agent.model = resolve_model(agent.model)
    for child in getattr(agent, "sub_agents", None) or []:
        _prepare_agent(child)
    return agent


async def run_agent_once(
    agent: Any,
    *,
    state: dict[str, Any],
    message: str,
    on_usage: Optional[UsageCallback] = None,
) -> str:
    """Run one ADK agent in an isolated in-memory session and return its final text.

    ``on_usage`` receives one normalized token-usage dict per model response.
    """

    try:
        from google.adk.runners import InMemoryRunner
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running a Google ADK agent.") from exc

    _prepare_agent(agent)
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
        _report_usage(event, on_usage)
        if event.is_final_response():
            parts = getattr(getattr(event, "content", None), "parts", None) or []
            final_text = _answer_text(parts).strip()
    if not final_text:
        raise RuntimeError("The ADK agent completed without a final text response.")
    return final_text


async def run_agent_stream(
    agent: Any,
    *,
    state: dict[str, Any],
    message: str,
    on_text: Callable[[str, str], None] | None = None,
    on_usage: Optional[UsageCallback] = None,
) -> str:
    """Run one ADK agent with genuine SSE model streaming.

    ``on_text`` receives ``(delta, accumulated_text)`` only for text emitted by
    ADK events. The helper never manufactures a character-by-character stream.
    ``on_usage`` receives token usage from completed (non-partial) responses only.
    """
    try:
        from google.adk.agents.run_config import RunConfig, StreamingMode
        from google.adk.runners import InMemoryRunner
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running a Google ADK agent.") from exc

    _prepare_agent(agent)
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
        _report_usage(event, on_usage)
        parts = getattr(getattr(event, "content", None), "parts", None) or []
        text = _answer_text(parts)
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
