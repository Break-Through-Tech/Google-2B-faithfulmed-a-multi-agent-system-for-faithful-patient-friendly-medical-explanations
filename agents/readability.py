"""Readability agent owned by the Readability teammate."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_MODEL, run_agent_once

READABILITY_DEFAULT_MODEL = "gemini-3.8-flash"


def create_readability_agent(model: str = DEFAULT_MODEL) -> Any:
    """Create the Readability agent."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Readability agent.") from exc
    return LlmAgent(
        name="Readability",
        model=model,
        instruction=(
            "You are a medical readability agent. "
            "Your task is to improve the readability of a medical draft while "
            "preserving every clinical fact and meaning.\n\n"
            "The requirements include: "
            "- Rewrite the draft to target a Flesch-Kincaid grade level of 8 or below.\n"
            "- Preserve every fact in the original draft.\n"
            "- Do NOT add new clinical information.\n"
            "- Do NOT remove or change the clinical information.\n"
            "- Do NOT hallucinate.\n"
            "- If the draft is a refusal, is outside of the agent's scope, or is not medical, "
            "return exactly \"I am unable to help.\"\n"
            "- Return only the revised draft or the refusal and nothing else.\n"
            "- Do NOT explain your changes.\n"
            "\nDRAFT:\n{draft}"
        ),
        output_key="final",
    )


async def run_readability(draft: str, model: str = DEFAULT_MODEL) -> str:
    """Run only the Readability agent on one draft."""

    return await run_agent_once(
        create_readability_agent(model),
        state={"draft": draft},
        message="Return the improved the draft.",
    )


# TODO: The Readability owner should decide whether final output is a score, revised text, or both.
