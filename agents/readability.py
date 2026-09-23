"""Readability agent owned by the Readability teammate."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_MODEL, run_agent_once


def create_readability_agent(model: str = DEFAULT_MODEL) -> Any:
    """Create the starter Readability agent from the provided notebook instruction."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Readability agent.") from exc
    return LlmAgent(
        name="Readability",
        model=model,
        instruction=(
            "Polish the draft to a Flesch-Kincaid grade level of 8 or below while preserving every "
            "fact. Do not introduce new clinical content.\n\nDRAFT:\n{draft}"
        ),
        output_key="final",
    )


async def run_readability(draft: str, model: str = DEFAULT_MODEL) -> str:
    """Run only the Readability agent on one draft."""

    return await run_agent_once(
        create_readability_agent(model),
        state={"draft": draft},
        message="Check and improve the draft's readability without changing its facts.",
    )


# TODO: The Readability owner should decide whether final output is a score, revised text, or both.
