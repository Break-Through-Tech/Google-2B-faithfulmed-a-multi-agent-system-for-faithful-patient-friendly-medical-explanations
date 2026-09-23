"""Refiner agent owned by the Refiner teammate."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_MODEL, run_agent_once


def create_refiner_agent(model: str = DEFAULT_MODEL) -> Any:
    """Create the starter Refiner agent from the provided notebook instruction."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Refiner agent.") from exc
    return LlmAgent(
        name="Refiner",
        model=model,
        instruction=(
            "Use the Verifier feedback to correct the draft. Remove unsupported claims and restore "
            "reported omissions while changing nothing else. If the feedback says the draft is "
            "faithful, return it unchanged.\n\nMEDICAL SOURCE:\n{source_text}\n\nDRAFT:\n{draft}"
            "\n\nVERIFIER FEEDBACK:\n{verdict}"
        ),
        output_key="draft",
    )


async def run_refiner(
    source_text: str,
    draft: str,
    verdict: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Run only the Refiner using one draft and its Verifier feedback."""

    return await run_agent_once(
        create_refiner_agent(model),
        state={"source_text": source_text, "draft": draft, "verdict": verdict},
        message="Refine the draft using the supplied feedback.",
    )


# TODO: The Refiner owner should define how unchanged drafts and revisions are represented.
