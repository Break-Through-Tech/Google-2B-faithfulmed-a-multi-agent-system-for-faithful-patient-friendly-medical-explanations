"""Simplifier agent owned by the Simplifier teammate."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_MODEL, run_agent_once


def create_simplifier_agent(model: str = DEFAULT_MODEL) -> Any:
    """Create the starter Simplifier agent from the provided notebook instruction."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Simplifier agent.") from exc
    return LlmAgent(
        name="Simplifier",
        model=model,
        instruction=(
            "Rewrite the medical source for a patient at an 8th-grade reading level using only "
            "the extracted facts. Add nothing new.\n\nMEDICAL SOURCE:\n{source_text}\n\n"
            "EXTRACTED FACTS:\n{atoms}"
        ),
        output_key="draft",
    )


async def run_simplifier(
    source_text: str,
    atoms: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Run only the Simplifier using source text and extracted facts."""

    return await run_agent_once(
        create_simplifier_agent(model),
        state={"source_text": source_text, "atoms": atoms},
        message="Create the patient-friendly draft.",
    )


# TODO: The Simplifier owner should add the planned glossary retrieval and final output contract.
