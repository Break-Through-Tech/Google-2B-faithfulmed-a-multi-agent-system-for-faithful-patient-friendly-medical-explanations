"""Extractor agent owned by the Extractor teammate."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_MODEL, run_agent_once


def create_extractor_agent(model: str = DEFAULT_MODEL) -> Any:
    """Create the starter Extractor agent from the provided notebook instruction."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Extractor agent.") from exc
    return LlmAgent(
        name="Extractor",
        model=model,
        instruction=(
            "Extract every clinically meaningful fact from the medical source below as a JSON "
            "list. Each item should have fields named text and type. Type should be diagnosis, "
            "medication, lab_value, procedure, or instruction. Do not add unsupported facts.\n\n"
            "MEDICAL SOURCE:\n{source_text}"
        ),
        output_key="atoms",
    )


async def run_extractor(source_text: str, model: str = DEFAULT_MODEL) -> str:
    """Run only the Extractor on one scratch medical source."""

    return await run_agent_once(
        create_extractor_agent(model),
        state={"source_text": source_text},
        message="Extract the clinical facts from the supplied medical source.",
    )


# TODO: The Extractor owner should formalize and test the final output schema.
