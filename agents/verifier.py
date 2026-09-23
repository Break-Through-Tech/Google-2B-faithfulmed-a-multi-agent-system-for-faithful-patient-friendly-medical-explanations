"""Verifier agent for checking a generated explanation against its medical source."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_MODEL, run_agent_once


EXPECTED_OUTPUT_FIELDS = (
    "faithful",
    "unsupported_claims",
    "omissions",
    "reading_level_ok",
)


def create_verifier_agent(
    model: str = DEFAULT_MODEL,
    *,
    candidate_state_key: str = "candidate_text",
) -> Any:
    """Create the current Verifier without claiming a final project schema."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Verifier agent.") from exc
    candidate_placeholder = "{" + candidate_state_key + "}"
    return LlmAgent(
        name="Verifier",
        model=model,
        instruction=(
            "Compare the generated explanation with the original medical source. Return one JSON "
            "object with exactly these fields: faithful (boolean), unsupported_claims (list of "
            "strings), omissions (list of strings), and reading_level_ok (boolean). Be strict: an "
            "unsupported clinical claim is a failure.\n\nORIGINAL MEDICAL SOURCE:\n{source_text}\n\n"
            f"GENERATED EXPLANATION:\n{candidate_placeholder}"
        ),
        output_key="verdict",
    )


async def run_verifier(
    source_text: str,
    candidate_text: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Run only the Verifier and return its raw structured response text."""

    return await run_agent_once(
        create_verifier_agent(model),
        state={"source_text": source_text, "candidate_text": candidate_text},
        message="Verify the generated explanation against the medical source.",
    )


# TODO: Add contradiction, numeric, and uncertainty fields only after the team agrees on
# a human-label mapping and a final structured schema.
