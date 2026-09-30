"""Verifier agent for checking a generated explanation against its medical source."""

from __future__ import annotations

from typing import Any

from . import run_agent_once

# Gemini returned 404 for the repository-wide legacy default (gemini-2.0-flash)
# in September 2026. Keep this override scoped to the Verifier.
VERIFIER_DEFAULT_MODEL = "gemini-3.8-flash"


def create_verifier_agent(
    model: str = VERIFIER_DEFAULT_MODEL,
    *,
    candidate_state_key: str = "candidate_text",
) -> Any:
    """Create the current Verifier without claiming a final project schema."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Verifier agent.") from exc
    return LlmAgent(
        name="Verifier",
        model=model,
        instruction="{evaluation_prompt}",
        output_key="verdict",
    )


async def run_verifier(
    source_text: str,
    candidate_text: str,
    model: str = VERIFIER_DEFAULT_MODEL,
    *,
    question: str = "",
    evidence_context: str | None = None,
) -> str:
    """Compatibility wrapper for an ad-hoc answer-only Verifier request."""
    from evals.verifier_workbench import prompt_for
    payload = {"question_id": "ad_hoc", "question": question, "method_id": "ad_hoc", "candidate_answer": candidate_text,
               "sentences": [], "evidence_context": evidence_context if evidence_context is not None else source_text}
    return await run_medaesqa_verifier(prompt_for(payload), model)


async def run_medaesqa_verifier(evaluation_prompt: str, model: str = VERIFIER_DEFAULT_MODEL) -> str:
    """One Gemini call for one complete MedAESQA answer and its evidence items."""
    return await run_agent_once(create_verifier_agent(model), state={"evaluation_prompt": evaluation_prompt},
                                message="Return the requested JSON verification object.")


# TODO: Add contradiction, numeric, and uncertainty fields only after the team agrees on
# a human-label mapping and a final structured schema.
