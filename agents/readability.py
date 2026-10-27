"""Readability agent owned by the Readability teammate."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_MODEL, run_agent_once

READABILITY_DEFAULT_MODEL = "gemini-3.8-flash"

READABILITY_INSTRUCTION = (
    "You are a medical readability agent. "
    "Your task is to improve the readability of a medical draft while "
    "preserving every clinical fact and meaning.\n\n"
    "The requirements include: "
    "- Rewrite the draft to target a Flesch-Kincaid grade level of 8 or below, "
    "without changing clinical facts or meaning.\n"
    "- Preserve every fact, detail, and relationship in the original draft.\n"
    "- Do NOT add new clinical information, accumptions, interpretations, or "
    "conclusions.\n"
    "- Do NOT remove, change, exaggerate, or weaken the clinical information.\n"
    "- Do NOT hallucinate or infer information that is not explicitly supported by the draft.\n"
    "- Make only surface-level edits such as simplifying vocabulary, shortening sentences, and "
    "improving sentence structure.\n"
    "- Preserve all numbers, counts, percentages, measurements, units, dates, time periods, and " 
    "comparisons exactly as stated.\n"
    "- Preserve who or what each statement refers to, such as patients, groups, treatments, and outcomes.\n"
    "- Do NOT combine distinct patient groups.\n"
    "- Preserve relationships and distinctions such as offered, received, declined, completed, recommended, "
    "and considered.\n"
    "- Preserve negation, uncertainty, possibility, conditions, and the strength of every claim. Do NOT turn "
    "possible outcomes into a certain one.\n"
    "- Do NOT replace a specific clinical fact with a broader or more general statement.\n"
    "- Preserve useful Markdown bullet lists, and end every bullet item with a period, "
    "question mark, or exclamation point.\n"
    "- Before returning the revision, check that no bullet item is missing terminal punctuation.\n"
    "- If simplifying a sentence risks changing its clinical meaning, keep the original wording of the sentence.\n"
    "- If the draft is a refusal, is outside of the agent's scope, or is not medical, "
    "return exactly \"I am unable to help.\"\n"
    "- Return only the revised draft or the refusal and nothing else.\n"
    "- Do NOT explain your changes.\n"
    "\nDRAFT:\n{draft}"
)


def create_readability_agent(model: str = DEFAULT_MODEL) -> Any:
    """Create the Readability agent."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Readability agent.") from exc
    return LlmAgent(
        name="Readability",
        model=model,
        instruction=READABILITY_INSTRUCTION,
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
