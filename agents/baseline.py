"""Single Gemini Baseline"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DEFAULT_MODEL, run_agent_once

BASELINE_DEFAULT_MODEL = "gemini-3.8-flash"


def load_prompt() -> str:
    """Load the Baseline prompt from the prompts folder."""

    prompt_path = (
        Path(__file__).resolve().parent.parent / "prompts" / "baseline_v1.txt"
    )

    return prompt_path.read_text(encoding="utf-8")

def create_baseline_agent(model: str = DEFAULT_MODEL) -> Any:
    """Create the Baseline agent."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Baseline agent.") from exc
    return LlmAgent(
        name="Baseline",
        model=model,
        instruction=load_prompt(),
        output_key="final",
    )


async def run_baseline(source_text: str, model: str = DEFAULT_MODEL) -> str:
    """Run only the Baseline agent on one draft."""

    return await run_agent_once(
        create_baseline_agent(model),
        state={"source_text": source_text},
        message="Return the patient-friendly explanation",
    )
