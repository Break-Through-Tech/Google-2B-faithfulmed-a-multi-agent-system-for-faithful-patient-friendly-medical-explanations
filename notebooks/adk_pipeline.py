"""FaithfulMed's current Google ADK integration point.

The individual agents live in ``agents/``. This script imports and connects them as a
sequential workflow with one bounded Verifier/Refiner loop. It is still an experiment,
not a production pipeline, and several teammate-owned agents contain TODOs.

State flow between the stages (ADK fills each ``{placeholder}`` from shared session state):

    source_text --Extractor--> atoms
                --Simplifier--> draft
                --[ Verifier --> verdict --> Refiner --> draft ] (bounded, <=2 passes)
                --Readability--> final

Setup:
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...        # key from Google AI Studio

Building the workflow does not call Gemini. Running it through an ADK runner does.
"""

from __future__ import annotations

from typing import Any

from agents import run_agent_once
from agents.extractor import create_extractor_agent
from agents.readability import create_readability_agent
from agents.refiner import create_refiner_agent
from agents.simplifier import create_simplifier_agent

# The repo's original default (gemini-2.0-flash) returns 404; the agents that pin a
# model use gemini-3.8-flash, so the whole pipeline runs on the same working model.
MODEL = "gemini-3.8-flash"

# Ordered stage names, exposed for documentation and offline tests without building
# any ADK object. "RefineLoop" expands to Verifier -> Refiner, repeated up to twice.
PIPELINE_STAGES = ("Extractor", "Simplifier", "RefineLoop", "Readability")
REFINE_LOOP_STAGES = ("Verifier", "Refiner")
MAX_REFINE_ITERATIONS = 2

# Draft-mode Verifier used inside the loop.
#
# agents/verifier.py (owned by the Verifier teammate) is currently specialized for
# MedAESQA scoring: its instruction is "{evaluation_prompt}" and it does not read the
# draft/source from state. The Refiner<->Verifier loop needs a Verifier that checks the
# current patient-facing draft against the source and emits the four-field verdict the
# Refiner consumes (faithful / unsupported_claims / omissions / reading_level_ok).
#
# This pipeline-local agent provides exactly that. It intentionally mirrors the four
# fields agents/verifier.py.EXPECTED_OUTPUT_FIELDS once documented. When the Verifier
# owner adds a draft-mode builder to agents/verifier.py, replace this with it so the
# prompt lives in one place.
DRAFT_VERIFIER_INSTRUCTION = (
    "Compare the generated explanation with the original medical source. Return one JSON "
    "object with exactly these fields: faithful (boolean), unsupported_claims (list of "
    "strings), omissions (list of strings), and reading_level_ok (boolean). Be strict: an "
    "unsupported clinical claim is a failure. List each unsupported claim and each clinically "
    "important omission verbatim enough that it can be located in the text.\n\n"
    "ORIGINAL MEDICAL SOURCE:\n{source_text}\n\nGENERATED EXPLANATION:\n{draft}"
)


def create_draft_verifier_agent(model: str = MODEL) -> Any:
    """Create the draft-mode Verifier used inside the bounded loop.

    Reads ``source_text`` and ``draft`` from state and writes ``verdict``. Building it
    does not call Gemini.
    """
    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before building the ADK pipeline.") from exc
    return LlmAgent(
        name="Verifier",
        model=model,
        instruction=DRAFT_VERIFIER_INSTRUCTION,
        output_key="verdict",
    )


def build_pipeline(model: str = MODEL):
    """Create the current experimental ADK workflow from the individual agents."""

    try:
        from google.adk.agents import LoopAgent, SequentialAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before building the ADK pipeline.") from exc

    extractor = create_extractor_agent(model)       # source_text -> atoms
    simplifier = create_simplifier_agent(model)     # source_text + atoms -> draft
    verifier = create_draft_verifier_agent(model)   # source_text + draft -> verdict
    refiner = create_refiner_agent(model)           # source_text + draft + verdict -> draft
    readability = create_readability_agent(model)   # draft -> final

    # The one and only loop. Its hard cap prevents open-ended Verifier/Refiner recursion;
    # because the Refiner returns a faithful draft unchanged, a second pass is cheap.
    refinement_loop = LoopAgent(
        name="RefineLoop",
        sub_agents=[verifier, refiner],
        max_iterations=MAX_REFINE_ITERATIONS,
    )
    return SequentialAgent(
        name="FaithfulMed",
        sub_agents=[extractor, simplifier, refinement_loop, readability],
    )


def pipeline_plan() -> list[str]:
    """Return the ordered stage names, expanding the bounded loop. No ADK, no Gemini.

    Useful for documentation and offline tests of the wiring.
    """
    plan: list[str] = []
    for stage in PIPELINE_STAGES:
        if stage == "RefineLoop":
            for _ in range(MAX_REFINE_ITERATIONS):
                plan.extend(REFINE_LOOP_STAGES)
        else:
            plan.append(stage)
    return plan


async def run_pipeline(source_text: str, model: str = MODEL) -> str:
    """Run the whole workflow on one medical source and return the final explanation.

    Seeds shared state with ``source_text`` and returns the Readability stage's final
    output. CALLS GEMINI AND MAY USE API QUOTA. Use scratch/dev material only.
    """
    return await run_agent_once(
        build_pipeline(model),
        state={"source_text": source_text},
        message="Produce the final patient-friendly explanation of the medical source.",
    )


if __name__ == "__main__":
    pipeline = build_pipeline()
    print("FaithfulMed ADK pipeline assembled:", pipeline.name)
    print("  Extractor -> Simplifier -> [Verifier -> Refiner, at most "
          f"{MAX_REFINE_ITERATIONS} passes] -> Readability")
    print("  Expanded stage order:", " -> ".join(pipeline_plan()))
    print("Building does not call Gemini. Use an ADK Runner (e.g. run_pipeline) only when you "
          "intend to use API quota.")
