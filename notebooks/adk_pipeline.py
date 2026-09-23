"""FaithfulMed's current Google ADK integration point.

The individual agents live in ``agents/``. This starter script imports and connects them as a
sequential workflow with one bounded Verifier/Refiner loop. It is still an experiment, not a
production pipeline, and several teammate-owned agents contain TODOs.

Setup:
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...        # key from Google AI Studio

Building the workflow does not call Gemini. Running it through an ADK runner does.
"""

from agents.extractor import create_extractor_agent
from agents.readability import create_readability_agent
from agents.refiner import create_refiner_agent
from agents.simplifier import create_simplifier_agent
from agents.verifier import create_verifier_agent


MODEL = "gemini-2.0-flash"


def build_pipeline(model: str = MODEL):
    """Create the current experimental ADK workflow from the five individual agents."""

    try:
        from google.adk.agents import LoopAgent, SequentialAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before building the ADK pipeline.") from exc

    extractor = create_extractor_agent(model)
    simplifier = create_simplifier_agent(model)
    verifier = create_verifier_agent(model, candidate_state_key="draft")
    refiner = create_refiner_agent(model)
    readability = create_readability_agent(model)

    # This remains the only loop. Its hard limit prevents open-ended Verifier/Refiner recursion.
    refinement_loop = LoopAgent(
        name="RefineLoop",
        sub_agents=[verifier, refiner],
        max_iterations=2,
    )
    return SequentialAgent(
        name="FaithfulMed",
        sub_agents=[extractor, simplifier, refinement_loop, readability],
    )


if __name__ == "__main__":
    pipeline = build_pipeline()
    print("FaithfulMed ADK pipeline assembled:", pipeline.name)
    print("  Extractor -> Simplifier -> [Verifier -> Refiner, at most 2 passes] -> Readability")
    print("Building does not call Gemini. Use an ADK Runner only when you intend to use API quota.")
