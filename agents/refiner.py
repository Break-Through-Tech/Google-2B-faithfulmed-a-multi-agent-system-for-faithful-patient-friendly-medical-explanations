"""Refiner agent owned by the Refiner teammate (Sameen).

The Refiner is the correction stage of the bounded Refiner <-> Verifier loop. It
takes the medical source, the current draft explanation, and the Verifier's
feedback, then makes targeted edits: remove the unsupported claims the Verifier
flagged, restore the omissions it flagged, and change nothing else. If the
Verifier says the draft is already faithful, the draft is returned unchanged.

Output contract (resolves the earlier TODO)
--------------------------------------------
The agent's final output is the **full revised explanation as plain text**, stored
under state key ``draft``. Plain text (not JSON) is deliberate so the bounded loop
can feed the revised draft straight back into the Verifier for re-checking. An
unchanged draft is represented by returning the draft text verbatim.

"What changed" is derived afterwards at the Python layer by diffing the input
draft against the output (``refine_and_describe`` -> ``RefinerResult``), reusing
``evals.refiner_eval``. The model is never asked to self-report its edits, which
keeps its output clean and loop-safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import run_agent_once

# The repository-wide default (gemini-2.0-flash) returned 404 in Sept 2026; the
# Verifier already pins a working model, so the Refiner matches it. Keep this
# override scoped to the Refiner.
REFINER_DEFAULT_MODEL = "gemini-3.8-flash"


_REFINER_INSTRUCTION = (
    "You are the Refiner in a medical-explanation pipeline. You make the smallest "
    "possible edits to a patient-friendly DRAFT so that it is faithful to the "
    "MEDICAL SOURCE, guided only by the VERIFIER FEEDBACK.\n\n"
    "Rules:\n"
    "1. The VERIFIER FEEDBACK may be a JSON object with fields: faithful (boolean), "
    "unsupported_claims (list of strings), and omissions (list of strings).\n"
    "2. If faithful is true, or there are no unsupported_claims and no omissions, "
    "return the DRAFT exactly as given, character for character.\n"
    "3. For each item in unsupported_claims: remove that claim from the draft. Do "
    "not leave a partial or reworded version of it.\n"
    "4. For each item in omissions: add that fact back, using only wording that is "
    "supported by the MEDICAL SOURCE.\n"
    "5. Change nothing else. Preserve the draft's sentences, order, tone, and "
    "reading level wherever they are not the target of a fix. Do not add new "
    "claims, do not re-summarize, do not polish.\n"
    "6. Never introduce information that is not in the MEDICAL SOURCE.\n\n"
    "Output ONLY the revised explanation as plain text. No preamble, no commentary, "
    "no markdown fences, no list of changes.\n\n"
    "MEDICAL SOURCE:\n{source_text}\n\nDRAFT:\n{draft}\n\nVERIFIER FEEDBACK:\n{verdict}"
)


def create_refiner_agent(model: str = REFINER_DEFAULT_MODEL) -> Any:
    """Create the Refiner ``LlmAgent``.

    ``output_key="draft"`` overwrites the draft in shared state so the bounded
    loop re-verifies the corrected version. Building the agent does not call
    Gemini; running it does.
    """

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Refiner agent.") from exc
    return LlmAgent(
        name="Refiner",
        model=model,
        instruction=_REFINER_INSTRUCTION,
        output_key="draft",
    )


def _verdict_to_text(verdict: Any) -> str:
    """Normalize Verifier feedback to a string for prompt injection.

    Accepts the raw JSON string the Verifier produces (passed through as-is, since
    the instruction knows how to read it), or a dict (serialized to JSON), or any
    other value (coerced to ``str``).
    """
    if isinstance(verdict, str):
        return verdict
    if isinstance(verdict, dict):
        return json.dumps(verdict, ensure_ascii=False)
    return str(verdict)


async def run_refiner(
    source_text: str,
    draft: str,
    verdict: Any,
    model: str = REFINER_DEFAULT_MODEL,
) -> str:
    """Run only the Refiner on one draft and its Verifier feedback.

    Returns the revised explanation as plain text (the draft verbatim when the
    feedback reports it is already faithful). CALLS GEMINI AND MAY USE API QUOTA.
    """

    return await run_agent_once(
        create_refiner_agent(model),
        state={
            "source_text": source_text,
            "draft": draft,
            "verdict": _verdict_to_text(verdict),
        },
        message="Refine the draft using the supplied feedback.",
    )


@dataclass(frozen=True)
class RefinerResult:
    """A Refiner run plus the derived description of what it did.

    ``changed`` and ``minimal_edit_ratio`` are computed by diffing the original
    draft against the revised text, not reported by the model. ``minimal_edit_ratio``
    is 0..1 (1.0 = identical); a well-behaved targeted edit stays close to 1.0.
    """

    draft: str
    revised_draft: str
    changed: bool
    minimal_edit_ratio: float


async def refine_and_describe(
    source_text: str,
    draft: str,
    verdict: Any,
    model: str = REFINER_DEFAULT_MODEL,
) -> RefinerResult:
    """Run the Refiner and return a structured result describing the edit.

    This is the convenience entry point for testing and the workbench: it gives
    both the revised text (for the loop) and a quick, offline-computed summary of
    how much changed. CALLS GEMINI AND MAY USE API QUOTA.
    """

    revised = await run_refiner(source_text, draft, verdict, model)
    return describe_refinement(draft, revised)


def describe_refinement(draft: str, revised_draft: str) -> RefinerResult:
    """Derive a :class:`RefinerResult` from a draft/revision pair. No model call.

    Separated out so it can be unit-tested offline without invoking Gemini.
    """
    from evals.refiner_eval import is_unchanged, minimal_edit_ratio

    return RefinerResult(
        draft=draft,
        revised_draft=revised_draft,
        changed=not is_unchanged(draft, revised_draft),
        minimal_edit_ratio=minimal_edit_ratio(draft, revised_draft),
    )
