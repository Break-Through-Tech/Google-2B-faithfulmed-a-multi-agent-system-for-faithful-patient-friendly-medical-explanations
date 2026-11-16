"""Independent evidence-fidelity evaluator for completed full-system outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from . import run_agent_once, run_agent_stream
from .verifier import VERIFIER_DEFAULT_MODEL


FACTUAL_EVALUATOR_PROMPT = """You are an independent factual-fidelity evaluator for FaithfulMed. Return one valid JSON object only: no Markdown, code fences, or text outside the JSON.

You receive numbered spans from an original medical source, numbered spans from a final patient-friendly explanation, optional UMLS reference definitions, and optional post-generation references. Ignore every prediction or verdict produced by the pipeline's Verifier. Judge only against the supplied report spans and UMLS definitions, not unsupported outside knowledge.

The original source is authoritative for every patient-specific claim. MTSamples adjudicated facts may help locate relevant source passages, but they are not a checklist of facts that the explanation must include. PLABA plain-language adaptations are alternative human references, not required wording and not a substitute for the source. Do not penalize omission of optional information. Treat faithful paraphrases as supported.

UMLS REFERENCE-DEFINITION BOUNDARY
- A general term explanation may be supported by an explicit definition in the
  medical source or by a matching UMLS definition. UMLS may support only a short,
  general explanation of the matching medical term.
- It cannot establish that this patient currently has a condition, has its typical symptoms, cause, complications, or severity, received a medication for its usual indication, or should receive a treatment.
- A medication definition cannot establish why this patient takes that medication.
- Preserve report qualifiers such as "family history of," "history of," "possible," "suspected," "resolved," and "ruled out."
- Split a mixed sentence into separate patient-specific and general-definition claims. The patient-specific claim requires report evidence; the general definition may cite its matching UMLS source_id.
- Never cite UMLS for a patient-specific claim. If that claim lacks report support, classify it as unsupported. If a general explanation conflicts with its matching UMLS definition, classify it as contradicted.

Split compound explanation spans into atomic factual claims. Multiple atomic claims may use the same explanation span. For every factual claim, return the exact explanation span ID containing it, a brief atomic claim description, whether it is clinically meaningful, and one classification:
- "supported": the report supports a patient-specific claim; the report
  explicitly supports a general term explanation; or a matching UMLS definition
  supports a general term explanation.
- "unsupported": the allowed evidence does not establish the claim and does not explicitly conflict with it.
- "contradicted": the report or matching UMLS definition explicitly conflicts with the claim.
- "uncertain": the supplied material is insufficient or ambiguous.

Span handling is strict:
- Copy claim_span_id from final_explanation_spans. Never invent or alter an ID.
- Use source_span_ids only from original_source_spans. Return every source span needed to support or contradict the claim; multiple IDs are allowed.
- For a patient-specific "supported" or "contradicted" claim, source_span_ids must contain at least one valid report source ID.
- For a general-definition "supported" claim, provide source_span_ids when the
  medical source explicitly states the explanation, reference_definition_ids
  when UMLS supports it, or both. A UMLS-based "contradicted" claim must contain
  a valid UMLS source_id.
- evidence_type must be "medical_report", "umls_definition", "both", or "none" and must match the supplied IDs.
- claim_type must be "patient_specific" or "general_term_explanation". A patient_specific claim must not cite UMLS evidence at all; split any general explanation into its own claim.
- For "unsupported" or "uncertain", use an empty source_span_ids list when no source span supplies relevant evidence. Do not cite post-generation references as source evidence.
- Do not copy or reconstruct quotations. The application resolves IDs back to the exact stored text.

Keep each reason brief and source-grounded. Do not expose private chain-of-thought. If the output contains no factual claims, return status "no_claims" and an empty claims list. If the source is missing or unusable, return status "insufficient_source" and an empty claims list. Otherwise return status "scored".

Return exactly this structure:
{
  "status": "scored|no_claims|insufficient_source",
  "status_reason": "string",
  "claims": [
    {
      "claim_id": "C1",
      "claim_span_id": "E001",
      "claim_text": "brief atomic factual claim represented by that span",
      "claim_type": "patient_specific|general_term_explanation",
      "classification": "supported|unsupported|contradicted|uncertain",
      "clinically_meaningful": true,
      "evidence_type": "medical_report|umls_definition|both|none",
      "source_span_ids": ["S001"],
      "reference_definition_ids": ["C0004096"],
      "reason": "string"
    }
  ]
}"""


def indexed_text_spans(text: str, prefix: str) -> list[dict[str, str]]:
    """Create stable, exact text spans for evaluator citation without quote copying."""
    pieces: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:#{1,6}\s|[-*+]\s)", line):
            pieces.append(line)
            continue
        pieces.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line)
            if part.strip()
        )
    return [
        {"span_id": f"{prefix}{index:03d}", "text": piece}
        for index, piece in enumerate(pieces, start=1)
    ]


def factual_evaluator_prompt(
    source_text: str,
    final_explanation: str,
    references: dict[str, Any] | None = None,
    reference_definitions: list[dict[str, Any]] | None = None,
) -> str:
    payload = factual_evaluator_request(
        source_text, final_explanation, references, reference_definitions,
    )
    return FACTUAL_EVALUATOR_PROMPT + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False)


def factual_evaluator_request(
    source_text: str,
    final_explanation: str,
    references: dict[str, Any] | None = None,
    reference_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the exact structured payload supplied to the evaluator."""
    return {
        "original_source_spans": indexed_text_spans(source_text, "S"),
        "final_explanation_spans": indexed_text_spans(final_explanation, "E"),
        "reference_definitions": reference_definitions or [],
        "post_generation_references": references or {},
    }


def create_factual_evaluator_agent(model: str = VERIFIER_DEFAULT_MODEL) -> Any:
    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the factual evaluator.") from exc
    return LlmAgent(
        name="FactualEvaluator",
        model=model,
        instruction="{evaluation_prompt}",
        output_key="factual_evaluation",
    )


async def run_factual_evaluator(
    source_text: str,
    final_explanation: str,
    references: dict[str, Any] | None = None,
    model: str = VERIFIER_DEFAULT_MODEL,
    *,
    reference_definitions: list[dict[str, Any]] | None = None,
    on_text: Callable[[str, str], None] | None = None,
    on_usage: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Run an evaluation call that is independent of the pipeline Verifier."""
    prompt = factual_evaluator_prompt(
        source_text, final_explanation, references, reference_definitions,
    )
    agent = create_factual_evaluator_agent(model)
    state = {"evaluation_prompt": prompt}
    message = "Return the requested independent factual-fidelity JSON object."
    usage_kw = {"on_usage": on_usage} if on_usage else {}  # only usage-aware callers pass it
    if on_text:
        return await run_agent_stream(agent, state=state, message=message, on_text=on_text, **usage_kw)
    return await run_agent_once(agent, state=state, message=message, **usage_kw)
