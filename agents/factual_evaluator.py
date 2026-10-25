"""Independent source-fidelity evaluator for completed full-system outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from . import run_agent_once, run_agent_stream
from .verifier import VERIFIER_DEFAULT_MODEL


FACTUAL_EVALUATOR_PROMPT = """You are an independent factual-fidelity evaluator for FaithfulMed. Return one valid JSON object only: no Markdown, code fences, or text outside the JSON.

You receive numbered spans from an original medical source, numbered spans from a final patient-friendly explanation, and optional post-generation references. Ignore every prediction or verdict produced by the pipeline's Verifier. Judge fidelity to the supplied source spans, not whether a claim is medically correct according to outside knowledge.

The original source is authoritative. MTSamples adjudicated facts may help locate relevant source passages, but they are not a checklist of facts that the explanation must include. PLABA plain-language adaptations are alternative human references, not required wording and not a substitute for the source. Do not penalize omission of optional information. Treat faithful paraphrases as supported.

Split compound explanation spans into atomic factual claims. Multiple atomic claims may use the same explanation span. For every factual claim, return the exact explanation span ID containing it, a brief atomic claim description, whether it is clinically meaningful, and one classification:
- "supported": the source directly supports the claim or a faithful paraphrase of it.
- "unsupported": the source does not establish the claim and does not explicitly conflict with it.
- "contradicted": the source explicitly conflicts with the claim.
- "uncertain": the supplied material is insufficient or ambiguous.

Span handling is strict:
- Copy claim_span_id from final_explanation_spans. Never invent or alter an ID.
- Use source_span_ids only from original_source_spans. Return every source span needed to support or contradict the claim; multiple IDs are allowed.
- For "supported" and "contradicted", source_span_ids must contain at least one valid source ID.
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
      "classification": "supported|unsupported|contradicted|uncertain",
      "clinically_meaningful": true,
      "source_span_ids": ["S001"],
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
) -> str:
    payload = factual_evaluator_request(source_text, final_explanation, references)
    return FACTUAL_EVALUATOR_PROMPT + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False)


def factual_evaluator_request(
    source_text: str,
    final_explanation: str,
    references: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact structured payload supplied to the evaluator."""
    return {
        "original_source_spans": indexed_text_spans(source_text, "S"),
        "final_explanation_spans": indexed_text_spans(final_explanation, "E"),
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
    on_text: Callable[[str, str], None] | None = None,
) -> str:
    """Run an evaluation call that is independent of the pipeline Verifier."""
    prompt = factual_evaluator_prompt(source_text, final_explanation, references)
    agent = create_factual_evaluator_agent(model)
    state = {"evaluation_prompt": prompt}
    message = "Return the requested independent factual-fidelity JSON object."
    if on_text:
        return await run_agent_stream(agent, state=state, message=message, on_text=on_text)
    return await run_agent_once(agent, state=state, message=message)
