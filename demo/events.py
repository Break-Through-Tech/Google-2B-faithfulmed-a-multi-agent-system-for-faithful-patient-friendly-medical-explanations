"""Turn pipeline callbacks and saved run records into one browser event stream.

Live runs and replays emit the same event types, so the frontend renders both
the same way:

    run_started      {meta}
    stage_started    {stage}            stage without its (large) input
    stage_chunk      {stage_id, delta}  only the new text, never the accumulation
    stage_completed  {stage}            includes output and parsed_output
    pipeline_done    {status, accepted, final_explanation, latest_draft, error}
    evaluator_started
    evaluator_chunk  {delta}
    result           {scores, spans, claims}
    error            {message}
    done
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.factual_evaluator import indexed_text_spans

# "HISTORY OF PRESENT ILLNESS:" and similar headings that open a report section.
SECTION_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,/&()'-]{2,}:")

STAGE_FIELDS = (
    "stage_id", "agent", "iteration", "status", "started_at_utc", "completed_at_utc",
    "output", "parsed_output", "error",
)


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def public_stage(record: dict[str, Any], *, with_output: bool = True) -> dict[str, Any]:
    stage = {key: record.get(key) for key in STAGE_FIELDS}
    if not with_output:
        stage["output"] = ""
        stage["parsed_output"] = None
    return stage


def _source_grade(source_text: str) -> float | None:
    try:
        from evals.readability_eval import evaluate_single
        return evaluate_single(source_text).get("flesch_kincaid_grade")
    except Exception:  # noqa: BLE001 - best-effort metric; the run result does not depend on it
        return None


def _span_containing(quote: str | None, spans: list[dict[str, str]]) -> list[str]:
    """Map a legacy quote-based claim onto span IDs for highlighting."""
    if not quote:
        return []
    quote = " ".join(quote.split())
    hits = [s["span_id"] for s in spans if quote in s["text"] or s["text"] in quote]
    return hits


def _with_layout(text: str, spans: list[dict[str, str]]) -> list[dict[str, str]]:
    """Mark where each source span starts a new line or paragraph, so the report keeps its shape.

    Reports flattened onto one line (most MTSamples notes) also get a paragraph
    break before each ALL-CAPS section heading.
    """
    position = 0
    for index, span in enumerate(spans):
        found = text.find(span["text"], position)
        newlines = text[position:found].count("\n") if found >= 0 else 0
        if found >= 0:
            position = found + len(span["text"])
        heading = SECTION_HEADING.match(span["text"])
        if heading:
            span["heading"] = heading.group(0)
        if index and (newlines >= 2 or heading):
            span["break"] = "paragraph"
        elif index and newlines:
            span["break"] = "line"
    return spans


def result_event(record: dict[str, Any]) -> dict[str, Any]:
    """Scores plus the span-level evidence attribution for one finished run."""
    source_text = record.get("source_text") or ""
    final = record.get("final_explanation") or ""
    source_spans = _with_layout(source_text, indexed_text_spans(source_text, "S"))
    explanation_spans = indexed_text_spans(final, "E") if final else []
    evaluator = record.get("factual_evaluator") or {}
    parsed = evaluator.get("parsed_output") or {}
    reference_definitions = (
        (evaluator.get("request") or {}).get("reference_definitions")
        or (((record.get("pipeline_trace") or {}).get("results") or {}).get("reference_definitions") or {}).get("entries")
        or []
    )
    claims = []
    for claim in parsed.get("claims") or []:
        claim_span = claim.get("claim_span_id")
        source_ids = claim.get("source_span_ids")
        if claim_span is None:
            matches = _span_containing(claim.get("claim_quote"), explanation_spans)
            claim_span = matches[0] if matches else None
        if source_ids is None:
            source_ids = _span_containing(claim.get("source_quote"), source_spans)
        claims.append({
            "claim_id": claim.get("claim_id"),
            "claim_span_id": claim_span,
            "claim_text": claim.get("claim_text") or claim.get("claim_quote"),
            "claim_type": claim.get("claim_type"),
            "classification": claim.get("classification"),
            "clinically_meaningful": claim.get("clinically_meaningful"),
            "evidence_type": claim.get("evidence_type"),
            "source_span_ids": source_ids,
            "reference_definition_ids": claim.get("reference_definition_ids") or [],
            "reason": claim.get("reason"),
        })
    return {
        "type": "result",
        "readability": record.get("readability_metrics"),
        "readability_error": record.get("readability_error"),
        "source_grade": _source_grade(source_text),
        "factual": record.get("factual_metrics"),
        "evaluator_status": parsed.get("status"),
        "evaluator_status_reason": parsed.get("status_reason"),
        "evaluator_error": evaluator.get("error"),
        "source_spans": source_spans,
        "reference_definitions": reference_definitions,
        "explanation_spans": explanation_spans,
        "claims": claims,
        "references": record.get("post_generation_references") or {},
    }


def pipeline_done_event(trace: dict[str, Any], final: str | None, accepted: bool) -> dict[str, Any]:
    return {
        "type": "pipeline_done",
        "status": trace.get("status"),
        "accepted": accepted,
        "final_explanation": final,
        "latest_draft": trace.get("latest_draft"),
        "error": trace.get("error"),
    }
