"""Durable human review of factual-evaluator hallucination candidates."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_PATH = ROOT / "eval_results" / "system_claim_reviews.jsonl"
REVIEW_SOURCE = "factual_evaluator"
REVIEW_STATUSES = {"hallucination", "not_hallucination", "ignore"}


def reviewable_claims(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evaluator claims eligible for human hallucination review."""
    evaluator = record.get("factual_evaluator") or {}
    parsed = evaluator.get("parsed_output")
    if not isinstance(parsed, dict) or parsed.get("status") != "scored":
        return []
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        return []
    return [
        copy.deepcopy(claim)
        for claim in claims
        if isinstance(claim, dict)
        and claim.get("clinically_meaningful") is True
        and claim.get("classification") in {"unsupported", "contradicted"}
        and isinstance(claim.get("claim_id"), str)
    ]


class ReviewDecisionStore:
    """Append-only human decisions, separate from the model-output cache."""

    def __init__(self, path: Path = DEFAULT_REVIEW_PATH):
        self.path = path

    def all_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events

    def latest(self) -> dict[tuple[str, str], dict[str, Any]]:
        decisions: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self.all_events():
            # Legacy decisions belonged to the removed second screener. They must
            # not attach to evaluator claims that happen to reuse the same ID.
            if event.get("review_source") != REVIEW_SOURCE:
                continue
            run_id = event.get("run_id")
            claim_id = event.get("claim_id")
            if isinstance(run_id, str) and isinstance(claim_id, str):
                decisions[(run_id, claim_id)] = event
        return decisions

    def for_run(self, run_id: str) -> dict[str, dict[str, Any]]:
        return {
            claim_id: event
            for (event_run_id, claim_id), event in self.latest().items()
            if event_run_id == run_id
        }

    def append(
        self,
        run_id: str,
        claim_id: str,
        status: str,
        *,
        metric_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not run_id or not claim_id:
            raise ValueError("run_id and claim_id are required.")
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status: {status}")
        event = {
            "schema": "faithfulmed_factual_claim_review_v2",
            "review_source": REVIEW_SOURCE,
            "run_id": run_id,
            "claim_id": claim_id,
            "status": status,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if metric_snapshot is not None:
            event["metric_snapshot_after_decision"] = copy.deepcopy(metric_snapshot)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


def review_metrics(
    records: list[dict[str, Any]],
    decisions: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Apply human overrides, defaulting eligible claims to Gemini's verdict."""
    counts = {
        "hallucination": 0,
        "not_hallucination": 0,
        "ignored": 0,
        "gemini_default": 0,
        "human_reviewed": 0,
    }
    completed_outputs = 0
    unavailable_outputs = 0
    no_candidate_outputs = 0
    hallucination_outputs = 0

    for record in records:
        evaluator = record.get("factual_evaluator") or {}
        parsed = evaluator.get("parsed_output")
        if (
            not record.get("final_explanation")
            or evaluator.get("error")
            or not isinstance(parsed, dict)
            or parsed.get("status") != "scored"
        ):
            unavailable_outputs += 1
            continue

        claims = reviewable_claims(record)
        completed_outputs += 1
        if not claims:
            no_candidate_outputs += 1
            continue

        effective_statuses: list[str] = []
        for claim in claims:
            event = decisions.get((record.get("run_id"), claim["claim_id"]))
            status = event.get("status") if event else None
            if status in REVIEW_STATUSES:
                counts["human_reviewed"] += 1
            else:
                status = "hallucination"
                counts["gemini_default"] += 1
            effective_statuses.append(status)
            if status == "hallucination":
                counts["hallucination"] += 1
            elif status == "not_hallucination":
                counts["not_hallucination"] += 1
            elif status == "ignore":
                counts["ignored"] += 1

        if "hallucination" in effective_statuses:
            hallucination_outputs += 1

    no_hallucination_outputs = completed_outputs - hallucination_outputs
    claim_denominator = counts["hallucination"] + counts["not_hallucination"]
    return {
        **counts,
        "eligible_outputs": completed_outputs,
        "completed_outputs": completed_outputs,
        "unavailable_outputs": unavailable_outputs,
        "no_candidate_outputs": no_candidate_outputs,
        "hallucination_outputs": hallucination_outputs,
        "no_hallucination_outputs": no_hallucination_outputs,
        "hallucination_rate": (
            hallucination_outputs / completed_outputs * 100 if completed_outputs else None
        ),
        "no_hallucination_rate": (
            no_hallucination_outputs / completed_outputs * 100 if completed_outputs else None
        ),
        "reviewed_claim_denominator": claim_denominator,
        "not_hallucination_claim_rate": (
            counts["not_hallucination"] / claim_denominator * 100
            if claim_denominator else None
        ),
    }
