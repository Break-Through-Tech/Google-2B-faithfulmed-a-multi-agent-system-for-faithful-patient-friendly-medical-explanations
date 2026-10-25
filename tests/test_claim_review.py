"""Offline tests for human review of factual-evaluator claim candidates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.claim_review import ReviewDecisionStore, review_metrics, reviewable_claims


def evaluated_record(run_id: str, claims: list[dict] | None = None) -> dict:
    return {
        "run_id": run_id,
        "final_explanation": "A synthetic explanation.",
        "factual_evaluator": {
            "error": None,
            "parsed_output": {
                "status": "scored",
                "claims": claims or [],
            },
        },
    }


def candidate(claim_id: str = "C1", classification: str = "unsupported") -> dict:
    return {
        "claim_id": claim_id,
        "claim_quote": "Synthetic claim.",
        "classification": classification,
        "clinically_meaningful": True,
        "source_quotes": [],
        "reason": "Synthetic reason.",
    }


class ClaimReviewTests(unittest.TestCase):
    def test_only_clinically_meaningful_unsupported_or_contradicted_claims_are_reviewed(self) -> None:
        record = evaluated_record("run", [
            candidate("C1", "unsupported"),
            candidate("C2", "contradicted"),
            candidate("C3", "supported"),
            {**candidate("C4", "unsupported"), "clinically_meaningful": False},
            candidate("C5", "uncertain"),
        ])
        self.assertEqual(
            [item["claim_id"] for item in reviewable_claims(record)],
            ["C1", "C2"],
        )

    def test_gemini_verdict_is_default_until_human_override(self) -> None:
        records = [evaluated_record("run", [candidate()])]
        default = review_metrics(records, {})
        self.assertEqual(default["hallucination"], 1)
        self.assertEqual(default["gemini_default"], 1)
        self.assertEqual(default["human_reviewed"], 0)
        self.assertEqual(default["hallucination_rate"], 100)

        overridden = review_metrics(records, {
            ("run", "C1"): {"status": "not_hallucination"},
        })
        self.assertEqual(overridden["hallucination"], 0)
        self.assertEqual(overridden["not_hallucination"], 1)
        self.assertEqual(overridden["hallucination_rate"], 0)
        self.assertEqual(overridden["human_reviewed"], 1)

    def test_ignore_excludes_claim_without_counting_hallucination(self) -> None:
        result = review_metrics(
            [evaluated_record("run", [candidate()])],
            {("run", "C1"): {"status": "ignore"}},
        )
        self.assertEqual(result["ignored"], 1)
        self.assertEqual(result["reviewed_claim_denominator"], 0)
        self.assertEqual(result["hallucination_rate"], 0)

    def test_output_is_counted_once_when_multiple_claims_are_hallucinations(self) -> None:
        result = review_metrics([
            evaluated_record("run", [candidate("C1"), candidate("C2", "contradicted")]),
        ], {})
        self.assertEqual(result["hallucination"], 2)
        self.assertEqual(result["hallucination_outputs"], 1)
        self.assertEqual(result["hallucination_rate"], 100)

    def test_decisions_persist_and_legacy_screener_events_are_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.jsonl"
            store = ReviewDecisionStore(path)
            store.append("run-a", "C1", "hallucination", metric_snapshot={"eligible_outputs": 1})
            store.append("run-a", "C1", "not_hallucination")
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"run_id":"run-b","claim_id":"C1","status":"unsupported"}\n')

            restored = ReviewDecisionStore(path)
            self.assertEqual(restored.for_run("run-a")["C1"]["status"], "not_hallucination")
            self.assertEqual(restored.for_run("run-b"), {})
            self.assertEqual(
                restored.all_events()[0]["metric_snapshot_after_decision"]["eligible_outputs"], 1
            )


if __name__ == "__main__":
    unittest.main()
