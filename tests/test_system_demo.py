"""Offline tests for the full-system demo evaluation workflow."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.factual_evaluator import factual_evaluator_prompt, indexed_text_spans
from agents.production_pipeline import PipelineRunResult
from evals.system_demo import (
    AppendOnlyRunCache,
    SystemExample,
    aggregate_metrics,
    cache_key,
    factual_metrics,
    load_mtsamples,
    load_plaba,
    parse_factual_evaluation,
    run_system_example,
    select_examples,
)


class SystemDemoTests(unittest.TestCase):
    def test_split_loaders_parse_sources_and_keep_references_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mts = root / "data" / "MTSamples" / "scratch" / "adjudicated"
            mts.mkdir(parents=True)
            (mts / "MTS_SCRATCH_01_gold.json").write_text(json.dumps({
                "example_id": "MTS_SCRATCH_01",
                "source_text": "Original medical report.",
                "facts": [{"fact_id": "F1", "fact_text": "A fact."}],
            }))
            plaba = root / "data" / "PLABA" / "scratch"
            plaba.mkdir(parents=True)
            (plaba / "PLABA_SCRATCH_01.json").write_text(json.dumps({
                "1": {
                    "PMID1": {
                        "Title": "Professional title",
                        "abstract": {"1": "First source sentence.", "2": "Second source sentence."},
                        "adaptations": {"adaptation1": {"1": "Plain reference."}},
                    }
                }
            }))

            mts_examples = load_mtsamples("scratch", root)
            plaba_examples = load_plaba("scratch", root)
            self.assertEqual(len(mts_examples), 1)
            self.assertEqual(mts_examples[0].source_text, "Original medical report.")
            self.assertEqual(mts_examples[0].references["adjudicated_facts"][0]["fact_id"], "F1")
            self.assertEqual(len(plaba_examples), 1)
            self.assertIn("Professional title", plaba_examples[0].source_text)
            self.assertIn("First source sentence. Second source sentence.", plaba_examples[0].source_text)
            self.assertNotIn("Plain reference.", plaba_examples[0].source_text)
            self.assertEqual(
                plaba_examples[0].references["plain_language_adaptations"][0]["text"],
                "Plain reference.",
            )

    def test_selection_alternates_datasets(self) -> None:
        examples = [
            SystemExample("MTSamples", "dev", "M1", "m1", {}, "m1"),
            SystemExample("MTSamples", "dev", "M2", "m2", {}, "m2"),
            SystemExample("PLABA", "dev", "P1", "p1", {}, "p1"),
            SystemExample("PLABA", "dev", "P2", "p2", {}, "p2"),
        ]
        selected = select_examples(examples, 3)
        self.assertEqual([(item.dataset, item.example_id) for item in selected],
                         [("MTSamples", "M1"), ("PLABA", "P1"), ("MTSamples", "M2")])

    def test_aggregate_metrics_uses_claim_and_output_denominators(self) -> None:
        records = [
            {
                "readability_metrics": {"flesch_kincaid_grade": 7, "meets_target": True, "smog_index": 8,
                                          "jargon_density": .1, "word_count": 100, "is_refusal": False},
                "factual_metrics": {"claims_assessed": 2, "supported_claims": 1, "uncertain_claims": 1,
                                      "clinically_meaningful_hallucination": False},
            },
            {
                "readability_metrics": {"flesch_kincaid_grade": 9, "meets_target": False, "smog_index": 10,
                                          "jargon_density": .2, "word_count": 200, "is_refusal": True},
                "factual_metrics": {"claims_assessed": 3, "supported_claims": 2, "uncertain_claims": 0,
                                      "clinically_meaningful_hallucination": True},
            },
            {"readability_metrics": None, "factual_metrics": {"claims_assessed": 0}},
        ]
        result = aggregate_metrics(records)
        self.assertEqual(result["percent_at_or_below_grade_8"], 50)
        self.assertEqual(result["mean_smog"], 9)
        self.assertEqual(result["mean_word_count"], 150)
        self.assertEqual(result["refusal_rate"], 50)
        self.assertEqual(result["factual_fidelity_estimate"], 60)
        self.assertEqual(result["hallucination_rate"], 50)
        self.assertEqual(result["factual_outputs_scored"], 2)
        self.assertEqual(result["uncertain_claims"], 1)
        self.assertEqual(result["claim_validation_coverage"], 100)

    def test_factual_prompt_uses_exact_numbered_spans_instead_of_copied_quotes(self) -> None:
        source = "First source sentence. Second source sentence."
        explanation = "* First explanation claim.\n* Second explanation claim."
        rendered = factual_evaluator_prompt(source, explanation)
        payload = json.loads(rendered.split("\n\nINPUT JSON:\n", 1)[1])
        self.assertEqual(
            payload["original_source_spans"],
            [
                {"span_id": "S001", "text": "First source sentence."},
                {"span_id": "S002", "text": "Second source sentence."},
            ],
        )
        self.assertEqual(
            payload["final_explanation_spans"],
            [
                {"span_id": "E001", "text": "* First explanation claim."},
                {"span_id": "E002", "text": "* Second explanation claim."},
            ],
        )
        self.assertNotIn("original_source_text", payload)
        self.assertNotIn("final_explanation", payload)

    def test_span_ids_resolve_claim_and_multiple_source_excerpts(self) -> None:
        source = "First source fact. Second source fact."
        explanation = "* Combined patient-facing claim."
        parsed = parse_factual_evaluation(json.dumps({
            "status": "scored", "status_reason": "x",
            "claims": [{
                "claim_id": "C1", "claim_span_id": "E001",
                "claim_text": "Combined claim.", "classification": "supported",
                "clinically_meaningful": True, "source_span_ids": ["S001", "S002"],
                "reason": "Both source spans support the claim.",
            }],
        }), explanation, source)
        claim = parsed["claims"][0]
        self.assertEqual(claim["claim_quote"], explanation)
        self.assertEqual(claim["claim_quote_match"], "span_id")
        self.assertEqual([item["span_id"] for item in claim["source_quotes"]], ["S001", "S002"])
        self.assertEqual(claim["source_quote"], "First source fact.\nSecond source fact.")
        metrics = factual_metrics(parsed)
        self.assertEqual(metrics["claim_validation_coverage"], 100)

    def test_invalid_evidence_span_is_separately_identified_and_unscored(self) -> None:
        parsed = parse_factual_evaluation(json.dumps({
            "status": "scored", "status_reason": "x",
            "claims": [{
                "claim_id": "C1", "claim_span_id": "E001",
                "claim_text": "A claim.", "classification": "supported",
                "clinically_meaningful": True, "source_span_ids": ["S999"],
                "reason": "Bad source ID.",
            }],
        }), "Final claim.", "Source fact.")
        self.assertEqual(parsed["claims"], [])
        skipped = parsed["skipped_claims"][0]
        self.assertEqual(skipped["validation_stage"], "evidence_alignment")
        self.assertEqual(skipped["validated_claim_quote"], "Final claim.")
        metrics = factual_metrics(parsed)
        self.assertEqual(metrics["claim_validation_coverage"], 0)
        self.assertIsNone(metrics["factual_fidelity_estimate"])

    def test_invalid_result_shapes_are_rejected_and_invalid_claims_are_skipped(self) -> None:
        with self.assertRaises(ValueError):
            parse_factual_evaluation('{"status":"scored","status_reason":"x","claims":[]}', "Final.")
        parsed_invalid = parse_factual_evaluation(json.dumps({
            "status": "scored", "status_reason": "x",
            "claims": [{"claim_id": "C1", "claim_quote": "Not present", "classification": "supported",
                        "clinically_meaningful": True, "source_quote": "Source", "reason": "x"}],
        }), "Final.")
        self.assertEqual(parsed_invalid["claims"], [])
        self.assertEqual(len(parsed_invalid["skipped_claims"]), 1)
        parsed = parse_factual_evaluation(
            '{"status":"no_claims","status_reason":"No claims.","claims":[]}', "Hello."
        )
        self.assertEqual(parsed["status"], "no_claims")

    def test_unique_formatting_normalization_resolves_to_exact_spans(self) -> None:
        parsed = parse_factual_evaluation(json.dumps({
            "status": "scored", "status_reason": "x",
            "claims": [{
                "claim_id": "C1", "claim_quote": "Patient: 14-month-old boy",
                "classification": "supported", "clinically_meaningful": True,
                "source_quote": "The patient’s fever improved.", "reason": "Supported.",
            }],
        }), "* **Patient:** 14-month-old boy", "The patient's fever\n improved.")
        claim = parsed["claims"][0]
        self.assertEqual(claim["claim_quote_match"], "normalized_unique")
        self.assertIn(claim["claim_quote"], "* **Patient:** 14-month-old boy")
        self.assertEqual(claim["source_quote_match"], "normalized_unique")
        self.assertIn(claim["source_quote"], "The patient's fever\n improved.")

    def test_unmatched_or_ambiguous_claim_is_skipped_without_erasing_valid_claims(self) -> None:
        parsed = parse_factual_evaluation(json.dumps({
            "status": "scored", "status_reason": "x",
            "claims": [
                {"claim_id": "C1", "claim_quote": "Valid claim.", "classification": "supported",
                 "clinically_meaningful": True, "source_quote": "Source fact.", "reason": "Supported."},
                {"claim_id": "C2", "claim_quote": "Rewritten claim.", "classification": "unsupported",
                 "clinically_meaningful": False, "source_quote": None, "reason": "Unsupported."},
                {"claim_id": "C3", "claim_quote": "Patient: repeated.", "classification": "supported",
                 "clinically_meaningful": False, "source_quote": None, "reason": "Repeated."},
            ],
        }), "Valid claim. **Patient:** repeated. Then *Patient:* repeated.", "Source fact.")
        self.assertEqual([claim["claim_id"] for claim in parsed["claims"]], ["C1"])
        self.assertEqual(len(parsed["skipped_claims"]), 2)
        metrics = factual_metrics(parsed)
        self.assertEqual(metrics["claims_assessed"], 1)
        self.assertEqual(metrics["claims_skipped"], 2)
        self.assertEqual(metrics["factual_fidelity_estimate"], 100)
        self.assertIsNone(metrics["clinically_meaningful_hallucination"])

    def test_cache_hit_and_forced_run_behavior(self) -> None:
        calls = {"pipeline": 0, "evaluator": 0}
        seen_sources: list[str] = []

        async def pipeline(source, model, on_event=None):
            calls["pipeline"] += 1
            seen_sources.append(source)
            return PipelineRunResult(
                {"status": "completed", "stages": [], "results": {"final_explanation": "Final claim."}},
                "Final claim.", True,
            )

        async def evaluator(source, final, references, model, on_text=None):
            calls["evaluator"] += 1
            return json.dumps({
                "status": "scored", "status_reason": "Scored.",
                "claims": [{
                    "claim_id": "C1", "claim_quote": "Final claim.", "classification": "supported",
                    "clinically_meaningful": True, "source_quote": source, "reason": "Supported.",
                }],
            })

        example = SystemExample(
            "MTSamples", "dev", "M1", "Only source text.",
            {"adjudicated_facts": [{"fact_id": "GOLD"}]}, "sample.json",
        )
        readable = {"flesch_kincaid_grade": 7.0, "smog_index": 8.0, "jargon_density": .1,
                    "word_count": 10, "meets_target": True, "is_refusal": False}
        with tempfile.TemporaryDirectory() as directory, patch(
            "evals.system_demo.output_metrics", return_value=(readable, None)
        ):
            cache = AppendOnlyRunCache(Path(directory) / "runs.jsonl")
            first = asyncio.run(run_system_example(
                example, "pipeline-model", "eval-model", cache=cache,
                pipeline_runner=pipeline, evaluator_runner=evaluator,
            ))
            cached = asyncio.run(run_system_example(
                example, "pipeline-model", "eval-model", cache=cache,
                pipeline_runner=pipeline, evaluator_runner=evaluator,
            ))
            forced = asyncio.run(run_system_example(
                example, "pipeline-model", "eval-model", cache=cache, force=True,
                pipeline_runner=pipeline, evaluator_runner=evaluator,
            ))
            lines = cache.path.read_text().splitlines()

        self.assertEqual(calls, {"pipeline": 2, "evaluator": 2})
        self.assertEqual(seen_sources, ["Only source text.", "Only source text."])
        self.assertEqual(first["result_source"], "fresh")
        self.assertNotIn("claim_screening", first)
        self.assertEqual(cached["result_source"], "cache")
        self.assertEqual(forced["result_source"], "fresh_forced")
        self.assertNotEqual(first["run_id"], forced["run_id"])
        self.assertEqual(len(lines), 2)
        self.assertNotEqual(cache_key(example, "pipeline-model", "eval-model"),
                            cache_key(example, "different-model", "eval-model"))

    def test_invalid_evaluator_keeps_readability_metrics_and_raw_output(self) -> None:
        async def pipeline(source, model, on_event=None):
            return PipelineRunResult({"status": "completed", "stages": []}, "Final claim.", True)

        async def evaluator(source, final, references, model, on_text=None):
            return "not json"

        readable = {"flesch_kincaid_grade": 7.0, "smog_index": 8.0, "jargon_density": .1,
                    "word_count": 10, "meets_target": True, "is_refusal": False}
        example = SystemExample("Pasted", "session", "P1", "Source.", {}, "session")
        with patch("evals.system_demo.output_metrics", return_value=(readable, None)):
            record = asyncio.run(run_system_example(
                example, "model", "model", persist=False,
                pipeline_runner=pipeline, evaluator_runner=evaluator,
            ))
        self.assertEqual(record["readability_metrics"], readable)
        self.assertEqual(record["factual_evaluator"]["raw_output"], "not json")
        self.assertIn("JSONDecodeError", record["factual_evaluator"]["error"])
        self.assertIsNone(record["factual_metrics"]["factual_fidelity_estimate"])


if __name__ == "__main__":
    unittest.main()
