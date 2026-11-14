"""Offline tests for adapting cached baseline outputs to the system demo."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from evals.baseline_system import (
    BaselineEnrichmentCache,
    enrich_baseline_record,
    has_compatible_source_matching,
    has_factual_evaluation,
    load_baseline_records,
    run_forced_baseline,
)
from evals.source_matching import EmbeddingCache, EmbeddingSettings


def baseline_row(example_id: str = "MTS_DEV_01") -> dict:
    return {
        "split": "dev",
        "example_id": example_id,
        "source_file": f"data/MTSamples/dev/adjudicated/{example_id}_gold.json",
        "source_text": "Source fact.",
        "model": "baseline-model",
        "generated_text": "Patient-friendly claim.",
        "metrics": {
            "flesch_kincaid_grade": 6.5,
            "smog_index": 7.0,
            "jargon_density": 0.1,
            "word_count": 3,
            "meets_target": True,
            "is_refusal": False,
        },
        "status": "success",
        "error": None,
    }


class RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, settings):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class BaselineSystemTests(unittest.TestCase):
    def test_loader_keeps_latest_success_and_ignores_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "baseline.jsonl"
            first = baseline_row()
            latest = {**first, "generated_text": "Updated cached explanation."}
            failed = {**baseline_row("FAILED"), "status": "failed"}
            results.write_text(
                "\n".join([
                    json.dumps(first),
                    "not json",
                    json.dumps(failed),
                    json.dumps(latest),
                ]) + "\n",
                encoding="utf-8",
            )
            cache = BaselineEnrichmentCache(root / "enriched.jsonl")

            records = load_baseline_records(
                "evaluator-model", results_path=results, cache=cache,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["final_explanation"], "Updated cached explanation.")
            self.assertEqual(records[0]["readability_metrics"]["flesch_kincaid_grade"], 6.5)
            self.assertEqual(records[0]["result_source"], "baseline_generation_cache")
            self.assertFalse(has_factual_evaluation(records[0]))

    def test_enrichment_calls_only_evaluator_and_embeddings_then_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "baseline.jsonl"
            results.write_text(json.dumps(baseline_row()) + "\n", encoding="utf-8")
            cache = BaselineEnrichmentCache(root / "enriched.jsonl")
            vector_cache = EmbeddingCache(root / "embeddings.sqlite3")
            settings = EmbeddingSettings(output_dimensionality=2)
            embedder = RecordingEmbedder()
            evaluator_calls = []

            async def evaluator(source, explanation, references, model, on_text=None):
                evaluator_calls.append((source, explanation, model))
                return json.dumps({
                    "status": "scored",
                    "status_reason": "Assessed.",
                    "claims": [{
                        "claim_id": "C1",
                        "claim_span_id": "E001",
                        "claim_text": "Patient-friendly claim.",
                        "classification": "supported",
                        "clinically_meaningful": True,
                        "source_span_ids": ["S001"],
                        "reason": "The source supports the claim.",
                    }],
                })

            record = load_baseline_records(
                "evaluator-model", results_path=results, cache=cache,
            )[0]
            enriched = asyncio.run(enrich_baseline_record(
                record,
                "evaluator-model",
                cache=cache,
                embedding_cache=vector_cache,
                embedding_settings=settings,
                embed_batch=embedder,
                evaluator_runner=evaluator,
            ))

            self.assertEqual(len(evaluator_calls), 1)
            self.assertEqual(len(embedder.calls), 1)
            self.assertTrue(has_factual_evaluation(enriched))
            self.assertTrue(has_compatible_source_matching(enriched, settings))
            self.assertEqual(enriched["factual_metrics"]["factual_fidelity_estimate"], 100)
            self.assertEqual(enriched["pipeline_trace"]["stages"][0]["status"], "loaded_from_cache")

            reloaded = load_baseline_records(
                "evaluator-model", results_path=results, cache=cache,
            )[0]
            reused = asyncio.run(enrich_baseline_record(
                reloaded,
                "evaluator-model",
                cache=cache,
                embedding_cache=vector_cache,
                embedding_settings=settings,
                embed_batch=embedder,
                evaluator_runner=evaluator,
            ))
            self.assertEqual(len(evaluator_calls), 1)
            self.assertEqual(len(embedder.calls), 1)
            self.assertEqual(reused["result_source"], "baseline_enrichment_cache")
            self.assertEqual(len(cache.records()), 1)

    def test_forced_evaluator_run_gets_new_human_review_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "baseline.jsonl"
            results.write_text(json.dumps(baseline_row()) + "\n", encoding="utf-8")
            cache = BaselineEnrichmentCache(root / "enriched.jsonl")
            vector_cache = EmbeddingCache(root / "embeddings.sqlite3")
            settings = EmbeddingSettings(output_dimensionality=2)
            embedder = RecordingEmbedder()

            async def evaluator(source, explanation, references, model, on_text=None):
                return json.dumps({
                    "status": "no_claims",
                    "status_reason": "No factual claims.",
                    "claims": [],
                })

            record = load_baseline_records(
                "evaluator-model", results_path=results, cache=cache,
            )[0]
            first = asyncio.run(enrich_baseline_record(
                record, "evaluator-model", cache=cache,
                embedding_cache=vector_cache, embedding_settings=settings,
                embed_batch=embedder, evaluator_runner=evaluator,
            ))
            forced = asyncio.run(enrich_baseline_record(
                first, "evaluator-model", cache=cache,
                embedding_cache=vector_cache, embedding_settings=settings,
                embed_batch=embedder, evaluator_runner=evaluator,
                force_evaluator=True,
            ))

            self.assertNotEqual(first["run_id"], forced["run_id"])
            self.assertEqual(len(cache.records()), 2)

    def test_force_new_run_regenerates_answer_and_reuses_source_vector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "baseline.jsonl"
            original_line = json.dumps(baseline_row()) + "\n"
            results.write_text(original_line, encoding="utf-8")
            cache = BaselineEnrichmentCache(root / "enriched.jsonl")
            vector_cache = EmbeddingCache(root / "embeddings.sqlite3")
            settings = EmbeddingSettings(output_dimensionality=2)
            embedder = RecordingEmbedder()
            generation_calls = []
            evaluator_calls = []

            async def evaluator(source, explanation, references, model, on_text=None):
                evaluator_calls.append(explanation)
                return json.dumps({
                    "status": "scored",
                    "status_reason": "Assessed.",
                    "claims": [{
                        "claim_id": "C1",
                        "claim_span_id": "E001",
                        "claim_text": explanation,
                        "classification": "supported",
                        "clinically_meaningful": True,
                        "source_span_ids": ["S001"],
                        "reason": "Supported for this synthetic test.",
                    }],
                })

            async def generator(source, model):
                generation_calls.append((source, model))
                return "New patient-friendly claim."

            record = load_baseline_records(
                "evaluator-model", results_path=results, cache=cache,
            )[0]
            original = asyncio.run(enrich_baseline_record(
                record, "evaluator-model", cache=cache,
                embedding_cache=vector_cache, embedding_settings=settings,
                embed_batch=embedder, evaluator_runner=evaluator,
            ))
            fresh = asyncio.run(run_forced_baseline(
                original, "evaluator-model", cache=cache,
                embedding_cache=vector_cache, embedding_settings=settings,
                embed_batch=embedder, baseline_runner=generator,
                evaluator_runner=evaluator,
            ))

            self.assertEqual(generation_calls, [("Source fact.", "baseline-model")])
            self.assertEqual(evaluator_calls[-1], "New patient-friendly claim.")
            self.assertEqual(fresh["final_explanation"], "New patient-friendly claim.")
            self.assertEqual(fresh["generation_mode"], "forced")
            self.assertEqual(fresh["result_source"], "baseline_fresh_generation")
            self.assertNotEqual(original["run_id"], fresh["run_id"])
            self.assertEqual(embedder.calls[0], ["Source fact.", "Patient-friendly claim."])
            self.assertEqual(embedder.calls[1], ["New patient-friendly claim."])
            self.assertEqual(results.read_text(encoding="utf-8"), original_line)

            latest = load_baseline_records(
                "evaluator-model", results_path=results, cache=cache,
            )[0]
            self.assertEqual(latest["final_explanation"], "New patient-friendly claim.")


if __name__ == "__main__":
    unittest.main()
