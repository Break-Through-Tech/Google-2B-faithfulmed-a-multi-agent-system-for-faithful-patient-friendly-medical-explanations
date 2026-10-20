"""Tests for the offline Extractor evaluator and advisory semantic matcher."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from evals.extractor_eval import (
    generate_predictions, load_gold, load_predictions, parse_prediction,
    save_predictions, score_predictions, generate_semantic_reviews, semantic_review_queue,
)

SOURCE = "The patient is 91 years old. The patient denies chest pain."


def atom(text="The patient denies chest pain.", kind="diagnosis"):
    return {"fact_id": "P-001", "fact_text": text, "fact_type": kind,
            "source_evidence": ["The patient denies chest pain."]}


def example(facts=None):
    facts = [atom()] if facts is None else facts
    return {"example_id": "scratch-1", "source_text": SOURCE,
            "facts": [dict(f, fact_id=f"F{i}") for i, f in enumerate(facts, 1)]}


def score(facts, gold=None):
    return score_predictions([example(gold)], [
        {"example_id": "scratch-1", "raw_prediction": json.dumps([dict(f, fact_id=f"P-{i:03d}") for i, f in enumerate(facts, 1)])}
    ])


class ExtractorEvalTests(unittest.TestCase):
    def test_perfect(self):
        r = score([atom()])
        self.assertEqual(r["exact_f1"], 1)
        self.assertEqual(r["evidence_verbatim_rate"], 1)

    def test_extra_missing(self):
        r = score([atom(), atom("The patient has a fever.")],
                  [atom(), atom("The patient is 91 years old.", "demographics")])
        self.assertEqual(r["exact_precision"], .5)
        self.assertEqual(r["exact_recall"], .5)

    def test_duplicates_do_not_inflate_matches(self):
        r = score([atom(), atom()])
        self.assertEqual(r["matched_atoms"], 1)
        self.assertEqual(r["exact_precision"], .5)

    def test_wrong_type(self):
        self.assertEqual(score([atom(kind="procedure")])["matched_atoms"], 0)

    def test_paraphrase_is_unmatched_baseline(self):
        self.assertEqual(score([atom("No chest pain was reported.")])["matched_atoms"], 0)

    def test_negation_and_numbers_preserved(self):
        self.assertEqual(score([atom("The patient has chest pain.")])["matched_atoms"], 0)
        self.assertEqual(score([atom("The patient is 19 years old.", "demographics")],
                               [atom("The patient is 91 years old.", "demographics")])["matched_atoms"], 0)

    def test_prediction_ids_required_and_unique(self):
        with self.assertRaises(ValueError):
            parse_prediction(json.dumps([{k: v for k, v in atom().items() if k != "fact_id"}]))
        with self.assertRaises(ValueError):
            parse_prediction(json.dumps([atom(), atom()]))

    def test_semantic_uses_real_prediction_id(self):
        r = semantic_review_queue(example(), [dict(atom("Patient reports no chest pain."), fact_id="CUSTOM-7")], lambda _: [1., 0.])
        self.assertEqual(r["review_queue"][0]["prediction_id"], "CUSTOM-7")

    def test_malformed_output(self):
        for raw in ["not JSON", "{}", '[{"fact_text":"x"}]',
                    json.dumps([atom(kind="unknown")]),
                    json.dumps([dict(atom(), source_evidence="sentence")])]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_prediction(raw)

    def test_invalid_and_missing_reported(self):
        r = score_predictions([example()], [{"example_id": "scratch-1", "raw_prediction": "bad"}])
        self.assertEqual(r["schema_valid_rate"], 0)
        self.assertIsNone(r["exact_f1"])
        self.assertEqual(r["all_outputs"]["recall"], 0)
        self.assertIn("scratch-1", r["invalid_predictions"])
        r = score_predictions([example()], [])
        self.assertEqual(r["missing_prediction_ids"], ["scratch-1"])
        self.assertEqual(r["schema_valid_rate"], 0)

    def test_empty_predictions(self):
        r = score([])
        self.assertIsNone(r["exact_precision"])
        self.assertEqual(r["exact_recall"], 0)
        self.assertEqual(r["exact_f1"], 0)
        self.assertIsNone(score([], [])["exact_f1"])

    def test_evidence_check_is_separate(self):
        r = score([dict(atom(), source_evidence=["Invented sentence."])])
        self.assertEqual(r["exact_f1"], 1)
        self.assertEqual(r["evidence_verbatim_rate"], 0)

    def test_round_trip_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            records = [{"example_id": "scratch-1", "raw_prediction": "bad"}]
            save_predictions(path, records)
            self.assertEqual(load_predictions(path), records)
            with self.assertRaises(ValueError):
                save_predictions(path, records * 2)

    def test_gold_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gold.json"
            path.write_text(json.dumps(example()), encoding="utf-8")
            self.assertEqual(len(load_gold([path])), 1)
            with self.assertRaises(ValueError):
                load_gold([path, path])
            bad = example()
            bad["facts"][0]["source_evidence"] = ["Absent sentence."]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_gold([path])

    def test_runner_receives_only_source(self):
        seen = []
        async def runner(source):
            seen.append(source)
            return json.dumps([atom()])
        records = asyncio.run(generate_predictions([example()], runner))
        self.assertEqual(seen, [SOURCE])
        self.assertEqual(records[0]["example_id"], "scratch-1")

    def test_validity_and_two_micro_f1_scores(self):
        gold = [example(), {**example(), "example_id": "scratch-2"}]
        predictions = [{"example_id": "scratch-1", "raw_prediction": json.dumps([atom()])},
                       {"example_id": "scratch-2", "raw_prediction": "invalid"}]
        r = score_predictions(gold, predictions)
        self.assertEqual(r["schema_valid_rate"], .5)
        self.assertEqual(r["valid_only"]["f1"], 1)
        self.assertAlmostEqual(r["all_outputs"]["f1"], 2/3)

    def test_semantic_queue_paraphrase_and_unmatched(self):
        e = example([atom(), atom("Patient takes metformin.", "medication")])
        e["source_text"] += " Patient takes metformin."
        predicted = [atom(), dict(atom("Patient uses metformin.", "medication"), fact_id="P-002"),
                     dict(atom("Patient has asthma."), fact_id="P-003")]
        vectors = {"Patient takes metformin.": [1., 0.],
                   "Patient uses metformin.": [0.95, 0.1]}
        result = semantic_review_queue(e, predicted, lambda t: vectors[t])
        self.assertEqual(len(result["exact_matches"]), 1)
        self.assertEqual([r["review_type"] for r in result["review_queue"]],
                         ["semantic_candidate", "unmatched_prediction"])
        self.assertTrue(all(r["human_verdict"] is None for r in result["review_queue"]))

    def test_semantic_wrong_type_never_pairs(self):
        e = example()
        r = semantic_review_queue(e, [atom(kind="procedure")],
                                  lambda _: self.fail("Should not embed mismatched types"))
        self.assertEqual({x["review_type"] for x in r["review_queue"]},
                         {"unmatched_gold", "unmatched_prediction"})

    def test_semantic_negation_flag(self):
        r = semantic_review_queue(example(), [atom("The patient has chest pain.")],
                                  lambda _: [1., 0.])
        self.assertIn("negation_mismatch", r["review_queue"][0]["flags"])

    def test_generate_semantic_reviews_skips_invalid_and_missing(self):
        gold = [example(), {**example(), "example_id": "scratch-2"},
                {**example(), "example_id": "scratch-3"}]
        records = [
            {"example_id": "scratch-1", "raw_prediction": json.dumps([atom()])},
            {"example_id": "scratch-2", "raw_prediction": "invalid"},
        ]
        result = generate_semantic_reviews(
            gold, records, lambda _: self.fail("Exact matches should not call embeddings")
        )
        self.assertEqual(len(result["examples"]), 1)
        self.assertEqual(result["skipped_examples"]["scratch-3"], "missing_prediction")
        self.assertIn("scratch-2", result["skipped_examples"])
        self.assertEqual(result["status"], "advisory_only_not_human_validated")


if __name__ == "__main__":
    unittest.main()
