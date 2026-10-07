"""Offline Extractor evaluation Tests."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from evals.extractor_eval import (
    generate_predictions, load_gold, load_predictions, parse_prediction,
    save_predictions, score_predictions,
)

SOURCE = "The patient is 91 years old. The patient denies chest pain."


def atom(text="The patient denies chest pain.", kind="diagnosis"):
    return {"fact_text": text, "fact_type": kind, "source_evidence": [SOURCE.split(". ")[1]]}


def example(facts=None):
    facts = [atom()] if facts is None else facts
    return {"example_id": "scratch-1", "source_text": SOURCE,
            "facts": [dict(fact, fact_id=f"F{i}") for i, fact in enumerate(facts, 1)]}


def score(facts, gold=None):
    return score_predictions([example(gold)], [
        {"example_id": "scratch-1", "raw_prediction": json.dumps(facts)}
    ])


class ExtractorEvalTests(unittest.TestCase):
    def test_perfect(self):
        result = score([atom()])
        self.assertEqual(result["exact_f1"], 1)
        self.assertEqual(result["evidence_verbatim_rate"], 1)

    def test_extra_missing(self):
        gold = [atom(), atom("The patient is 91 years old.", "demographics")]
        result = score([atom(), atom("The patient has a fever.")], gold)
        self.assertEqual(result["exact_precision"], .5)
        self.assertEqual(result["exact_recall"], .5)

    def test_duplicates_do_not_inflate_matches(self):
        result = score([atom(), atom()])
        self.assertEqual(result["matched_atoms"], 1)
        self.assertEqual(result["exact_precision"], .5)

    def test_wrong_type(self):
        self.assertEqual(score([atom(kind="procedure")])["matched_atoms"], 0)

    def test_paraphrase_is_unmatched_baseline(self):
        result = score([atom("No chest pain was reported.")])
        self.assertEqual(result["matched_atoms"], 0)

    def test_negation_and_numbers_preserved(self):
        self.assertEqual(score([atom("The patient has chest pain.")])["matched_atoms"], 0)
        result = score([atom("The patient is 19 years old.", "demographics")],
                       [atom("The patient is 91 years old.", "demographics")])
        self.assertEqual(result["matched_atoms"], 0)

    def test_malformed_output(self):
        for raw in ["not JSON", "{}", '[{"fact_text": "x"}]',
                    json.dumps([atom(kind="unknown")]),
                    json.dumps([dict(atom(), source_evidence="sentence")])]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_prediction(raw)

    def test_invalid_and_missing_reported(self):
        result = score_predictions([example()], [
            {"example_id": "scratch-1", "raw_prediction": "bad"}
        ])
        self.assertEqual(result["schema_valid_rate"], 0)
        self.assertIsNone(result["exact_f1"])
        self.assertIn("scratch-1", result["invalid_predictions"])
        result = score_predictions([example()], [])
        self.assertEqual(result["missing_prediction_ids"], ["scratch-1"])

    def test_empty_predictions(self):
        result = score([])
        self.assertIsNone(result["exact_precision"])
        self.assertEqual(result["exact_recall"], 0)
        self.assertEqual(result["exact_f1"], 0)
        self.assertIsNone(score([], [])["exact_f1"])

    def test_evidence_check_is_separate(self):
        result = score([dict(atom(), source_evidence=["Invented sentence."])])
        self.assertEqual(result["exact_f1"], 1)
        self.assertEqual(result["evidence_verbatim_rate"], 0)

    def test_round_trip_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            records = [{"example_id": "scratch-1", "raw_prediction": "bad"}]
            save_predictions(path, records)
            self.assertEqual(load_predictions(path), records)
            with self.assertRaises(ValueError):
                save_predictions(path, records * 2)

    def test_gold_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
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
        async def fake_runner(source):
            seen.append(source)
            return json.dumps([atom()])
        records = asyncio.run(generate_predictions([example()], fake_runner))
        self.assertEqual(seen, [SOURCE])
        self.assertEqual(records[0]["example_id"], "scratch-1")


if __name__ == "__main__":
    unittest.main()
