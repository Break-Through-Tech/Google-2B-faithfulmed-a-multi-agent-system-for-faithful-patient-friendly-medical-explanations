"""Offline tests for Extractor evaluation and tiered matching."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from evals.extractor_eval import (
    cosine,
    generate_predictions,
    load_gold,
    load_predictions,
    match_facts,
    parse_prediction,
    save_predictions,
    score_predictions,
    score_tiered_predictions,
)


SOURCE = "The patient is 91 years old. The patient denies chest pain."


def atom(text="The patient denies chest pain.", kind="diagnosis"):
    return {
        "fact_text": text,
        "fact_type": kind,
        "source_evidence": [SOURCE.split(". ")[1]],
    }


def example(facts=None):
    facts = [atom()] if facts is None else facts
    return {
        "example_id": "scratch-1",
        "source_text": SOURCE,
        "facts": [
            dict(fact, fact_id=f"F{i}")
            for i, fact in enumerate(facts, 1)
        ],
    }


def score(facts, gold=None):
    return score_predictions(
        [example(gold)],
        [{
            "example_id": "scratch-1",
            "raw_prediction": json.dumps(facts),
        }],
    )


class ExtractorEvalTests(unittest.TestCase):
    def test_perfect(self):
        result = score([atom()])
        self.assertEqual(result["exact_f1"], 1)
        self.assertEqual(result["evidence_verbatim_rate"], 1)

    def test_extra_missing(self):
        gold = [
            atom(),
            atom("The patient is 91 years old.", "demographics"),
        ]
        result = score(
            [atom(), atom("The patient has a fever.")],
            gold,
        )
        self.assertEqual(result["exact_precision"], 0.5)
        self.assertEqual(result["exact_recall"], 0.5)

    def test_duplicates_do_not_inflate_matches(self):
        result = score([atom(), atom()])
        self.assertEqual(result["matched_atoms"], 1)
        self.assertEqual(result["exact_precision"], 0.5)

    def test_wrong_type(self):
        self.assertEqual(
            score([atom(kind="procedure")])["matched_atoms"], 0
        )

    def test_paraphrase_is_unmatched_baseline(self):
        self.assertEqual(
            score([atom("No chest pain was reported.")])["matched_atoms"],
            0,
        )

    def test_negation_and_numbers_preserved(self):
        self.assertEqual(
            score([atom("The patient has chest pain.")])["matched_atoms"],
            0,
        )
        result = score(
            [atom("The patient is 19 years old.", "demographics")],
            [atom("The patient is 91 years old.", "demographics")],
        )
        self.assertEqual(result["matched_atoms"], 0)

    def test_malformed_output(self):
        cases = [
            "not JSON",
            "{}",
            '[{"fact_text": "x"}]',
            json.dumps([atom(kind="unknown")]),
            json.dumps([dict(atom(), source_evidence="sentence")]),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_prediction(raw)

    def test_invalid_and_missing_reported(self):
        result = score_predictions(
            [example()],
            [{"example_id": "scratch-1", "raw_prediction": "bad"}],
        )
        self.assertEqual(result["schema_valid_rate"], 0)
        self.assertIsNone(result["exact_f1"])
        self.assertIn("scratch-1", result["invalid_predictions"])

        result = score_predictions([example()], [])
        self.assertEqual(
            result["missing_prediction_ids"], ["scratch-1"]
        )

    def test_empty_predictions(self):
        result = score([])
        self.assertIsNone(result["exact_precision"])
        self.assertEqual(result["exact_recall"], 0)
        self.assertEqual(result["exact_f1"], 0)
        self.assertIsNone(score([], [])["exact_f1"])

    def test_evidence_check_is_separate(self):
        result = score([
            dict(atom(), source_evidence=["Invented sentence."])
        ])
        self.assertEqual(result["exact_f1"], 1)
        self.assertEqual(result["evidence_verbatim_rate"], 0)

    def test_round_trip_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            records = [{
                "example_id": "scratch-1",
                "raw_prediction": "bad",
            }]
            save_predictions(path, records)
            self.assertEqual(load_predictions(path), records)

            with self.assertRaises(ValueError):
                save_predictions(path, records * 2)

    def test_gold_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(
                json.dumps(example()), encoding="utf-8"
            )
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

        records = asyncio.run(
            generate_predictions([example()], fake_runner)
        )
        self.assertEqual(seen, [SOURCE])
        self.assertEqual(records[0]["example_id"], "scratch-1")

    def test_mixed_valid_invalid_missing(self):
        gold = [
            dict(example(), example_id=str(i))
            for i in range(3)
        ]
        result = score_predictions(
            gold,
            [
                {
                    "example_id": "0",
                    "raw_prediction": json.dumps([atom()]),
                },
                {"example_id": "1", "raw_prediction": "bad"},
            ],
        )
        self.assertEqual(result["exact_f1_valid_only"], 1)
        self.assertEqual(result["exact_f1_all_outputs"], 0.5)
        self.assertEqual(result["exact_recall_all_outputs"], 1 / 3)
        self.assertEqual(result["schema_valid_rate"], 0.5)
        self.assertEqual(result["output_validity_rate"], 1 / 3)
        self.assertEqual(result["missing_output_rate"], 1 / 3)
        self.assertEqual(result["invalid_output_rate"], 1 / 3)
        self.assertEqual(
            result["exact_report_mean_f1_all_outputs"], 1 / 3
        )
        self.assertEqual(result["failed_output_gold_atoms"], 2)

    def test_all_failed(self):
        cases = [
            [],
            [{"example_id": "scratch-1", "raw_prediction": "bad"}],
        ]
        for records in cases:
            with self.subTest(records=records):
                result = score_predictions([example()], records)
                self.assertIsNone(result["exact_f1_valid_only"])
                self.assertEqual(result["exact_f1_all_outputs"], 0)
                self.assertEqual(result["output_validity_rate"], 0)

    def test_no_examples_has_no_scores(self):
        result = score_predictions([], [])
        keys = [
            "exact_f1_valid_only",
            "exact_f1_all_outputs",
            "output_validity_rate",
            "missing_output_rate",
            "exact_report_mean_f1_all_outputs",
        ]
        for key in keys:
            self.assertIsNone(result[key])

    def test_empty_gold_failed_report_still_zero(self):
        result = score_predictions([example([])], [])
        self.assertIsNone(result["exact_f1_all_outputs"])
        self.assertEqual(
            result["exact_report_mean_f1_all_outputs"], 0
        )
        self.assertEqual(
            score([], [])["exact_report_mean_f1_all_outputs"], 1
        )

    def test_all_valid_aliases_unchanged(self):
        result = score([atom(), atom()])
        for metric in ("precision", "recall", "f1"):
            self.assertEqual(
                result[f"exact_{metric}"],
                result[f"exact_{metric}_valid_only"],
            )
            self.assertEqual(
                result[f"exact_{metric}"],
                result[f"exact_{metric}_all_outputs"],
            )

    def test_micro_vs_report_mean_documented(self):
        gold = [
            dict(example(), example_id="good"),
            dict(
                example([
                    atom(),
                    atom("The patient is 91 years old.", "demographics"),
                    atom("A third gold fact."),
                ]),
                example_id="missing",
            ),
        ]
        result = score_predictions(
            gold,
            [{
                "example_id": "good",
                "raw_prediction": json.dumps([atom()]),
            }],
        )
        self.assertEqual(result["exact_f1_all_outputs"], 0.4)
        self.assertEqual(
            result["exact_report_mean_f1_all_outputs"], 0.5
        )
        self.assertEqual(
            result["unmatched_gold_atoms_all_outputs"], 3
        )

    def test_prediction_without_gold_does_not_affect_validity(self):
        result = score_predictions(
            [example()],
            [
                {
                    "example_id": "scratch-1",
                    "raw_prediction": json.dumps([atom()]),
                },
                {"example_id": "outside", "raw_prediction": "bad"},
            ],
        )
        self.assertEqual(result["output_validity_rate"], 1)
        self.assertEqual(
            result["prediction_ids_without_gold"], ["outside"]
        )


def fact(text, kind="diagnosis", fid=None):
    result = {
        "fact_text": text,
        "fact_type": kind,
        "source_evidence": ["Source sentence."],
    }
    return dict(result, fact_id=fid) if fid else result


def encode(texts):
    """Controlled test embeddings, not real model embeddings."""
    return [[1.0, 0.0] for _ in texts]


def match(predicted, gold, encoder=encode):
    return match_facts(
        predicted,
        gold,
        encoder,
        review_threshold=0.8,
        accept_threshold=0.95,
    )


class TieredMatcherTests(unittest.TestCase):
    def test_exact_skips_encoder(self):
        def forbidden(texts):
            raise AssertionError("No embeddings needed")

        result = match(
            [fact("A")],
            [fact("A", fid="F1")],
            forbidden,
        )
        self.assertEqual(result["matches"][0]["method"], "exact")
        self.assertEqual(result["review_queue"], [])

    def test_high_similarity_still_reviewed(self):
        result = match(
            [fact("Chest pain absent.")],
            [fact("Chest pain is absent.", fid="F1")],
        )
        self.assertEqual(
            result["matches"][0]["method"], "embedding_advisory"
        )
        self.assertEqual(
            result["review_queue"][0]["reason"],
            "high_similarity_unvalidated",
        )

    def test_wrong_type_never_matches(self):
        result = match(
            [fact("A", "procedure")],
            [fact("B", fid="F1")],
        )
        self.assertEqual(result["matches"], [])

    def test_uncertain_and_low_band(self):
        cases = [
            ([0.9, (1 - 0.9**2)**0.5], "uncertain_similarity"),
            ([0, 1], "unmatched_prediction"),
        ]
        for vector, reason in cases:
            def encoder(texts):
                return [[1, 0], vector]

            result = match(
                [fact("A")],
                [fact("B", fid="F1")],
                encoder,
            )
            self.assertEqual(result["matches"], [])
            self.assertEqual(
                result["review_queue"][0]["reason"], reason
            )

    def test_number_and_negation_block_high_proposals(self):
        cases = [
            ("Age 19.", "Age 91."),
            ("Chest pain.", "No chest pain."),
        ]
        for predicted, gold in cases:
            result = match(
                [fact(predicted)],
                [fact(gold, fid="F1")],
            )
            self.assertEqual(result["matches"], [])
            self.assertIsNotNone(
                result["review_queue"][0]
                ["candidates"][0]["detail_flag"]
            )

    def test_competing_predictions_not_double_matched(self):
        result = match(
            [fact("A"), fact("B")],
            [fact("C", fid="F1")],
        )
        self.assertEqual(result["matches"], [])
        self.assertTrue(
            all(
                item["reason"] == "ambiguous_candidates"
                for item in result["review_queue"]
            )
        )

    def test_competing_gold_requires_review(self):
        result = match(
            [fact("A")],
            [fact("B", fid="F1"), fact("C", fid="F2")],
        )
        self.assertEqual(result["matches"], [])

    def test_invalid_embeddings_and_thresholds(self):
        cases = [
            ([], []),
            ([0, 0], [1, 0]),
            ([1], [1, 0]),
            ([float("nan")], [1]),
        ]
        for left, right in cases:
            with self.assertRaises(ValueError):
                cosine(left, right)

        with self.assertRaises(ValueError):
            match_facts(
                [], [], encode,
                review_threshold=0.95,
                accept_threshold=0.8,
            )

        with self.assertRaises(ValueError):
            match(
                [fact("A")],
                [fact("B", fid="F1")],
                lambda texts: [],
            )

    def test_wrapper_preserves_exact_and_failed_outputs(self):
        gold = [
            {
                "example_id": str(i),
                "source_text": "Source sentence.",
                "facts": [fact("Chest pain absent.", fid="F1")],
            }
            for i in range(3)
        ]
        predictions = [
            {
                "example_id": "0",
                "raw_prediction": json.dumps([
                    fact("Chest pain is absent.")
                ]),
            },
            {"example_id": "1", "raw_prediction": "bad"},
        ]
        result = score_tiered_predictions(
            gold,
            predictions,
            encode,
            review_threshold=0.8,
            accept_threshold=0.95,
            embedding_model="test-vectors",
        )
        self.assertEqual(result["exact_baseline"]["exact_f1"], 0)
        self.assertEqual(
            result["semantic_advisory"]["f1_valid_only"], 1
        )
        self.assertEqual(
            result["semantic_advisory"]["f1_all_outputs"], 0.5
        )
        self.assertEqual(
            result["review_queue"][0]["example_id"], "0"
        )
        self.assertEqual(
            result["review_queue"][0]["source_text"],
            "Source sentence.",
        )


if __name__ == "__main__":
    unittest.main()
