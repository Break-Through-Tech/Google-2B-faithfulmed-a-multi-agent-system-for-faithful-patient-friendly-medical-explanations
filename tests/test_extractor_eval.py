"""Tests for the offline Extractor evaluator and provisional semantic matcher."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from evals.extractor_eval import (
    generate_predictions, load_gold, load_predictions, parse_prediction,
    save_predictions, score_predictions, generate_semantic_reviews, semantic_review_queue,
    _review_flags,
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
        {"example_id": "scratch-1", "raw_prediction": json.dumps([
            dict(f, fact_id=f"P-{i:03d}") for i, f in enumerate(facts, 1)
        ])}
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
        r = semantic_review_queue(
            example(), [dict(atom("Patient reports no chest pain."), fact_id="CUSTOM-7")],
            lambda _: [1., 0.]
        )
        self.assertEqual(r["semantic_matches"][0]["prediction_id"], "CUSTOM-7")

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
        self.assertEqual(len(result["semantic_matches"]), 1)
        self.assertEqual(result["semantic_matches"][0]["prediction_id"], "P-002")
        self.assertEqual([r["review_type"] for r in result["review_queue"]],
                         ["unmatched_prediction"])
        self.assertTrue(all(r["human_verdict"] is None for r in result["review_queue"]))
        self.assertAlmostEqual(result["provisional_semantic_metrics"]["precision"], 2/3)
        self.assertEqual(result["provisional_semantic_metrics"]["recall"], 1)

    def test_semantic_wrong_type_never_pairs(self):
        e = example()
        r = semantic_review_queue(e, [atom(kind="procedure")],
                                  lambda _: self.fail("Should not embed mismatched types"))
        self.assertEqual({x["review_type"] for x in r["review_queue"]},
                         {"unmatched_gold", "unmatched_prediction"})

    def test_semantic_negation_flag(self):
        r = semantic_review_queue(example(), [atom("The patient has chest pain.")],
                                  lambda _: [1., 0.])
        self.assertEqual(r["semantic_matches"], [])
        self.assertIn("negation_mismatch", r["review_queue"][0]["flags"])

    def test_semantic_number_mismatch_not_auto_accepted(self):
        gold = example([atom("The patient is 91 years old.", "demographics")])
        predicted = [atom("The patient is 19 years old.", "demographics")]
        r = semantic_review_queue(gold, predicted, lambda _: [1., 0.])
        self.assertEqual(r["semantic_matches"], [])
        self.assertIn("number_mismatch", r["review_queue"][0]["flags"])

    def test_borderline_semantic_candidate_stays_in_review(self):
        gold = example()
        predicted = [atom("Patient reports no chest pain.")]
        vectors = {gold["facts"][0]["fact_text"]: [1., 0.],
                   predicted[0]["fact_text"]: [0.8, 0.6]}
        r = semantic_review_queue(gold, predicted, lambda t: vectors[t])
        self.assertEqual(r["semantic_matches"], [])
        self.assertEqual(r["review_queue"][0]["review_type"], "semantic_candidate")

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
        self.assertEqual(result["status"], "provisional_auto_matches_not_human_validated")


GOLD_MEDS = {
    "zocor": "Zocor was prescribed at 20 mg, one tablet p.o. at bedtime.",
    "prevacid": "Prevacid was prescribed at 30 mg p.o. once a day.",
    "cipro": "The patient was instructed to continue Cipro 250 mg, one tablet p.o. b.i.d., for an additional two days for UTI.",
    "docusate": "Docusate sodium was prescribed at 100 mg, one cap p.o. b.i.d.",
}


class ReviewFlagTests(unittest.TestCase):
    def flags(self, key, prediction, kind="medication"):
        return _review_flags(GOLD_MEDS[key], prediction, kind)

    def test_equivalent_abbreviation_and_format_variants_not_flagged(self):
        cases = [
            ("zocor", "The patient is prescribed Zocor 20 mg, one tablet orally at bedtime."),
            ("prevacid", "The patient is prescribed Prevacid 30 mg orally once a day."),
            ("docusate", "The patient is prescribed docusate sodium 100 mg, one capsule orally twice a day."),
            ("docusate", "Docusate sodium 100mg, 1 cap by mouth b.i.d."),
        ]
        for key, pred in cases:
            with self.subTest(pred=pred):
                self.assertEqual(self.flags(key, pred), [])

    def test_clinically_wrong_medication_changes_are_flagged(self):
        cases = [
            ("zocor", "Zocor 40 mg, one tablet orally at bedtime.", "number_mismatch"),
            ("zocor", "Zocor 20 mcg, one tablet orally at bedtime.", "number_mismatch"),
            ("zocor", "Zocor 20 mg, one tablet orally in the morning.", "time_of_day_mismatch"),
            ("prevacid", "Prilosec 30 mg orally once a day.", "term_mismatch"),
            ("prevacid", "Prevacid 30 mg orally twice a day.", "frequency_mismatch"),
            ("prevacid", "Prevacid 30 mg intravenously once a day.", "route_mismatch"),
            ("cipro", "The patient is to stop Cipro 250 mg, one tablet orally twice a day for an additional two days for UTI.", "status_mismatch"),
            ("cipro", "The patient is to take Cipro 250 mg, one tablet orally twice a day for an additional two days.", "status_mismatch"),
            ("cipro", "The patient is to continue Cipro 250 mg, one tablet orally twice a day for an additional three days for UTI.", "number_mismatch"),
            ("cipro", "The patient is to continue Cipro 500mg, one tablet orally twice a day for an additional two days for UTI.", "number_mismatch"),
            ("docusate", "Docusate sodium 100 mg, two capsules orally twice a day.", "number_mismatch"),
        ]
        for key, pred, expected in cases:
            with self.subTest(pred=pred):
                self.assertIn(expected, self.flags(key, pred))

    def test_negation_and_contractions(self):
        self.assertIn("negation_mismatch", _review_flags("The patient denies fever.", "The patient has fever.", "diagnosis"))
        self.assertIn("negation_mismatch", _review_flags("The patient didn't have fever.", "The patient has fever.", "diagnosis"))
        self.assertEqual(_review_flags("The patient denies chest pain.", "No chest pain was reported.", "diagnosis"), [])

    def test_known_limit_antonym_is_not_flagged(self):
        # Documented gap: flags cannot catch every opposite word. Auto-accepted
        # matches therefore still need a sampled human audit.
        self.assertEqual(_review_flags("The patient was stable at discharge.",
                                       "The patient was unstable at discharge.", "diagnosis"), [])


class MergeSignalTests(unittest.TestCase):
    def test_merged_prediction_and_split_gold_share_evidence(self):
        shared = "PT, OT, speech therapist to evaluate and treat at Siskin Rehab Hospital."
        def fact(fid, text):
            return {"fact_id": fid, "fact_text": text, "fact_type": "instruction", "source_evidence": [shared]}
        e = {"example_id": "m", "source_text": shared,
             "facts": [fact("F5", "PT was to evaluate and treat the patient."),
                       fact("F6", "OT was to evaluate and treat the patient."),
                       fact("F7", "A speech therapist was to evaluate and treat the patient.")]}
        merged = dict(fact("F-005", "PT, OT, and speech therapy are to evaluate and treat the patient."),
                      fact_type="procedure")
        r = semantic_review_queue(e, [merged], lambda _: self.fail("different types must not embed"))
        by_type = {}
        for item in r["review_queue"]:
            by_type.setdefault(item["review_type"], []).append(item)
        self.assertEqual([i["shares_evidence_with_predictions"] for i in by_type["unmatched_gold"]],
                         [["F-005"]] * 3)
        self.assertEqual(by_type["unmatched_prediction"][0]["shares_evidence_with_gold"],
                         ["F5", "F6", "F7"])

    def test_semantic_items_record_fact_type(self):
        r = semantic_review_queue(example(), [atom("The patient has chest pain.")], lambda _: [1., 0.])
        self.assertEqual(r["review_queue"][0]["fact_type"], "diagnosis")

    def test_medication_dose_change_is_never_auto_accepted(self):
        gold = example([atom("Zocor 20 mg at bedtime.", "medication")])
        gold["source_text"] += " Zocor 20 mg at bedtime."
        pred = [atom("Zocor 40 mg at bedtime.", "medication")]
        r = semantic_review_queue(gold, pred, lambda _: [1., 0.])
        self.assertEqual(r["semantic_matches"], [])
        self.assertIn("number_mismatch", r["review_queue"][0]["flags"])


if __name__ == "__main__":
    unittest.main()
