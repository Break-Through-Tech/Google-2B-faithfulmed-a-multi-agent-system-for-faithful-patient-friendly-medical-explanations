"""
FaithfulMed — Refiner evaluation harness tests.

Offline and synthetic only: no ADK import, no Gemini call.
"""

from __future__ import annotations
import unittest

from pathlib import Path
import sys

EVALS = Path(__file__).resolve().parent.parent / "evals"
sys.path.append(str(EVALS))

from refiner_eval import (
    parse_verdict,
    is_unchanged,
    minimal_edit_ratio,
    unsupported_removed_rate,
    omissions_restored_rate,
    evaluate_single,
    evaluate_multiple,
    summarize_results,
    compare_texts,
    contract_pass_rate,
    find_failures,
)


class RefinerEvaluationTests(unittest.TestCase):

    # ---- parse_verdict ----

    def test_parse_verdict_from_dict(self) -> None:
        verdict = {
            "faithful": False,
            "unsupported_claims": ["the patient has cancer"],
            "omissions": ["blood pressure was 120/88"],
            "reading_level_ok": True,
        }
        parsed = parse_verdict(verdict)
        self.assertEqual(parsed["faithful"], False)
        self.assertEqual(parsed["unsupported_claims"], ["the patient has cancer"])
        self.assertEqual(parsed["omissions"], ["blood pressure was 120/88"])

    def test_parse_verdict_from_json_string(self) -> None:
        parsed = parse_verdict('{"faithful": true, "unsupported_claims": [], "omissions": []}')
        self.assertTrue(parsed["faithful"])
        self.assertEqual(parsed["unsupported_claims"], [])

    def test_parse_verdict_malformed_is_safe(self) -> None:
        parsed = parse_verdict("not json at all")
        self.assertIsNone(parsed["faithful"])
        self.assertEqual(parsed["unsupported_claims"], [])
        self.assertEqual(parsed["omissions"], [])

    # ---- primitives ----

    def test_is_unchanged(self) -> None:
        self.assertTrue(is_unchanged("same text", "same text"))
        self.assertTrue(is_unchanged("trimmed  ", "  trimmed"))
        self.assertFalse(is_unchanged("before", "after"))

    def test_minimal_edit_ratio_identical_is_one(self) -> None:
        self.assertEqual(minimal_edit_ratio("identical", "identical"), 1.0)

    def test_minimal_edit_ratio_targeted_edit_stays_high(self) -> None:
        draft = "You have high blood pressure and diabetes."
        refined = "You have high blood pressure."
        self.assertGreater(minimal_edit_ratio(draft, refined), 0.6)

    # ---- removal / restoration ----

    def test_unsupported_removed_rate(self) -> None:
        draft = "You have high blood pressure. You also have cancer."
        refined = "You have high blood pressure."
        rate = unsupported_removed_rate(draft, refined, ["you also have cancer"])
        self.assertEqual(rate, 1.0)

    def test_unsupported_removed_rate_none_when_no_flags(self) -> None:
        self.assertIsNone(unsupported_removed_rate("a", "a", []))

    def test_unsupported_removed_rate_partial(self) -> None:
        refined = "You still have cancer here."
        rate = unsupported_removed_rate("draft", refined, ["you still have cancer", "you have gout"])
        self.assertEqual(rate, 0.5)

    def test_omissions_restored_rate(self) -> None:
        refined = "Your blood pressure was 120/88 and you should follow up in two weeks."
        rate = omissions_restored_rate(refined, ["blood pressure was 120/88"])
        self.assertEqual(rate, 1.0)

    def test_omissions_restored_rate_missing(self) -> None:
        rate = omissions_restored_rate("Nothing relevant here.", ["blood pressure was 120/88"])
        self.assertEqual(rate, 0.0)

    # ---- evaluate_single: the three contract cases ----

    def test_evaluate_single_good_refine(self) -> None:
        draft = "You have high blood pressure. You also have cancer."
        refined = "You have high blood pressure. Your heart rate was 104."
        verdict = {
            "faithful": False,
            "unsupported_claims": ["you also have cancer"],
            "omissions": ["heart rate was 104"],
        }
        res = evaluate_single(draft, refined, verdict)
        self.assertEqual(res["unsupported_removed_rate"], 1.0)
        self.assertEqual(res["omissions_restored_rate"], 1.0)
        self.assertTrue(res["contract_pass"])
        self.assertFalse(res["is_noop_case"])

    def test_evaluate_single_noop_respected(self) -> None:
        draft = "You have high blood pressure."
        verdict = {"faithful": True, "unsupported_claims": [], "omissions": []}
        res = evaluate_single(draft, draft, verdict)
        self.assertTrue(res["is_noop_case"])
        self.assertTrue(res["noop_respected"])
        self.assertTrue(res["contract_pass"])

    def test_evaluate_single_noop_violated(self) -> None:
        draft = "You have high blood pressure."
        refined = "You have high blood pressure and should exercise more."
        verdict = {"faithful": True, "unsupported_claims": [], "omissions": []}
        res = evaluate_single(draft, refined, verdict)
        self.assertTrue(res["is_noop_case"])
        self.assertFalse(res["noop_respected"])
        self.assertFalse(res["contract_pass"])

    def test_evaluate_single_left_unsupported_fails(self) -> None:
        draft = "You have cancer."
        refined = "You have cancer."  # failed to remove the flagged claim
        verdict = {"faithful": False, "unsupported_claims": ["you have cancer"], "omissions": []}
        res = evaluate_single(draft, refined, verdict)
        self.assertEqual(res["unsupported_removed_rate"], 0.0)
        self.assertFalse(res["contract_pass"])

    # ---- multiple / summarize / failures ----

    def test_evaluate_multiple_and_summarize(self) -> None:
        examples = [
            {
                "example_id": "ex1",
                "draft": "You have high blood pressure. You also have cancer.",
                "refined": "You have high blood pressure.",
                "verdict": {"faithful": False, "unsupported_claims": ["you also have cancer"], "omissions": []},
            },
            {
                "example_id": "ex2",
                "draft": "You should rest.",
                "refined": "You should rest.",
                "verdict": {"faithful": True, "unsupported_claims": [], "omissions": []},
            },
        ]
        results = evaluate_multiple(examples)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["example_id"], "ex1")

        summary = summarize_results(results)
        self.assertEqual(summary["num_results"], 2)
        self.assertEqual(summary["percent_contract_pass"], 100.0)
        self.assertIn("mean_minimal_edit_ratio", summary)

    def test_summarize_results_empty(self) -> None:
        summary = summarize_results([])
        self.assertEqual(summary["num_results"], 0)
        self.assertIsNone(summary["mean_minimal_edit_ratio"])
        self.assertIsNone(summary["percent_contract_pass"])

    def test_compare_texts(self) -> None:
        draft = "You have high blood pressure. You also have cancer."
        refined = "You have high blood pressure. Your heart rate was 104."
        result = compare_texts(draft, refined)
        self.assertEqual(result["before"], draft)
        self.assertEqual(result["after"], refined)
        self.assertTrue(result["changed"])
        self.assertIn("You also have cancer.", result["removed_sentences"])
        self.assertIn("Your heart rate was 104.", result["added_sentences"])

    def test_contract_pass_rate_empty(self) -> None:
        self.assertEqual(contract_pass_rate([]), -1.0)

    def test_find_failures(self) -> None:
        examples = [
            {
                "example_id": "bad",
                "draft": "You have cancer.",
                "refined": "You have cancer.",
                "verdict": {"faithful": False, "unsupported_claims": ["you have cancer"], "omissions": []},
            },
        ]
        results = evaluate_multiple(examples)
        failures = find_failures(results)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["example_id"], "bad")
        self.assertIn("left unsupported claims", failures[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
