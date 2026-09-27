"""
FaithfulMed — Raedability evaluation harness tests
"""

from __future__ import annotations
import unittest

from pathlib import Path
import sys

NOTEBOOKS = Path(__file__).resolve().parent.parent / "evals"
sys.path.append(str(NOTEBOOKS))

from readability_eval import (
    is_refusal,
    evaluate_single,
    evaluate_multiple,
    summarize_results,
    compare_scores,
    readability_target_rate,
    find_failures,
    refusal_rate
)


class ReadabilityEvaluationTests(unittest.TestCase):

    def test_is_refusal_detects_refusal(self) -> None:
        self.assertTrue(is_refusal("I cannot help with that request."))
        self.assertTrue(is_refusal("I'm unable to answer that."))
        self.assertTrue(is_refusal("I cannot provide that information."))

    def test_is_refusal_is_case_insensitive(self) -> None:
        self.assertTrue(is_refusal("I CANNOT HELP with that."))
        self.assertTrue(is_refusal("I Can'T hELp with that."))

    def test_is_refusal_non_refusal(self) -> None:
        self.assertFalse(is_refusal("The patient should drink plenty of water."))
        self.assertFalse(is_refusal("The patient was diagnosed with diabetes."))

    def assert_has_readability_scores(self, result) -> None:
        self.assertIn("flesch_kincaid_grade", result)
        self.assertIn("smog_index", result)
        self.assertIn("jargon_density", result)
        self.assertIn("word_count", result)
        self.assertIn("meets_target", result)
        self.assertIn("is_refusal", result)

    def test_evaluate_single_contains_expected(self) -> None:
        text = "The patient should drink plenty of water."

        result = evaluate_single(text)

        self.assert_has_readability_scores(result)

    def test_evaluate_single_detects_refusal(self) -> None:
        result = evaluate_single("I cannot provide that information.")

        self.assertTrue(result["is_refusal"])

    def test_evaluate_multiple_texts(self) -> None:
        texts = [
            "The patient should drink water.",
            "I cannot provide that information.",
            "The patient is experiencing a cold.",
        ]

        results = evaluate_multiple(texts)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["text"], texts[0])
        self.assertEqual(results[1]["text"], texts[1])
        self.assertEqual(results[2]["text"], texts[2])
        self.assertFalse(results[0]["is_refusal"])
        self.assertTrue(results[1]["is_refusal"])
        self.assertFalse(results[2]["is_refusal"])

        for result in results:
            self.assert_has_readability_scores(result)

    def test_summarize_results(self) -> None:
        texts = [
            "The patient should drink water.",
            "The patient should rest.",
        ]

        results = evaluate_multiple(texts)
        summary = summarize_results(results)

        self.assertEqual(summary["num_results"], 2)
        self.assertIn("mean_fk_grade", summary)
        self.assertIn("mean_smog_index", summary)
        self.assertIn("mean_jargon_density", summary)
        self.assertIn("mean_word_count", summary)
        self.assertIn("percent_successful", summary)
        self.assertIn("percent_refusal", summary)

    def test_summarize_results_empty(self) -> None:
        summary = summarize_results([])
        
        self.assertEqual(summary["num_results"], 0)
        self.assertEqual(summary["mean_fk_grade"], None)
        self.assertEqual(summary["mean_smog_index"], None)
        self.assertEqual(summary["mean_jargon_density"], None)
        self.assertEqual(summary["mean_word_count"], None)
        self.assertEqual(summary["percent_successful"], None)
        self.assertEqual(summary["percent_refusal"], None)


    def test_compare_scores(self) -> None:
        before = (
            "The patient should maintain adequate hydration "
            "to facilitate optimal physiological function."
        )
        after = "The patient should drink enough water."

        result = compare_scores(before, after)

        self.assertEqual(result["before"], before)
        self.assertEqual(result["after"], after)
        self.assertIn("fk_diff", result)
        self.assertIn("smog_index_diff", result)
        self.assertIn("jargon_density_diff", result)
        self.assertIn("word_count_diff", result)

    def test_readability_target_rate(self) -> None:
        texts = [
            "The patient should drink water.",
            "The patient should rest.",
        ]

        rate = readability_target_rate(texts)

        self.assertGreaterEqual(rate, 0)
        self.assertLessEqual(rate, 100)

    def test_readability_target_rate_empty(self) -> None:
        rate = readability_target_rate([])

        self.assertEqual(rate, -1.0)

    def test_find_failures(self) -> None:
        texts = [
            "The patient should drink water.",
            (
                "The patient should maintain adequate hydration "
                "to facilitate optimal physiological function."
            ),
        ]

        failures = find_failures(texts)

        for failure in failures:
            self.assertIn("text", failure)
            self.assertIn("score", failure)

    def test_find_failures_fail(self) -> None:
        text = [
            "The pathophysiological mechanism of myocardial infarction involves the acute occlusion"
        ]
        failure = find_failures(text)

        self.assertEqual(failure[0]["text"], text[0])
        self.assertIn("score", failure[0])

    def test_refusal_rate(self) -> None:
        texts = [
            "The patient should drink water.",
            "I cannot provide that information.",
        ]

        rate = refusal_rate(texts)

        self.assertEqual(rate, 50.0)

    def test_refusal_rate_empty(self) -> None:
        rate = refusal_rate([])
        
        self.assertEqual(rate, -1.0)


if __name__ == "__main__":
    unittest.main()