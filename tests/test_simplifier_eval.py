"""Offline tests for Simplifier readability evaluation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from evals.simplifier_eval import (
    SimplifierExample,
    evaluate_example,
    score_examples,
    summarize_examples,
)


SCORES = {
    "technical source": {
        "flesch_kincaid_grade": 12.0,
        "smog_index": 14.0,
        "jargon_density": 0.50,
        "word_count": 100,
    },
    "plain rewrite": {
        "flesch_kincaid_grade": 6.0,
        "smog_index": 7.0,
        "jargon_density": 0.20,
        "word_count": 80,
    },
}


class SimplifierEvaluationTests(unittest.TestCase):
    @patch("evals.simplifier_eval.readability_scores", side_effect=lambda text: SCORES[text])
    def test_evaluate_example_reports_reductions_and_target(self, _scores) -> None:
        result = evaluate_example(SimplifierExample("one", "technical source", "plain rewrite"))

        self.assertEqual(result.fk_grade_reduction, 6.0)
        self.assertEqual(result.smog_reduction, 7.0)
        self.assertEqual(result.jargon_reduction, 0.30)
        self.assertTrue(result.meets_readability_target)

    def test_empty_source_or_output_is_rejected(self) -> None:
        for example in (
            SimplifierExample("empty-source", "", "plain rewrite"),
            SimplifierExample("empty-output", "technical source", ""),
        ):
            with self.subTest(example_id=example.example_id):
                with self.assertRaises(ValueError):
                    evaluate_example(example)

    @patch("evals.simplifier_eval.readability_scores", side_effect=lambda text: SCORES[text])
    def test_score_examples_skips_malformed_output(self, _scores) -> None:
        results = score_examples([
            SimplifierExample("valid", "technical source", "plain rewrite"),
            SimplifierExample("invalid", "technical source", ""),
        ])

        self.assertEqual([result.example_id for result in results], ["valid"])

    @patch("evals.simplifier_eval.readability_scores", side_effect=lambda text: SCORES[text])
    def test_summary_reports_valid_and_malformed_examples(self, _scores) -> None:
        summary = summarize_examples([
            SimplifierExample("valid", "technical source", "plain rewrite"),
            SimplifierExample("invalid", "technical source", ""),
        ])

        self.assertEqual(summary.status, "partial")
        self.assertEqual(summary.n, 1)
        self.assertEqual(summary.mean_fk_grade_reduction, 6.0)
        self.assertEqual(summary.readability_target_rate, 1.0)
        self.assertEqual(summary.malformed_ids, ("invalid",))

    def test_empty_summary_has_no_fake_scores(self) -> None:
        summary = summarize_examples([])

        self.assertEqual(summary.status, "not_evaluated")
        self.assertEqual(summary.n, 0)
        self.assertIsNone(summary.mean_simplified_fk_grade)
        self.assertIsNone(summary.readability_target_rate)


if __name__ == "__main__":
    unittest.main()
