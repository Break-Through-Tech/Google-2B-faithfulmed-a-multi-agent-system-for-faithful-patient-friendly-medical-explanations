from __future__ import annotations

import unittest

from evals.independent_review import supplementary_agreement


def review_item(label: str, *, accepted: list[str] | None = None, scorable: bool = True) -> dict:
    return {"primary_label": label, "acceptable_labels": accepted or [label], "ambiguity_flag": len(accepted or [label]) > 1,
            "needs_human_review": not scorable, "rationale": "test", "alternative_rationales": {}, "scorable": scorable}


class IndependentReviewTests(unittest.TestCase):
    def test_accepted_label_match_is_not_limited_to_primary(self) -> None:
        records = [{"question_id": "1", "method_id": "M1", "answer_accuracy": review_item("yes", accepted=["yes", "no"]), "sentences": []}]
        prediction = {"1:M1": {"answer_accuracy": {"prediction": "no"}, "sentences": []}}
        result = supplementary_agreement(records, prediction, "answer_accuracy")
        self.assertEqual(result["exact_matches"], 0)
        self.assertEqual(result["accepted_matches"], 1)

    def test_unscorable_item_is_skipped_not_incorrect(self) -> None:
        records = [{"question_id": "1", "method_id": "M1", "answer_accuracy": review_item("yes"), "sentences": [{"answer_sentence_id": "s1", "sentence_relevance": review_item("required"), "citations": [{"citation_id": "s1:0", "evidence_relation": review_item("unscorable", scorable=False)}]}]}]
        prediction = {"1:M1": {"answer_accuracy": {"prediction": "yes"}, "sentences": [{"answer_sentence_id": "s1", "relevance_prediction": "required", "citation_assessments": []}]}}
        result = supplementary_agreement(records, prediction, "evidence_relation")
        self.assertEqual(result["scored"], 0)
        self.assertEqual(result["skipped_unscorable"], 1)
        self.assertEqual(result["incorrect"], 0)
        self.assertEqual(result["missing_prediction"], 1)

    def test_supplementary_counts_mirror_benchmark_core_columns(self) -> None:
        records = [{"question_id": "1", "method_id": "M1", "answer_accuracy": review_item("yes"), "sentences": []}]
        prediction = {"1:M1": {"answer_accuracy": {"prediction": "no"}, "sentences": []}}
        result = supplementary_agreement(records, prediction, "answer_accuracy")
        self.assertEqual((result["scored"], result["exact_matches"], result["incorrect"]), (1, 0, 1))
        self.assertEqual(result["error_rate"], 1.0)
        self.assertEqual(result["skipped_unscorable"], 0)

    def test_originally_unscored_item_has_no_supplementary_label_or_match(self) -> None:
        records = [{"question_id": "1", "method_id": "M1", "answer_accuracy": review_item("yes"), "sentences": []}]
        prediction = {"1:M1": {"answer_accuracy": {"prediction": "no"}, "sentences": []}}
        result = supplementary_agreement(
            records, prediction, "answer_accuracy",
            {("1:M1", "1:M1"): "Original human answer-accuracy label is missing or unscored."},
        )
        self.assertEqual((result["scored"], result["incorrect"], result["accepted_matches"]), (0, 0, 0))
        self.assertEqual(result["skipped_unscorable"], 1)
        self.assertIsNone(result["details"][0]["primary_label"])
        self.assertEqual(result["details"][0]["acceptable_labels"], [])

    def test_supplementary_comparison_does_not_mutate_prediction(self) -> None:
        records = [{"question_id": "1", "method_id": "M1", "answer_accuracy": review_item("yes"), "sentences": []}]
        prediction = {"1:M1": {"answer_accuracy": {"prediction": "yes"}, "sentences": []}}
        original = {"1:M1": {"answer_accuracy": {"prediction": "yes"}, "sentences": []}}
        supplementary_agreement(records, prediction, "answer_accuracy")
        self.assertEqual(prediction, original)


if __name__ == "__main__":
    unittest.main()
