from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.verifier_eval import (
    LabeledVerifierExample,
    VerifierPrediction,
    load_predictions,
    parse_prediction,
    save_predictions,
    score_predictions,
)


def example(example_id: str, gold: bool | None) -> LabeledVerifierExample:
    return LabeledVerifierExample(example_id, "synthetic source", "synthetic claim", gold)


def prediction(example_id: str, value: bool) -> VerifierPrediction:
    return VerifierPrediction(example_id, value)


class VerifierEvaluationTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        result = score_predictions(
            [example("positive", True), example("negative", False)],
            [prediction("positive", True), prediction("negative", False)],
        )
        self.assertEqual((result.true_positives, result.true_negatives), (1, 1))
        self.assertEqual((result.precision, result.recall, result.f1), (1.0, 1.0, 1.0))

    def test_one_false_positive(self) -> None:
        result = score_predictions([example("fp", False)], [prediction("fp", True)])
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_positive_ids, ("fp",))

    def test_one_false_negative(self) -> None:
        result = score_predictions([example("fn", True)], [prediction("fn", False)])
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.false_negative_ids, ("fn",))

    def test_mixed_confusion_counts(self) -> None:
        examples = [example("tp", True), example("fp", False), example("tn", False), example("fn", True)]
        predictions = [prediction("tp", True), prediction("fp", True), prediction("tn", False), prediction("fn", False)]
        result = score_predictions(examples, predictions)
        self.assertEqual(
            (result.true_positives, result.false_positives, result.true_negatives, result.false_negatives),
            (1, 1, 1, 1),
        )
        self.assertEqual((result.precision, result.recall, result.f1), (0.5, 0.5, 0.5))

    def test_mismatched_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            score_predictions([example("gold", True)], [prediction("different", True)])

    def test_no_predicted_positives(self) -> None:
        result = score_predictions([example("positive", True)], [prediction("positive", False)])
        self.assertIsNone(result.precision)
        self.assertEqual(result.recall, 0.0)
        self.assertIsNone(result.f1)

    def test_no_actual_positives(self) -> None:
        result = score_predictions([example("negative", False)], [prediction("negative", True)])
        self.assertEqual(result.precision, 0.0)
        self.assertIsNone(result.recall)
        self.assertIsNone(result.f1)

    def test_malformed_structured_prediction_is_separate(self) -> None:
        malformed = parse_prediction("bad", '{"faithful": true}')
        result = score_predictions([example("bad", True)], [malformed])
        self.assertEqual(result.status, "not_evaluated")
        self.assertEqual(result.malformed_ids, ("bad",))
        self.assertEqual(result.false_negatives, 0)

    def test_missing_human_label_is_not_scored(self) -> None:
        result = score_predictions([example("missing", None)], [prediction("missing", True)])
        self.assertEqual(result.status, "not_evaluated")
        self.assertEqual(result.missing_gold_ids, ("missing",))
        self.assertEqual(result.false_positives, 0)

    def test_saved_predictions_can_be_scored_without_a_model_call(self) -> None:
        original = [prediction("one", True), prediction("two", False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            save_predictions(path, original)
            loaded = load_predictions(path)
        self.assertEqual(loaded, original)


if __name__ == "__main__":
    unittest.main()
