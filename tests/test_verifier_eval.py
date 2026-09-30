"""Offline fixtures only; these are not MedAESQA project results."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.verifier_eval import (
    MedAESQAExample,
    VerifierPrediction,
    load_medaesqa_examples,
    load_predictions,
    parse_answer_prediction,
    parse_citation_prediction,
    save_predictions,
    score_predictions,
)


class VerifierEvaluationTests(unittest.TestCase):
    def test_answer_labels_use_yes_no_as_acceptability(self) -> None:
        examples = [
            MedAESQAExample("answer", "q1", "M1", "question", "answer", True, gold_label_raw="yes"),
            MedAESQAExample("answer", "q1", "M2", "question", "answer", False, gold_label_raw="no"),
        ]
        result = score_predictions(examples, [
            VerifierPrediction(examples[0].example_id, True),
            VerifierPrediction(examples[1].example_id, False),
        ])
        self.assertEqual((result.evaluation_level, result.accuracy, result.f1), ("answer", 1.0, 1.0))
        self.assertEqual(result.positive_class, "MedAESQA answer label 'yes' (acceptable)")

    def test_citation_labels_only_score_supporting_and_contradicting(self) -> None:
        supporting = MedAESQAExample("citation", "q1", "M1", "q", "claim", True, "s1", "1", ("evidence",), "supporting")
        contradicting = MedAESQAExample("citation", "q1", "M1", "q", "claim", False, "s2", "2", ("evidence",), "contradicting")
        result = score_predictions([supporting, contradicting], [
            VerifierPrediction(supporting.example_id, True), VerifierPrediction(contradicting.example_id, False),
        ])
        self.assertEqual((result.evaluation_level, result.true_positives, result.true_negatives), ("citation", 1, 1))

    def test_neutral_is_unavailable_not_a_negative_label(self) -> None:
        neutral = MedAESQAExample("citation", "q1", "M1", "q", "claim", None, "s1", "1", gold_label_raw="neutral", unavailable_reason="not evaluable")
        result = score_predictions([neutral], [VerifierPrediction(neutral.example_id, False)])
        self.assertEqual(result.status, "not_evaluated")
        self.assertEqual(result.unavailable_gold_ids, (neutral.example_id,))
        self.assertEqual(result.false_negatives, 0)

    def test_loader_preserves_real_vocab_and_evidence_input(self) -> None:
        fixture = [{"question_id": "q1", "question": "What?", "machine_generated_answers": {
            "M1": {"answer": "A", "is_answer_accurate": "yes", "answer_sentences": [
                {"answer_sentence_id": "s1", "answer_sentence": "A claim", "answer_sentence_relevance": "required",
                 "citation_assessment": [
                    {"cited_pmid": "10", "evidence_relation": "supporting", "evidence_support": "supports claim"},
                    {"cited_pmid": "11", "evidence_relation": "neutral", "evidence_support": "topic only"},
                ]}
            ]}
        }}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            answers = load_medaesqa_examples(path, level="answer")
            citations = load_medaesqa_examples(path, level="citation")
        self.assertEqual((answers[0].gold_label, answers[0].question), (True, "What?"))
        self.assertEqual(citations[0].evidence_snippets, ("supports claim",))
        self.assertIsNone(citations[1].gold_label)
        self.assertEqual(citations[1].gold_label_raw, "neutral")

    def test_malformed_output_is_separate_from_wrong_judgment(self) -> None:
        example = MedAESQAExample("answer", "q1", "M1", "q", "a", True)
        malformed = parse_answer_prediction(example.example_id, "{}")
        result = score_predictions([example], [malformed])
        self.assertEqual((result.status, result.malformed_ids, result.false_negatives), ("not_evaluated", (example.example_id,), 0))

    def test_answer_parser_accepts_markdown_fenced_json(self) -> None:
        prediction = parse_answer_prediction("q1:M1", '```json\n{"faithful": false}\n```')
        self.assertEqual((prediction.predicted_label, prediction.malformed_error), (False, None))

    def test_citation_parser_rejects_unknown_verdict(self) -> None:
        prediction = parse_citation_prediction("q:M:s:1", '{"citation_verdict": "neutral"}')
        self.assertIsNotNone(prediction.malformed_error)

    def test_saved_predictions_rescore_without_model_call(self) -> None:
        original = [VerifierPrediction("q1:M1", True)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            save_predictions(path, original)
            self.assertEqual(load_predictions(path), original)


if __name__ == "__main__":
    unittest.main()
