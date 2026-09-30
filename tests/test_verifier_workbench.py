from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.verifier_workbench import (WorkbenchAnswer, compare, load_workbench_answers,
                                      parse_workbench_prediction, prompt_for, select_representative_answers)


def answer() -> WorkbenchAnswer:
    return WorkbenchAnswer("1", "M1", "Question?", "Candidate.", "yes", ({
        "answer_sentence_id": "s1", "answer_sentence": "Candidate.", "relevance_gold": "required",
        "citations": [{"citation_id": "s1:0", "cited_pmid": "10", "evidence_support": "Excerpt.", "evidence_relation_gold": "supporting"},
                      {"citation_id": "s1:1", "cited_pmid": "11", "evidence_support": None, "evidence_relation_gold": "invalid citation"}],
    },))


def valid() -> str:
    return json.dumps({"question_id": "1", "method_id": "M1", "answer_accuracy": {"prediction": "yes", "reason": "reason"}, "sentences": [{"answer_sentence_id": "s1", "relevance_prediction": "required", "relevance_reason": "reason", "citation_assessments": [{"citation_id": "s1:0", "cited_pmid": "10", "evidence_relation_prediction": "supporting", "reason": "reason"}]}]})


class WorkbenchTests(unittest.TestCase):
    def test_actual_fields_load_without_gold_in_prompt(self) -> None:
        fixture = [{"question_id": "1", "question": "Q", "expert_curated_answer": "secret", "machine_generated_answers": {"M1": {"answer": "A", "is_answer_accurate": "yes", "answer_sentences": [{"answer_sentence_id": "s", "answer_sentence": "S", "answer_sentence_relevance": "required", "citation_assessment": [{"cited_pmid": "1", "evidence_support": "E", "evidence_relation": "supporting"}]}]}}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"; path.write_text(json.dumps(fixture))
            loaded = load_workbench_answers(path)[0]
        payload = json.dumps(loaded.prompt_input())
        rendered = prompt_for(loaded.prompt_input())
        self.assertNotIn("secret", payload); self.assertNotIn("supporting", payload)
        self.assertIn("evidence_support", rendered)

    def test_selection_is_deterministic_and_capped(self) -> None:
        examples = [WorkbenchAnswer(str(i), "M1", "q", "a", "yes", ()) for i in range(8)]
        selected, _ = select_representative_answers(examples, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual([item.example_id for item in selected], [item.example_id for item in select_representative_answers(examples, 5)[0]])

    def test_raw_and_fenced_json_parse(self) -> None:
        for raw in (valid(), "```\n" + valid() + "\n```", "```json\n" + valid() + "\n```"):
            self.assertIsNone(parse_workbench_prediction(answer(), raw).malformed_error)

    def test_malformed_missing_and_misaligned_are_not_scored(self) -> None:
        for raw in ("no json", '{"question_id":"1"}', valid().replace('"s1"', '"wrong"')):
            prediction = parse_workbench_prediction(answer(), raw)
            self.assertIsNotNone(prediction.malformed_error)
            self.assertEqual(compare(answer(), prediction)["answer_accuracy"], [])

    def test_unavailable_evidence_is_skipped(self) -> None:
        result = compare(answer(), parse_workbench_prediction(answer(), valid()))
        self.assertEqual(result["evidence_relation"][1]["status"], "skipped")

    def test_human_record_keeps_annotations_out_of_prompt(self) -> None:
        item = answer()
        self.assertEqual(item.human_record()["machine_generated_answer"]["is_answer_accurate"], "yes")
        self.assertNotIn("is_answer_accurate", item.prompt_input())


if __name__ == "__main__": unittest.main()
