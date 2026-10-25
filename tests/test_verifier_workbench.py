from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evals.verifier_workbench import (WorkbenchAnswer, compare, load_workbench_answers,
                                      export_review_summary, load_all_evaluation_splits, load_frozen_test_set, parse_workbench_prediction, prompt_for,
                                      REVISED_PROMPT_VERSION, mismatch_diagnostic,
                                      random_answer_per_question, resolve_answer, select_representative_answers,
                                      select_balanced_answer_accuracy_answers, task_agreement_statistics, task_error_rate)


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
        leaky_payload = {
            **loaded.prompt_input(),
            "expert_curated_answer": "secret",
            "is_answer_accurate": "yes",
            "answer_accuracy_gold": "yes",
        }
        rendered = prompt_for(leaky_payload)
        sent = json.loads(rendered.rsplit("INPUT JSON:\n", 1)[1])
        self.assertNotIn("secret", payload); self.assertNotIn("supporting", payload)
        self.assertEqual(sent, loaded.prompt_input())
        self.assertNotIn("expert_curated_answer", sent)
        self.assertNotIn("is_answer_accurate", sent["machine_generated_answer"])
        self.assertNotIn("evidence_relation", json.dumps(sent))
        self.assertIn("Assess internally in this order", rendered)
        self.assertIn("For the required-versus-borderline decision", rendered)
        self.assertIn("Omit that citation from citation_assessments", rendered)

    def test_selection_is_deterministic_and_capped(self) -> None:
        examples = [WorkbenchAnswer(str(i), "M1", "q", "a", "yes", ()) for i in range(8)]
        selected, _ = select_representative_answers(examples, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual([item.example_id for item in selected], [item.example_id for item in select_representative_answers(examples, 5)[0]])

    def test_random_candidate_pool_has_one_answer_per_question(self) -> None:
        examples = [
            WorkbenchAnswer("1", "M1", "q", "a", "yes", ()),
            WorkbenchAnswer("1", "M2", "q", "b", "yes", ()),
            WorkbenchAnswer("2", "M1", "q", "c", "yes", ()),
            WorkbenchAnswer("2", "M2", "q", "d", "yes", ()),
        ]
        selected = random_answer_per_question(examples, seed=42)
        self.assertEqual([item.question_id for item in selected], ["1", "2"])
        self.assertEqual(len({item.question_id for item in selected}), len(selected))

    def test_frozen_set_resolves_fixed_unique_question_and_method_pairs(self) -> None:
        examples = [
            WorkbenchAnswer("1", "M1", "q", "a", "yes", ()),
            WorkbenchAnswer("1", "M2", "q", "b", "yes", ()),
            WorkbenchAnswer("2", "M3", "q", "c", "yes", ()),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps({"records": [
                {"question_id": "1", "method_id": "M2"},
                {"question_id": "2", "method_id": "M3"},
            ]}))
            selected = load_frozen_test_set(path, examples)
        self.assertEqual([item.example_id for item in selected], ["1:M2", "2:M3"])

    def test_saved_ten_record_batch_is_stable_and_has_distinct_questions(self) -> None:
        examples = [WorkbenchAnswer(str(index), "M1", "q", "a", "yes", ()) for index in range(1, 11)]
        records = [{"question_id": str(index), "method_id": "M1"} for index in range(1, 11)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            path.write_text(json.dumps({"random_seed": 7, "records": records}))
            first = load_frozen_test_set(path, examples)
            second = load_frozen_test_set(path, examples)
        self.assertEqual([item.example_id for item in first], [item.example_id for item in second])
        self.assertEqual(len(first), 10)
        self.assertEqual(len({item.question_id for item in first}), 10)

    def test_three_evaluation_splits_are_disjoint_and_reproducible(self) -> None:
        examples = [WorkbenchAnswer(str(index), "M1", "q", "a", "yes", ()) for index in range(1, 31)]
        split_names = ("prompt-development", "validation", "final-test")
        manifest = {
            "selection_seed": 101,
            "splits": {
                name: {"role": name, "records": [
                    {"question_id": str(offset + number), "method_id": "M1"}
                    for number in range(1, 11)
                ]}
                for name, offset in zip(split_names, (0, 10, 20))
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps(manifest))
            first = load_all_evaluation_splits(path, examples)
            second = load_all_evaluation_splits(path, examples)
        first_ids = {name: [item.example_id for item in items] for name, (items, _) in first.items()}
        second_ids = {name: [item.example_id for item in items] for name, (items, _) in second.items()}
        self.assertEqual(first_ids, second_ids)
        self.assertEqual({name: len(ids) for name, ids in first_ids.items()}, {name: 10 for name in split_names})
        question_ids = [item.question_id for items, _ in first.values() for item in items]
        self.assertEqual(len(question_ids), len(set(question_ids)))
        self.assertEqual(first["validation"][1]["selection_seed"], 101)

    def test_balanced_answer_accuracy_selection_is_reproducible(self) -> None:
        examples = [
            WorkbenchAnswer(str(index), method, "q", "a", label, ())
            for index in range(1, 13)
            for method, label in (("M1", "yes"), ("M2", "no"))
        ]
        first = select_balanced_answer_accuracy_answers(examples, seed=42)
        second = select_balanced_answer_accuracy_answers(examples, seed=42)
        self.assertEqual([item.example_id for item in first], [item.example_id for item in second])
        self.assertEqual(Counter(item.answer_accuracy_gold for item in first), Counter({"yes": 5, "no": 5}))
        self.assertEqual(len({item.question_id for item in first}), 10)

    def test_balanced_v2_manifest_has_disjoint_five_yes_five_no_splits(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "evals" / "verifier_evaluation_splits_v2_balanced.json"
        manifest = json.loads(manifest_path.read_text())
        loaded = load_all_evaluation_splits(
            manifest_path,
            load_workbench_answers(root / "data" / "MedAESQA" / "medaesqa_v1.json"),
        )
        all_question_ids = []
        for name, (records, metadata) in loaded.items():
            self.assertEqual(len(records), 10)
            self.assertEqual(Counter(item.answer_accuracy_gold for item in records), Counter({"yes": 5, "no": 5}))
            self.assertEqual(metadata["label_counts"]["answer_accuracy"], {"yes": 5, "no": 5})
            all_question_ids.extend(item.question_id for item in records)
        self.assertEqual(len(all_question_ids), len(set(all_question_ids)))
        final_question_ids = {item.question_id for item in loaded["final-test"][0]}
        self.assertTrue(final_question_ids.isdisjoint(manifest["historical_verifier_question_ids"]))

    def test_typed_question_and_method_resolve_to_one_record(self) -> None:
        examples = [WorkbenchAnswer("127", "M6", "q", "a", "yes", ())]
        self.assertEqual(resolve_answer(examples, " 127 ", "m6").example_id, "127:M6")
        with self.assertRaisesRegex(ValueError, "No MedAESQA machine answer"):
            resolve_answer(examples, "127", "M7")

    def test_raw_and_fenced_json_parse(self) -> None:
        for raw in (valid(), "```\n" + valid() + "\n```", "```json\n" + valid() + "\n```"):
            self.assertIsNone(parse_workbench_prediction(answer(), raw).malformed_error)

    def test_only_retained_prompt_uses_expected_schema(self) -> None:
        payload = answer().prompt_input()
        revised = prompt_for(payload, REVISED_PROMPT_VERSION)
        self.assertIn("Answer accuracy:", revised)
        self.assertIn("INPUT JSON:\n" + json.dumps(payload), revised)
        with self.assertRaisesRegex(ValueError, "Unknown Verifier prompt version"):
            prompt_for(payload, "retired-prompt")

    def test_retained_prompt_is_sanitized(self) -> None:
        leaky = {**answer().prompt_input(), "expert_curated_answer": "secret", "answer_accuracy_gold": "yes"}
        rendered = prompt_for(leaky)
        self.assertIn("INPUT JSON:\n" + json.dumps(answer().prompt_input()), rendered)
        payload = json.loads(rendered.rsplit("INPUT JSON:\n", 1)[1])
        self.assertNotIn("secret", rendered)
        self.assertNotIn("answer_accuracy_gold", rendered)
        self.assertNotIn("is_answer_accurate", json.dumps(payload))

    def test_malformed_missing_and_misaligned_are_not_scored(self) -> None:
        for raw in ("no json", '{"question_id":"1"}', valid().replace('"s1"', '"wrong"')):
            prediction = parse_workbench_prediction(answer(), raw)
            self.assertIsNotNone(prediction.malformed_error)
            self.assertEqual(compare(answer(), prediction)["answer_accuracy"], [])

    def test_unavailable_evidence_is_skipped(self) -> None:
        result = compare(answer(), parse_workbench_prediction(answer(), valid()))
        self.assertEqual(result["evidence_relation"][1]["status"], "skipped")
        self.assertIsNone(result["evidence_relation"][1]["predicted"])

    def test_null_evidence_must_be_omitted_from_citation_assessments(self) -> None:
        data = json.loads(valid())
        data["sentences"][0]["citation_assessments"].append({
            "citation_id": "s1:1", "cited_pmid": "11", "evidence_relation_prediction": "invalid citation", "reason": "reason",
        })
        self.assertIsNotNone(parse_workbench_prediction(answer(), json.dumps(data)).malformed_error)

    def test_human_record_keeps_annotations_out_of_prompt(self) -> None:
        item = answer()
        self.assertEqual(item.human_record()["machine_generated_answer"]["is_answer_accurate"], "yes")
        self.assertNotIn("is_answer_accurate", item.prompt_input())

    def test_error_rate_excludes_unscored_items(self) -> None:
        result = compare(answer(), parse_workbench_prediction(answer(), valid()))
        result["evidence_relation"][0]["status"] = "incorrect"
        self.assertEqual(task_error_rate([result], "evidence_relation"), 1.0)

    def test_kappa_reports_perfect_agreement(self) -> None:
        comparisons = [{"answer_accuracy": [
            {"gold": "yes", "predicted": "yes"},
            {"gold": "no", "predicted": "no"},
        ], "malformed": False}]
        metrics = task_agreement_statistics(comparisons, "answer_accuracy")
        self.assertEqual(metrics["scored_pairs"], 2)
        self.assertEqual((metrics["correct"], metrics["incorrect"]), (2, 0))
        self.assertEqual(metrics["exact_agreement"], 1.0)
        self.assertEqual(metrics["kappa"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["yes"]["yes"], 1)
        self.assertEqual(metrics["confusion_matrix"]["no"]["no"], 1)

    def test_kappa_reports_chance_level_binary_agreement(self) -> None:
        comparisons = [{"answer_accuracy": [
            {"gold": "yes", "predicted": "yes"},
            {"gold": "no", "predicted": "no"},
            {"gold": "yes", "predicted": "no"},
            {"gold": "no", "predicted": "yes"},
        ], "malformed": False}]
        metrics = task_agreement_statistics(comparisons, "answer_accuracy")
        self.assertEqual(metrics["exact_agreement"], 0.5)
        self.assertAlmostEqual(metrics["kappa"], 0.0)

    def test_kappa_calculates_zero_when_gemini_uses_one_label(self) -> None:
        comparisons = [{"answer_accuracy": [
            *[{"gold": "yes", "predicted": "yes"} for _ in range(9)],
            {"gold": "no", "predicted": "yes"},
        ], "malformed": False}]
        metrics = task_agreement_statistics(comparisons, "answer_accuracy")
        self.assertEqual(metrics["confusion_matrix"]["yes"]["yes"], 9)
        self.assertEqual(metrics["confusion_matrix"]["no"]["yes"], 1)
        self.assertEqual(metrics["exact_agreement"], 0.9)
        self.assertEqual(metrics["kappa"], 0.0)

    def test_kappa_supports_multiclass_relevance_and_confusion_matrix(self) -> None:
        comparisons = [{"sentence_relevance": [
            {"gold": "required", "predicted": "required"},
            {"gold": "borderline", "predicted": "unnecessary"},
            {"gold": "unnecessary", "predicted": "unnecessary"},
            {"gold": "inappropriate", "predicted": "inappropriate"},
        ], "malformed": False}]
        metrics = task_agreement_statistics(comparisons, "sentence_relevance")
        self.assertEqual(metrics["labels"], ("required", "borderline", "unnecessary", "inappropriate"))
        self.assertEqual(metrics["confusion_matrix"]["borderline"]["unnecessary"], 1)
        self.assertEqual((metrics["correct"], metrics["incorrect"]), (3, 1))
        self.assertIsNotNone(metrics["kappa"])

    def test_kappa_separates_missing_and_skipped_pairs(self) -> None:
        comparisons = [{"evidence_relation": [
            {"gold": "supporting", "predicted": "supporting"},
            {"gold": None, "predicted": "supporting"},
            {"gold": "neutral", "predicted": None},
        ], "malformed": False}]
        metrics = task_agreement_statistics(comparisons, "evidence_relation")
        self.assertEqual(metrics["scored_pairs"], 1)
        self.assertEqual(metrics["skipped_pairs"], 0)
        self.assertEqual(metrics["unscored_pairs"], 2)
        self.assertEqual(metrics["missing_gold_pairs"], 1)
        self.assertEqual(metrics["missing_prediction_pairs"], 1)
        self.assertEqual(metrics["incorrect"], 0)

    def test_null_evidence_pair_is_reported_as_skipped(self) -> None:
        comparison = compare(answer(), parse_workbench_prediction(answer(), valid()))
        metrics = task_agreement_statistics([comparison], "evidence_relation")
        self.assertEqual(metrics["skipped_pairs"], 1)
        self.assertEqual(metrics["missing_gold_pairs"], 1)
        self.assertEqual(metrics["unscored_pairs"], 1)

    def test_kappa_is_undefined_only_when_expected_agreement_is_one(self) -> None:
        comparisons = [{"answer_accuracy": [
            {"gold": "yes", "predicted": "yes"},
            {"gold": "yes", "predicted": "yes"},
        ], "malformed": False}]
        metrics = task_agreement_statistics(comparisons, "answer_accuracy")
        self.assertIsNone(metrics["kappa"])
        self.assertIn("expected chance agreement is one", metrics["kappa_reason"])

    def test_kappa_is_undefined_without_eligible_pairs(self) -> None:
        comparisons = [{"evidence_relation": [
            {"gold": None, "predicted": None},
        ], "malformed": False}]
        metrics = task_agreement_statistics(comparisons, "evidence_relation")
        self.assertEqual(metrics["scored_pairs"], 0)
        self.assertIsNone(metrics["kappa"])
        self.assertIn("Not evaluated: no valid gold/prediction pairs", metrics["kappa_reason"])

    def test_mismatch_diagnostics_are_explicitly_heuristic_categories(self) -> None:
        self.assertEqual(mismatch_diagnostic("sentence_relevance", "incorrect"), "Label Disagreement")
        self.assertEqual(mismatch_diagnostic("answer_accuracy", "incorrect"), "Reasoning Failure")
        self.assertEqual(mismatch_diagnostic("evidence_relation", "malformed", malformed=True), "Missing Context")
        self.assertIsNone(mismatch_diagnostic("answer_accuracy", "correct"))

    def test_export_contains_aligned_human_and_gemini_fields(self) -> None:
        item = answer()
        prediction = parse_workbench_prediction(item, valid())
        exported = export_review_summary([{"answer": item, "prediction": prediction}])
        self.assertEqual(exported["answer_accuracy"][0]["human_label"], "yes")
        self.assertEqual(exported["sentence_relevance"][0]["answer_sentence"], "Candidate.")
        evidence = exported["evidence_relation"]
        self.assertEqual(evidence[0]["status"], "correct")
        self.assertEqual(evidence[1]["status"], "skipped")
        self.assertIsNone(evidence[1]["gemini_prediction"])


if __name__ == "__main__": unittest.main()
