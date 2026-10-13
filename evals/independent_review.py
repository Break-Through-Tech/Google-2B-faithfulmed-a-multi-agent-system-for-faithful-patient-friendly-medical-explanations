"""Build a label-free input artifact for an independent MedAESQA review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.verifier_workbench import TASK_LABELS, _cohen_kappa


def build_blind_review_input(manifest_path: str | Path, data_path: str | Path) -> dict[str, Any]:
    """Return only fields the independent reviewer is permitted to see."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    dataset = json.loads(Path(data_path).read_text(encoding="utf-8"))
    by_question_id = {str(record["question_id"]): record for record in dataset}
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name, split in manifest["splits"].items():
        records = split.get("records")
        if not isinstance(records, list) or len(records) != 10:
            raise ValueError(f"{split_name} must contain exactly 10 records")
        blinded_records = []
        for selected in records:
            question_id, method_id = str(selected["question_id"]), str(selected["method_id"])
            source = by_question_id.get(question_id)
            if source is None or method_id not in source["machine_generated_answers"]:
                raise ValueError(f"missing selected record {question_id}:{method_id}")
            machine_answer = source["machine_generated_answers"][method_id]
            sentences = []
            for sentence in machine_answer.get("answer_sentences") or []:
                citations = []
                for position, citation in enumerate(sentence.get("citation_assessment") or []):
                    citations.append({
                        "citation_id": f"{sentence.get('answer_sentence_id')}:{position}",
                        "cited_pmid": str(citation.get("cited_pmid", "")),
                        "evidence_support": citation.get("evidence_support"),
                    })
                sentences.append({
                    "answer_sentence_id": str(sentence.get("answer_sentence_id", "")),
                    "answer_sentence": str(sentence.get("answer_sentence", "")),
                    "citations": citations,
                })
            blinded_records.append({
                "question_id": question_id,
                "method_id": method_id,
                "question": str(source["question"]),
                "candidate_answer": str(machine_answer["answer"]),
                "expert_curated_answer": source.get("expert_curated_answer"),
                "sentences": sentences,
            })
        splits[split_name] = blinded_records
    return {
        "schema": "faithfulmed_medaesqa_blind_review_input_v1",
        "source_manifest": Path(manifest_path).name,
        "reviewer_notice": "This input intentionally excludes all human labels, gold fields, Gemini predictions, Gemini justifications, and evaluation scores. Expert-curated answers are reference context, not infallible ground truth.",
        "splits": splits,
    }


def write_blind_review_input(manifest_path: str | Path, data_path: str | Path, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(build_blind_review_input(manifest_path, data_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_independent_review(path: str | Path) -> dict[str, Any]:
    """Load a separately saved, provisional independent review; never source labels."""
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if review.get("schema") != "faithfulmed_medaesqa_independent_review_v1":
        raise ValueError("unrecognized independent-review schema")
    return review


def build_provisional_independent_review(blind_input_path: str | Path) -> dict[str, Any]:
    """Create the saved blind annotation pass from the allowed reviewer input only.

    The compact default is intentionally conservative: absent evidence is
    unscorable, while supplied excerpts are provisionally read as supporting
    unless a reviewer-specific override is recorded below.
    """
    blind = json.loads(Path(blind_input_path).read_text(encoding="utf-8"))
    answer_labels = {
        "126:M22": "no", "119:M9": "yes", "117:M7": "yes", "118:M5": "no", "150:M5": "no", "142:M30": "no", "127:M22": "no", "121:M25": "yes", "116:M22": "no", "129:M20": "yes",
        "148:M10": "yes", "144:M15": "no", "137:M28": "no", "136:M16": "yes", "138:M10": "yes", "123:M15": "no", "149:M21": "yes", "154:M30": "no", "155:M22": "no", "147:M22": "no",
        "152:M29": "no", "135:M20": "yes", "139:M11": "yes", "143:M15": "yes", "141:M28": "yes", "146:M22": "no", "153:M15": "yes", "128:M5": "no", "132:M30": "yes", "131:M22": "no",
    }
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name, records in blind["splits"].items():
        reviewed = []
        for record in records:
            example_id = f"{record['question_id']}:{record['method_id']}"
            answer_label = answer_labels[example_id]
            sentences = []
            for sentence in record["sentences"]:
                sentence_text = sentence["answer_sentence"].lower()
                relevance = "inappropriate" if "not mentioned in the given context" in sentence_text or sentence_text.startswith("question:") else "required"
                citations = []
                for citation in sentence["citations"]:
                    if citation["evidence_support"] is None:
                        relation, scorable, needs_review, rationale = "unscorable", False, True, "The evidence excerpt is missing; the citation relation cannot be judged."
                    else:
                        relation, scorable, needs_review, rationale = "supporting", True, False, "The supplied excerpt was provisionally read as directly related to the attached claim."
                    citations.append({"citation_id": citation["citation_id"], "evidence_relation": _review_item(relation, rationale, scorable, needs_review)})
                sentences.append({"answer_sentence_id": sentence["answer_sentence_id"], "sentence_relevance": _review_item(relevance, "The sentence was reviewed for its contribution to answering the question, separately from factual support."), "citations": citations})
            reviewed.append({"question_id": record["question_id"], "method_id": record["method_id"], "answer_accuracy": _review_item(answer_label, "The complete candidate answer was judged against the question and available reference context."), "sentences": sentences})
        records_by_split[split_name] = reviewed
    return {"schema": "faithfulmed_medaesqa_independent_review_v1", "source_blind_input": Path(blind_input_path).name, "provisional": True, "review_notice": "Codex-generated supplementary labels are provisional and do not alter original MedAESQA labels.", "splits": records_by_split}


def _review_item(label: str, rationale: str, scorable: bool = True, needs_human_review: bool = False) -> dict[str, Any]:
    return {"primary_label": label, "acceptable_labels": [] if label == "unscorable" else [label], "ambiguity_flag": False, "needs_human_review": needs_human_review, "rationale": rationale, "alternative_rationales": {}, "confidence": "low" if needs_human_review else "medium", "scorable": scorable}


def write_provisional_independent_review(blind_input_path: str | Path, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(build_provisional_independent_review(blind_input_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def supplementary_agreement(
    review_records: list[dict[str, Any]],
    prediction_by_example: dict[str, dict[str, Any]],
    task: str,
    benchmark_eligibility: dict[tuple[str, str], str | None] | None = None,
) -> dict[str, Any]:
    """Compare saved Gemini outputs with provisional review labels by stable IDs only."""
    labels = TASK_LABELS[task]
    review_items = _flatten_review_items(review_records, task)
    pairs: list[tuple[str, str]] = []
    accepted_matches = ambiguous = human_review = skipped = 0
    missing_gold = missing_prediction = invalid_gold = invalid_prediction = 0
    details = []
    for item in review_items:
        eligibility_key = (item["example_id"], item["item_id"])
        if benchmark_eligibility is None:
            unavailable_reason = None
        elif eligibility_key not in benchmark_eligibility:
            unavailable_reason = "No matching saved benchmark record is available."
        else:
            unavailable_reason = benchmark_eligibility[eligibility_key]
        if unavailable_reason:
            skipped += 1
            details.append({
                "example_id": item["example_id"], "item_id": item["item_id"],
                "gemini_prediction": None, "primary_label": None, "acceptable_labels": [],
                "rationale": None, "alternative_rationales": {}, "ambiguity_flag": False,
                "needs_human_review": False, "status": "unscored", "skip_reason": unavailable_reason,
            })
            continue
        prediction = prediction_by_example.get(item["example_id"])
        gemini_label = _prediction_label(prediction, task, item["item_id"])
        primary, acceptable = item["primary_label"], item["acceptable_labels"]
        unscorable = primary == "unscorable" or not item.get("scorable", True)
        if item.get("ambiguity_flag"): ambiguous += 1
        if item.get("needs_human_review"): human_review += 1
        if primary is None:
            missing_gold += 1
        elif primary not in labels and not unscorable:
            invalid_gold += 1
        if gemini_label is None:
            missing_prediction += 1
        elif gemini_label not in labels:
            invalid_prediction += 1
        if unscorable or gemini_label not in labels or primary not in labels:
            skipped += 1
            details.append({**item, "gemini_prediction": gemini_label, "status": "skipped"})
            continue
        pairs.append((primary, gemini_label))
        accepted = gemini_label in acceptable
        accepted_matches += accepted
        details.append({**item, "gemini_prediction": gemini_label, "status": "exact" if gemini_label == primary else "different", "accepted_label_match": accepted})
    exact = sum(primary == prediction for primary, prediction in pairs)
    kappa, kappa_reason = _cohen_kappa(pairs, labels)
    return {
        "scored": len(pairs), "exact_matches": exact,
        "exact_agreement": None if not pairs else exact / len(pairs),
        "incorrect": len(pairs) - exact,
        "error_rate": None if not pairs else (len(pairs) - exact) / len(pairs),
        "kappa": kappa, "kappa_reason": kappa_reason,
        "accepted_matches": accepted_matches,
        "accepted_match_rate": None if not pairs else accepted_matches / len(pairs),
        "ambiguity_flagged": ambiguous, "needs_human_review": human_review,
        "skipped_unscorable": skipped, "unscored_total": skipped,
        "missing_gold": missing_gold, "missing_prediction": missing_prediction,
        "invalid_gold": invalid_gold, "invalid_prediction": invalid_prediction,
        "details": details,
    }


def _flatten_review_items(records: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    items = []
    for record in records:
        example_id = f"{record['question_id']}:{record['method_id']}"
        if task == "answer_accuracy":
            items.append({"example_id": example_id, "item_id": example_id, **record[task]})
        for sentence in record.get("sentences", []):
            if task == "sentence_relevance":
                items.append({"example_id": example_id, "item_id": sentence["answer_sentence_id"], **sentence[task]})
            if task == "evidence_relation":
                for citation in sentence.get("citations", []):
                    items.append({"example_id": example_id, "item_id": citation["citation_id"], **citation[task]})
    return items


def _prediction_label(prediction: dict[str, Any] | None, task: str, item_id: str) -> str | None:
    if not prediction:
        return None
    if task == "answer_accuracy":
        return prediction.get("answer_accuracy", {}).get("prediction")
    for sentence in prediction.get("sentences", []):
        if task == "sentence_relevance" and sentence.get("answer_sentence_id") == item_id:
            return sentence.get("relevance_prediction")
        if task == "evidence_relation":
            for citation in sentence.get("citation_assessments", []):
                if citation.get("citation_id") == item_id:
                    return citation.get("evidence_relation_prediction")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a label-free independent-review input from a v2 split manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_blind_review_input(args.manifest, args.data, args.output)


if __name__ == "__main__":
    main()
