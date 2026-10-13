"""Offline Extractor evaluation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FACT_TYPES = frozenset({
    "diagnosis", "medication", "lab_value", "procedure", "instruction", "demographics"
})


def validate_facts(facts, *, gold=False):
    """Check structure, not clinical truth. Gold additionally requires unique IDs."""
    if not isinstance(facts, list):
        raise ValueError("Facts must be a JSON list.")
    ids = set()
    for index, fact in enumerate(facts):
        prefix = f"Fact {index + 1}"
        if not isinstance(fact, dict):
            raise ValueError(f"{prefix} must be an object.")
        required = {"fact_text", "fact_type", "source_evidence"}
        if gold:
            required.add("fact_id")
        if set(fact) != required:
            raise ValueError(f"{prefix} must have exactly these fields: {sorted(required)}")
        if not isinstance(fact["fact_text"], str) or not fact["fact_text"].strip():
            raise ValueError(f"{prefix} needs nonempty fact_text.")
        kind = fact["fact_type"]
        if not isinstance(kind, str) or kind not in FACT_TYPES:
            raise ValueError(f"{prefix} has an unapproved fact_type.")
        evidence = fact["source_evidence"]
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(sentence, str) or not sentence.strip() for sentence in evidence
        ):
            raise ValueError(f"{prefix} needs a nonempty list of evidence strings.")
        if gold:
            fact_id = fact["fact_id"]
            if not isinstance(fact_id, str) or not fact_id.strip() or fact_id in ids:
                raise ValueError(f"{prefix} needs a unique nonempty fact_id.")
            ids.add(fact_id)
    return facts


def parse_prediction(raw_prediction):
    """Require JSON only; prose and Markdown fences are reported as invalid."""
    if not isinstance(raw_prediction, str):
        raise ValueError("raw_prediction must be a string.")
    try:
        facts = json.loads(raw_prediction)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    return validate_facts(facts)


def load_gold(paths):
    """Load explicitly selected adjudicated JSON files; do not scan test folders."""
    examples = []
    ids = set()
    for path in paths:
        example = json.loads(Path(path).read_text(encoding="utf-8"))
        example_id = example.get("example_id")
        if not isinstance(example_id, str) or not example_id.strip() or example_id in ids:
            raise ValueError("Gold examples need unique nonempty example_id values.")
        source = example.get("source_text")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{example_id}: source_text must be nonempty.")
        validate_facts(example.get("facts"), gold=True)
        for fact in example["facts"]:
            if any(sentence not in source for sentence in fact["source_evidence"]):
                raise ValueError(f"{example_id}/{fact['fact_id']}: evidence absent from source.")
        ids.add(example_id)
        examples.append(example)
    return examples


def _check_records(records):
    ids = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"example_id", "raw_prediction"}:
            raise ValueError("Prediction records require example_id and raw_prediction only.")
        example_id = record["example_id"]
        if not isinstance(example_id, str) or not example_id.strip() or example_id in ids:
            raise ValueError("Prediction IDs must be unique nonempty strings.")
        if not isinstance(record["raw_prediction"], str):
            raise ValueError("raw_prediction must be a string.")
        ids.add(example_id)
    return records


def save_predictions(path, records):
    """Keep raw outputs, including malformed responses, for later offline scoring."""
    records = _check_records(list(records))
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_predictions(path):
    records = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return _check_records(records)


async def generate_predictions(examples, runner):
    """Calls the supplied agent runner. Unlike scoring, this may use model quota."""
    records = []
    for example in examples:
        raw = await runner(example["source_text"])
        records.append({"example_id": example["example_id"], "raw_prediction": raw})
    return _check_records(records)


def _key(fact):
    # Preserve case, punctuation, numbers and negation. Normalize whitespace only.
    return (" ".join(fact["fact_text"].split()), fact["fact_type"])


def score_predictions(gold_examples, predictions):
    """Micro exact-text-and-type scores on valid outputs; invalid/missing reported."""
    gold_examples = list(gold_examples)
    predictions = _check_records(list(predictions))
    gold_ids = [example["example_id"] for example in gold_examples]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Duplicate gold example_id.")
    by_id = {record["example_id"]: record["raw_prediction"] for record in predictions}
    result = {
        "matching_rule": "whitespace-normalized exact fact_text + fact_type; one-to-one",
        "gold_examples": len(gold_examples), "scored_examples": 0,
        "missing_prediction_ids": [], "invalid_predictions": {},
        "prediction_ids_without_gold": sorted(set(by_id) - set(gold_ids)),
        "matched_atoms": 0, "unmatched_predicted_atoms": 0, "unmatched_gold_atoms": 0,
        "examples": {},
    }
    evidence_total = evidence_present = 0
    for example in gold_examples:
        example_id = example["example_id"]
        validate_facts(example["facts"], gold=True)
        if example_id not in by_id:
            result["missing_prediction_ids"].append(example_id)
            continue
        try:
            predicted = parse_prediction(by_id[example_id])
        except ValueError as exc:
            result["invalid_predictions"][example_id] = str(exc)
            continue
        gold = example["facts"]
        expected = Counter(_key(fact) for fact in gold)
        actual = Counter(_key(fact) for fact in predicted)
        matched = sum((expected & actual).values())
        result["matched_atoms"] += matched
        result["unmatched_predicted_atoms"] += len(predicted) - matched
        result["unmatched_gold_atoms"] += len(gold) - matched
        result["scored_examples"] += 1

        remaining = expected.copy()
        extra_indices = []
        for index, fact in enumerate(predicted):
            key = _key(fact)
            if remaining[key]:
                remaining[key] -= 1
            else:
                extra_indices.append(index)
        remaining = actual.copy()
        missing_ids = []
        for fact in gold:
            key = _key(fact)
            if remaining[key]:
                remaining[key] -= 1
            else:
                missing_ids.append(fact["fact_id"])
        bad_evidence = []
        for index, fact in enumerate(predicted):
            for sentence in fact["source_evidence"]:
                evidence_total += 1
                if sentence in example["source_text"]:
                    evidence_present += 1
                else:
                    bad_evidence.append({"prediction_index": index, "sentence": sentence})
        result["examples"][example_id] = {
            "matched_atoms": matched,
            "unmatched_prediction_indices": extra_indices,
            "unmatched_gold_fact_ids": missing_ids,
            "duplicate_predicted_atoms": sum(count - 1 for count in actual.values()),
            "evidence_not_in_source": bad_evidence,
        }

    tp = result["matched_atoms"]
    fp = result["unmatched_predicted_atoms"]
    fn = result["unmatched_gold_atoms"]
    scored = result["scored_examples"]
    result["exact_precision"] = tp / (tp + fp) if scored and tp + fp else None
    result["exact_recall"] = tp / (tp + fn) if scored and tp + fn else None
    result["exact_f1"] = 2 * tp / (2 * tp + fp + fn) if scored and 2 * tp + fp + fn else None
    attempted = scored + len(result["invalid_predictions"])
    result["schema_valid_rate"] = scored / attempted if attempted else None
    result["evidence_verbatim_rate"] = evidence_present / evidence_total if evidence_total else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", nargs="+", required=True, help="Adjudicated JSON files")
    parser.add_argument("--predictions", required=True, help="Saved JSONL predictions")
    parser.add_argument("--output", required=True, help="Write metrics JSON here")
    args = parser.parse_args()
    result = score_predictions(load_gold(args.gold), load_predictions(args.predictions))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
