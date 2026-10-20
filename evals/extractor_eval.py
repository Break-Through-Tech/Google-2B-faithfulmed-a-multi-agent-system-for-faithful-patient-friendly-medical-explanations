"""Offline Extractor evaluation: exact baseline, optional semantic review, and F1."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

FACT_TYPES = frozenset({
    "diagnosis", "medication", "lab_value", "procedure", "instruction", "demographics"
})


def validate_facts(facts, *, gold=False):
    """Validate the Extractor JSON contract (gold facts also have unique IDs)."""
    if not isinstance(facts, list):
        raise ValueError("Facts must be a JSON list.")
    ids = set()
    for i, fact in enumerate(facts, 1):
        required = {"fact_id", "fact_text", "fact_type", "source_evidence"}
        if not isinstance(fact, dict) or set(fact) != required:
            raise ValueError(f"Fact {i} must have exactly {sorted(required)}")
        if not isinstance(fact["fact_text"], str) or not fact["fact_text"].strip():
            raise ValueError(f"Fact {i}: fact_text must be nonempty")
        if not isinstance(fact["fact_type"], str) or fact["fact_type"] not in FACT_TYPES:
            raise ValueError(f"Fact {i}: invalid fact_type")
        evidence = fact["source_evidence"]
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(s, str) or not s.strip() for s in evidence
        ):
            raise ValueError(f"Fact {i}: source_evidence must contain nonempty strings")
        fact_id = fact["fact_id"]
        if not isinstance(fact_id, str) or not fact_id.strip() or fact_id in ids:
            raise ValueError(f"Fact {i}: fact_id must be unique and nonempty")
        ids.add(fact_id)
    return facts


def parse_prediction(raw_prediction):
    if not isinstance(raw_prediction, str):
        raise ValueError("raw_prediction must be a string")
    try:
        facts = json.loads(raw_prediction)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    return validate_facts(facts)


def load_gold(paths):
    examples, ids = [], set()
    for path in paths:
        example = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(example, dict):
            raise ValueError("Gold example must be an object")
        eid, source = example.get("example_id"), example.get("source_text")
        if not isinstance(eid, str) or not eid.strip() or eid in ids:
            raise ValueError("Gold examples need unique nonempty example_id values")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{eid}: source_text must be nonempty")
        validate_facts(example.get("facts"), gold=True)
        for fact in example["facts"]:
            if any(sentence not in source for sentence in fact["source_evidence"]):
                raise ValueError(f"{eid}/{fact['fact_id']}: evidence absent from source")
        examples.append(example)
        ids.add(eid)
    return examples


def _check_records(records):
    ids = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"example_id", "raw_prediction"}:
            raise ValueError("Prediction records require example_id and raw_prediction only")
        eid = record["example_id"]
        if not isinstance(eid, str) or not eid.strip() or eid in ids:
            raise ValueError("Prediction IDs must be unique nonempty strings")
        if not isinstance(record["raw_prediction"], str):
            raise ValueError("raw_prediction must be a string")
        ids.add(eid)
    return records


def save_predictions(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in _check_records(list(records)):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_predictions(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return _check_records(records)


async def generate_predictions(examples, runner):
    records = []
    for example in examples:
        raw = await runner(example["source_text"])
        records.append({"example_id": example["example_id"], "raw_prediction": raw})
    return _check_records(records)


def _key(fact):
    # Baseline intentionally preserves case, punctuation, negation, and numbers.
    return (" ".join(fact["fact_text"].split()), fact["fact_type"])


def _metrics(tp, fp, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def score_predictions(gold_examples, predictions):
    """Exact micro baseline; missing/invalid examples count as FNs in all-output F1."""
    gold_examples = list(gold_examples)
    predictions = _check_records(list(predictions))
    gold_ids = [e["example_id"] for e in gold_examples]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Duplicate gold example_id")
    by_id = {r["example_id"]: r["raw_prediction"] for r in predictions}
    result = {
        "matching_rule": "whitespace-normalized exact fact_text + fact_type; one-to-one",
        "gold_examples": len(gold_examples), "scored_examples": 0,
        "missing_prediction_ids": [], "invalid_predictions": {},
        "prediction_ids_without_gold": sorted(set(by_id) - set(gold_ids)),
        "matched_atoms": 0, "unmatched_predicted_atoms": 0, "unmatched_gold_atoms": 0,
        "examples": {},
    }
    evidence_total = evidence_present = invalid_gold_count = 0
    for example in gold_examples:
        eid, gold = example["example_id"], example["facts"]
        validate_facts(gold, gold=True)
        if eid not in by_id:
            result["missing_prediction_ids"].append(eid)
            invalid_gold_count += len(gold)
            continue
        try:
            predicted = parse_prediction(by_id[eid])
        except ValueError as exc:
            result["invalid_predictions"][eid] = str(exc)
            invalid_gold_count += len(gold)
            continue
        expected, actual = Counter(map(_key, gold)), Counter(map(_key, predicted))
        matched = sum((expected & actual).values())
        result["matched_atoms"] += matched
        result["unmatched_predicted_atoms"] += len(predicted) - matched
        result["unmatched_gold_atoms"] += len(gold) - matched
        result["scored_examples"] += 1
        left = expected.copy()
        extra_indices = []
        for i, fact in enumerate(predicted):
            k = _key(fact)
            if left[k]:
                left[k] -= 1
            else:
                extra_indices.append(i)
        left = actual.copy()
        missing_ids = []
        for fact in gold:
            k = _key(fact)
            if left[k]:
                left[k] -= 1
            else:
                missing_ids.append(fact["fact_id"])
        bad_evidence = []
        for i, fact in enumerate(predicted):
            for sentence in fact["source_evidence"]:
                evidence_total += 1
                if sentence in example["source_text"]:
                    evidence_present += 1
                else:
                    bad_evidence.append({"prediction_index": i, "prediction_fact_id": fact["fact_id"], "sentence": sentence})
        result["examples"][eid] = {
            "matched_atoms": matched,
            "unmatched_prediction_indices": extra_indices,
            "unmatched_gold_fact_ids": missing_ids,
            "duplicate_predicted_atoms": sum(n - 1 for n in actual.values()),
            "evidence_not_in_source": bad_evidence,
        }
    tp, fp, fn = (result[k] for k in (
        "matched_atoms", "unmatched_predicted_atoms", "unmatched_gold_atoms"
    ))
    valid = _metrics(tp, fp, fn) if result["scored_examples"] else {"precision": None, "recall": None, "f1": None}
    all_outputs = _metrics(tp, fp, fn + invalid_gold_count) if gold_examples else {"precision": None, "recall": None, "f1": None}
    result.update({
        "exact_precision": valid["precision"], "exact_recall": valid["recall"], "exact_f1": valid["f1"],
        "valid_only": valid, "all_outputs": all_outputs,
        "schema_valid_rate": result["scored_examples"] / len(gold_examples) if gold_examples else None,
        "evidence_verbatim_rate": evidence_present / evidence_total if evidence_total else None,
    })
    return result


def _review_flags(gold_text, predicted_text):
    """Advisory heuristics, NOT a clinical entailment or evidence-support check."""
    neg = r"\b(?:no|not|denies|denied|without|never|negative for|doesn't|isn't)\b"
    numbers = r"\b\d+(?:\.\d+)?\b"
    flags = []
    if bool(re.search(neg, gold_text, re.I)) != bool(re.search(neg, predicted_text, re.I)):
        flags.append("negation_mismatch")
    if re.findall(numbers, gold_text) != re.findall(numbers, predicted_text):
        flags.append("number_mismatch")
    return flags


def semantic_review_queue(example, predicted, embed, *, min_similarity=0.75):
    """Exact-first matching, then embedding-based *review candidates*.

    embed(text) is an injected function (e.g. Gemini API), so offline scoring
    never requires a network call. Threshold is provisional, not validated.
    Every non-exact prediction/gold fact remains pending human adjudication.
    """
    gold = validate_facts(example["facts"], gold=True)
    predicted = validate_facts(predicted)
    remaining = list(enumerate(predicted, 1))
    remaining_gold, exact_matches = [], []
    for fact in gold:
        match = next(((i, p) for i, p in remaining if _key(fact) == _key(p)), None)
        if match is None:
            remaining_gold.append(fact)
        else:
            exact_matches.append({"gold_fact_id": fact["fact_id"], "prediction_id": match[1]["fact_id"]})
            remaining.remove(match)

    # Cache each embedding once; compare only same-type facts.
    vectors = {}
    def vector(text):
        if text not in vectors:
            vectors[text] = embed(text)
        return vectors[text]

    candidates = []
    for g in remaining_gold:
        for i, p in remaining:
            if g["fact_type"] != p["fact_type"]:
                continue
            a, b = vector(g["fact_text"]), vector(p["fact_text"])
            if not a or len(a) != len(b):
                raise ValueError("Embedding vectors must be nonempty and equal length")
            denom = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
            if not denom:
                raise ValueError("Embedding vector norm must be nonzero")
            similarity = sum(x*y for x, y in zip(a, b)) / denom
            if similarity >= min_similarity:
                candidates.append((similarity, g, i, p))
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gold, used_pred, queue = set(), set(), []
    eid = example["example_id"]
    for similarity, g, i, p in candidates:
        if g["fact_id"] in used_gold or i in used_pred:
            continue
        used_gold.add(g["fact_id"])
        used_pred.add(i)
        queue.append({
            "review_id": f"{eid}:{g['fact_id']}:P-{i:03d}", "review_type": "semantic_candidate",
            "gold_fact_id": g["fact_id"], "prediction_id": p["fact_id"],
            "gold_fact": g["fact_text"], "predicted_fact": p["fact_text"],
            "gold_evidence": g["source_evidence"], "predicted_evidence": p["source_evidence"],
            "similarity": similarity,
            "flags": _review_flags(g["fact_text"], p["fact_text"]),
            "human_verdict": None,
        })
    for g in remaining_gold:
        if g["fact_id"] not in used_gold:
            queue.append({"review_id": f"{eid}:{g['fact_id']}:unmatched", "review_type": "unmatched_gold",
                          "gold_fact_id": g["fact_id"], "gold_fact": g["fact_text"],
                          "gold_evidence": g["source_evidence"], "human_verdict": None})
    for i, p in remaining:
        if i not in used_pred:
            queue.append({"review_id": f"{eid}:unmatched:P-{i:03d}", "review_type": "unmatched_prediction",
                          "prediction_id": p["fact_id"], "predicted_fact": p["fact_text"],
                          "predicted_evidence": p["source_evidence"], "human_verdict": None,
                          "review_options": ["hallucination", "missed_paraphrase", "missing_from_gold"]})
    return {"example_id": eid, "source_text": example["source_text"],
            "exact_matches": exact_matches, "review_queue": queue,
            "note": "Advisory only: no semantic matches counted before human adjudication."}


def gemini_embedder(client):
    """Optional live embedder; import the SDK only when explicitly requested."""
    def embed(text):
        response = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return list(response.embeddings[0].values)
    return embed


def generate_semantic_reviews(gold_examples, predictions, embed, *, min_similarity=0.75):
    """Create advisory review queues for valid outputs; track skipped examples."""
    records = _check_records(list(predictions))
    by_id = {r["example_id"]: r["raw_prediction"] for r in records}
    reviews, skipped = [], {}
    for example in gold_examples:
        eid = example["example_id"]
        if eid not in by_id:
            skipped[eid] = "missing_prediction"
            continue
        try:
            facts = parse_prediction(by_id[eid])
        except ValueError as exc:
            skipped[eid] = str(exc)
            continue
        reviews.append(semantic_review_queue(example, facts, embed, min_similarity=min_similarity))
    return {
        "status": "advisory_only_not_human_validated",
        "min_similarity": min_similarity,
        "examples": reviews,
        "skipped_examples": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", nargs="+", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-review", metavar="JSON_PATH",
                        help="Also write Gemini embedding review candidates to this JSON file")
    parser.add_argument("--min-similarity", type=float, default=0.75,
                        help="Provisional semantic candidate cutoff (default: 0.75)")
    args = parser.parse_args()
    if not 0 <= args.min_similarity <= 1:
        parser.error("--min-similarity must be between 0 and 1")
    if args.semantic_review and Path(args.semantic_review).resolve() == Path(args.output).resolve():
        parser.error("--semantic-review and --output must be different paths")

    gold = load_gold(args.gold)
    predictions = load_predictions(args.predictions)
    result = score_predictions(gold, predictions)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.semantic_review:
        try:
            from google import genai
        except ImportError as exc:
            parser.error(f"Gemini SDK unavailable: {exc}. Install google-genai.")
        reviews = generate_semantic_reviews(
            gold, predictions, gemini_embedder(genai.Client()),
            min_similarity=args.min_similarity,
        )
        Path(args.semantic_review).write_text(
            json.dumps(reviews, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Review queue written to {args.semantic_review}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
