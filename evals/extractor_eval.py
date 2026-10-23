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


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_DAY = r"(?:a\s+|per\s+)?(?:day|daily)\b"
# Applied in order; matched text is removed so "twice daily" is not also read as "daily".
_TERMS = [
    ("frequency", "bid", rf"(?<!\w)b\.?i\.?d\.?(?!\w)|\btwice\s+{_DAY}|\b(?:2|two)\s*(?:x|times)\s*{_DAY}"),
    ("frequency", "tid", rf"(?<!\w)t\.?i\.?d\.?(?!\w)|\bthree\s+times\s+{_DAY}"),
    ("frequency", "qid", rf"(?<!\w)q\.?i\.?d\.?(?!\w)|\bfour\s+times\s+{_DAY}"),
    ("frequency", "prn", r"(?<!\w)p\.?r\.?n\.?(?!\w)|\bas\s+needed\b"),
    ("frequency", "qd", rf"(?<!\w)q\.?d\.?(?!\w)|\bonce\s+{_DAY}|\bdaily\b|\bevery\s+day\b"),
    ("time_of_day", "hs", r"\bbedtime\b|(?<!\w)h\.s\.(?!\w)|\bat\s+night\b|\bnightly\b"),
    ("time_of_day", "am", r"\b(?:in\s+the\s+|every\s+)?morning\b|(?<!\w)a\.m\.(?!\w)"),
    ("time_of_day", "pm", r"\b(?:in\s+the\s+|every\s+)?evening\b|(?<!\w)p\.m\.(?!\w)"),
    ("route", "po", r"(?<!\w)p\.?o\.?(?!\w)|\borally\b|\bby\s+mouth\b|\boral\b"),
    ("route", "iv", r"\bintravenous(?:ly)?\b|(?<!\w)i\.?v\.?(?!\w)"),
    ("route", "im", r"\bintramuscular(?:ly)?\b|(?<!\w)i\.m\.(?!\w)"),
    ("route", "sc", r"\bsubcutaneous(?:ly)?\b|(?<!\w)s\.c\.(?!\w)|\bsub-?q\b"),
    ("route", "topical", r"\btopical(?:ly)?\b"),
    ("form", "tablet", r"\btabs?\b|\btablets?\b"),
    ("form", "capsule", r"\bcaps?\b|\bcapsules?\b"),
    ("status", "continue", r"\bcontinu(?:e|ed|es|ing)\b|\bresum(?:e|ed|es|ing)\b"),
    ("status", "start", r"\b(?:start|started|initiate|initiated|begin|began)\b"),
    ("status", "stop", r"\b(?:stop|stopped|discontinue|discontinued|cease|ceased)\b"),
    ("status", "hold", r"\b(?:hold|held)\b"),
    ("status", "increase", r"\b(?:increase|increased)\b"),
    ("status", "decrease", r"\b(?:decrease|decreased|reduce|reduced)\b"),
]
_CATEGORIES = ("frequency", "time_of_day", "route", "form", "status")
_QTY = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|kg|g|ml|l|units?|iu|meq|mmol|%)?(?!\w)")
_NEG = re.compile(r"\b(?:no|not|none|never|without|denies|denied|denying|negative\s+for|"
                  r"absent|free\s+of|cannot|unable)\b|n[’']t\b", re.I)
# Plain-English filler ignored by the medication/lab term comparison. Anything NOT
# listed here and present on only one side raises term_mismatch, which routes the
# pair to human review (the safe direction).
_GENERIC = {"patient", "patients", "prescribed", "instructed", "advised", "take", "takes",
            "taking", "taken", "uses", "using", "used", "given", "receives", "receive",
            "that", "with", "this", "from", "were", "will", "been", "have", "also",
            "additional", "needs", "should", "days", "tablet", "tablets", "capsule", "capsules"}


def _canon(text):
    text = text.lower()
    markers = {c: set() for c in _CATEGORIES}
    for category, label, pattern in _TERMS:
        text, n = re.subn(pattern, " ", text)
        if n:
            markers[category].add(label)
    text = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: str(_NUM_WORDS[m.group(1)]), text)
    quantities = sorted(
        (f"{float(n):g}", {"µg": "mcg", "ug": "mcg"}.get(u, u) if u else "")
        for n, u in _QTY.findall(text)
    )
    return markers, quantities, text


def _review_flags(gold_text, predicted_text, fact_type=None):
    """Advisory heuristics, NOT a clinical entailment or evidence-support check.

    Flags only route a pair to human review; they never prove equivalence.
    Abbreviations are normalized first (p.o.=orally, b.i.d.=twice a day, cap=capsule).
    """
    flags = []
    if bool(_NEG.search(gold_text)) != bool(_NEG.search(predicted_text)):
        flags.append("negation_mismatch")
    gm, gq, gt = _canon(gold_text)
    pm, pq, pt = _canon(predicted_text)
    if gq != pq:
        flags.append("number_mismatch")
    for category in _CATEGORIES:
        if gm[category] != pm[category]:
            flags.append(f"{category}_mismatch")
    if fact_type in {"medication", "lab_value"}:
        tokens = lambda t: set(re.findall(r"[a-z]{4,}", t)) - _GENERIC
        if tokens(gt) ^ tokens(pt):
            flags.append("term_mismatch")
    return flags


def _norm_evidence(sentence):
    return " ".join(sentence.split()).lower()


def _shares_evidence(a_evidence, b_evidence):
    """True if any evidence string from one side overlaps one from the other."""
    return any(_norm_evidence(x) in _norm_evidence(y) or _norm_evidence(y) in _norm_evidence(x)
               for x in a_evidence for y in b_evidence)


def semantic_review_queue(example, predicted, embed, *, min_similarity=0.75,
                          auto_similarity=0.97):
    """Exact-first, one-to-one semantic matching with provisional auto-acceptance.

    The auto threshold is UNVALIDATED. Clinical flags are simple heuristics, not
    proof of equivalence; auto matches must be audited before clinical use.
    """
    if not 0 <= min_similarity <= auto_similarity <= 1:
        raise ValueError("Require 0 <= min_similarity <= auto_similarity <= 1")
    gold = validate_facts(example["facts"], gold=True)
    predicted = validate_facts(predicted)
    remaining = list(enumerate(predicted, 1))
    remaining_gold, exact_matches = [], []
    for fact in gold:
        match = next(((i, p) for i, p in remaining if _key(fact) == _key(p)), None)
        if match is None:
            remaining_gold.append(fact)
        else:
            exact_matches.append({"gold_fact_id": fact["fact_id"],
                                  "prediction_id": match[1]["fact_id"]})
            remaining.remove(match)

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

    used_gold, used_pred, queue, semantic_matches = set(), set(), [], []
    eid = example["example_id"]
    for similarity, g, i, p in candidates:
        if g["fact_id"] in used_gold or i in used_pred:
            continue
        used_gold.add(g["fact_id"])
        used_pred.add(i)
        flags = _review_flags(g["fact_text"], p["fact_text"], g["fact_type"])
        pair = {
            "gold_fact_id": g["fact_id"], "prediction_id": p["fact_id"],
            "gold_fact": g["fact_text"], "predicted_fact": p["fact_text"],
            "fact_type": g["fact_type"],
            "similarity": similarity, "flags": flags,
        }
        if similarity >= auto_similarity and not flags:
            semantic_matches.append({**pair, "decision": "provisional_auto_match"})
        else:
            queue.append({
                "review_id": f"{eid}:{g['fact_id']}:P-{i:03d}",
                "review_type": "semantic_candidate", **pair,
                "gold_evidence": g["source_evidence"],
                "predicted_evidence": p["source_evidence"],
                "human_verdict": None,
            })
    unmatched_gold = [g for g in remaining_gold if g["fact_id"] not in used_gold]
    unmatched_pred = [(i, p) for i, p in remaining if i not in used_pred]
    for g in unmatched_gold:
        queue.append({"review_id": f"{eid}:{g['fact_id']}:unmatched",
                      "review_type": "unmatched_gold",
                      "gold_fact_id": g["fact_id"], "gold_fact": g["fact_text"],
                      "fact_type": g["fact_type"],
                      "gold_evidence": g["source_evidence"],
                      # Possible merge/split: unmatched prediction cites the same evidence.
                      "shares_evidence_with_predictions": [
                          p["fact_id"] for _, p in unmatched_pred
                          if _shares_evidence(g["source_evidence"], p["source_evidence"])],
                      "human_verdict": None})
    for i, p in unmatched_pred:
        queue.append({"review_id": f"{eid}:unmatched:P-{i:03d}",
                      "review_type": "unmatched_prediction",
                      "prediction_id": p["fact_id"], "predicted_fact": p["fact_text"],
                      "fact_type": p["fact_type"],
                      "predicted_evidence": p["source_evidence"],
                      "shares_evidence_with_gold": [
                          g["fact_id"] for g in unmatched_gold
                          if _shares_evidence(g["source_evidence"], p["source_evidence"])],
                      "human_verdict": None,
                      "review_options": ["hallucination", "missed_paraphrase", "missing_from_gold"]})
    # Only exact and provisional auto matches count as TP. Review candidates
    # are deliberately treated as unmatched until a human adjudicates them.
    tp = len(exact_matches) + len(semantic_matches)
    metrics = _metrics(tp, len(predicted) - tp, len(gold) - tp)
    return {"example_id": eid, "source_text": example["source_text"],
            "exact_matches": exact_matches, "semantic_matches": semantic_matches,
            "review_queue": queue, "provisional_semantic_metrics": metrics,
            "note": "Provisional: auto-accept threshold and clinical checks are not human-validated."}


def gemini_embedder(client):
    """Optional live embedder; import the SDK only when explicitly requested."""
    def embed(text):
        response = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return list(response.embeddings[0].values)
    return embed


def generate_semantic_reviews(gold_examples, predictions, embed, *, min_similarity=0.75,
                              auto_similarity=0.97):
    """Score provisional semantic matches; keep unresolved pairs in review."""
    records = _check_records(list(predictions))
    by_id = {r["example_id"]: r["raw_prediction"] for r in records}
    reviews, skipped = [], {}
    total_tp = total_fp = total_fn = 0
    for example in gold_examples:
        eid = example["example_id"]
        if eid not in by_id:
            skipped[eid] = "missing_prediction"
            total_fn += len(example["facts"])
            continue
        try:
            facts = parse_prediction(by_id[eid])
        except ValueError as exc:
            skipped[eid] = str(exc)
            total_fn += len(example["facts"])
            continue
        review = semantic_review_queue(example, facts, embed,
                                       min_similarity=min_similarity,
                                       auto_similarity=auto_similarity)
        reviews.append(review)
        tp = len(review["exact_matches"]) + len(review["semantic_matches"])
        total_tp += tp
        total_fp += len(facts) - tp
        total_fn += len(example["facts"]) - tp
    return {
        "status": "provisional_auto_matches_not_human_validated",
        "min_similarity": min_similarity,
        "auto_similarity": auto_similarity,
        "provisional_semantic_all_outputs": _metrics(total_tp, total_fp, total_fn),
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
    parser.add_argument("--auto-similarity", type=float, default=0.97,
                        help="Provisional auto-match threshold (default: 0.97; not validated)")
    args = parser.parse_args()
    if not 0 <= args.min_similarity <= 1:
        parser.error("--min-similarity must be between 0 and 1")
    if not args.min_similarity <= args.auto_similarity <= 1:
        parser.error("Require --min-similarity <= --auto-similarity <= 1")
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
            auto_similarity=args.auto_similarity,
        )
        Path(args.semantic_review).write_text(
            json.dumps(reviews, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Review queue written to {args.semantic_review}")
        sem = reviews["provisional_semantic_all_outputs"]
        auto = sum(len(e["semantic_matches"]) for e in reviews["examples"])
        pending = sum(len(e["review_queue"]) for e in reviews["examples"])
        fmt = lambda m: " ".join(f"{k}={'n/a' if v is None else format(v, '.3f')}"
                                 for k, v in m.items())
        print("\n=== SUMMARY (provisional, NOT human-validated) ===")
        print("exact (valid only):        ", fmt(result["valid_only"]))
        print("exact (all outputs):       ", fmt(result["all_outputs"]))
        print("provisional semantic (all):", fmt(sem))
        print(f"auto semantic matches: {auto} | items pending human review: {pending}")
        print(f"auto threshold {args.auto_similarity} | candidate cutoff {args.min_similarity} (both unvalidated)\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
