"""Verifier-only MedAESQA workbench helpers; all functions are offline except an injected caller."""

from __future__ import annotations

import hashlib
import json
import math
from random import Random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REVISED_PROMPT_VERSION = "medaesqa-answer-v8"
PROMPT_VERSION = REVISED_PROMPT_VERSION
PROMPT_VERSION_LABELS = {
    REVISED_PROMPT_VERSION: "Revised prompt (v8)",
}
ANSWER_LABELS = {"yes", "no"}
RELEVANCE_LABELS = {"required", "borderline", "unnecessary", "inappropriate"}
EVIDENCE_LABELS = {"supporting", "contradicting", "neutral", "not relevant", "invalid citation"}
MAX_BATCH_SIZE = 10
TASK_LABELS = {
    "answer_accuracy": ("yes", "no"),
    "sentence_relevance": ("required", "borderline", "unnecessary", "inappropriate"),
    "evidence_relation": ("supporting", "contradicting", "neutral", "not relevant", "invalid citation"),
}
KAPPA_TARGETS = {
    "answer_accuracy": "Documented calibration target: ≥80% exact agreement and κ ≥0.60.",
    "evidence_relation": "Documented calibration target: ≥80% exact agreement and κ ≥0.60.",
}


@dataclass(frozen=True)
class WorkbenchAnswer:
    question_id: str
    method_id: str
    question: str
    candidate_answer: str
    answer_accuracy_gold: str | None
    sentences: tuple[dict[str, Any], ...]
    expert_reference: str | None = None

    @property
    def example_id(self) -> str:
        return f"{self.question_id}:{self.method_id}"

    def prompt_input(self) -> dict[str, Any]:
        """Build the allowlisted Gemini request; no expert answer or annotation fields."""
        sentences = []
        for sentence in self.sentences:
            citations = []
            for citation in sentence["citations"]:
                citations.append({key: citation[key] for key in ("citation_id", "cited_pmid", "evidence_support")})
            sentences.append({"answer_sentence_id": sentence["answer_sentence_id"], "answer_sentence": sentence["answer_sentence"], "citations": citations})
        return {
            "question_id": self.question_id,
            "question": self.question,
            "machine_generated_answer": {
                "method_id": self.method_id,
                "answer": self.candidate_answer,
                "answer_sentences": sentences,
            },
        }

    def coverage_labels(self) -> set[str]:
        labels = {f"accuracy:{self.answer_accuracy_gold}"} if self.answer_accuracy_gold else set()
        for sentence in self.sentences:
            if sentence["relevance_gold"]:
                labels.add(f"relevance:{sentence['relevance_gold']}")
            for citation in sentence["citations"]:
                if citation["evidence_relation_gold"]:
                    labels.add(f"evidence:{citation['evidence_relation_gold']}")
        return labels

    def human_record(self) -> dict[str, Any]:
        """Relevant original record for display only; never use as Gemini input."""
        return {"question_id": self.question_id, "question": self.question,
                "expert_curated_answer": self.expert_reference,
                "machine_generated_answer": {"method_id": self.method_id, "answer": self.candidate_answer,
                "is_answer_accurate": self.answer_accuracy_gold, "answer_sentences": self.sentences}}


@dataclass(frozen=True)
class WorkbenchPrediction:
    example_id: str
    data: dict[str, Any] | None
    raw_output: str
    malformed_error: str | None = None


def load_workbench_answers(path: str | Path) -> list[WorkbenchAnswer]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    answers: list[WorkbenchAnswer] = []
    for record in records:
        for method_id, answer in record["machine_generated_answers"].items():
            sentences = []
            for item in answer.get("answer_sentences") or []:
                citations = []
                for position, citation in enumerate(item.get("citation_assessment") or []):
                    citations.append({"citation_id": f"{item.get('answer_sentence_id')}:{position}",
                                      "cited_pmid": str(citation.get("cited_pmid", "")),
                                      "evidence_support": citation.get("evidence_support"),
                                      "evidence_relation_gold": _normal(citation.get("evidence_relation"), EVIDENCE_LABELS)})
                sentences.append({"answer_sentence_id": str(item.get("answer_sentence_id")),
                                  "answer_sentence": str(item.get("answer_sentence", "")),
                                  "relevance_gold": _normal(item.get("answer_sentence_relevance"), RELEVANCE_LABELS),
                                  "citations": citations})
            answers.append(WorkbenchAnswer(str(record["question_id"]), str(method_id), str(record["question"]),
                                           str(answer["answer"]), _normal(answer.get("is_answer_accurate"), ANSWER_LABELS), tuple(sentences),
                                           str(record["expert_curated_answer"]) if record.get("expert_curated_answer") else None))
    return answers


def _normal(value: Any, allowed: set[str]) -> str | None:
    value = value.strip().lower() if isinstance(value, str) else None
    return value if value in allowed else None


def select_representative_answers(answers: list[WorkbenchAnswer], limit: int = 5) -> tuple[list[WorkbenchAnswer], set[str]]:
    """Greedily maximize distinct human-label coverage, with stable ID tie-breaking."""
    if not 1 <= limit <= 5:
        raise ValueError("limit must be between 1 and 5")
    remaining = sorted(answers, key=lambda item: (int(item.question_id) if item.question_id.isdigit() else item.question_id, item.method_id))
    selected: list[WorkbenchAnswer] = []; seen: set[str] = set()
    while remaining and len(selected) < limit:
        choice = max(remaining, key=lambda item: (len(item.coverage_labels() - seen), tuple(-ord(c) for c in item.example_id)))
        selected.append(choice); remaining.remove(choice); seen |= choice.coverage_labels()
    all_labels = set().union(*(item.coverage_labels() for item in answers)) if answers else set()
    return selected, all_labels - seen


def random_answer_per_question(answers: list[WorkbenchAnswer], seed: int) -> list[WorkbenchAnswer]:
    """Choose one stable pseudo-random machine answer for each Question ID.

    A seed makes the displayed candidate answer stable while a user is selecting
    rows; changing the seed deliberately produces a different candidate pool.
    """
    grouped: dict[str, list[WorkbenchAnswer]] = {}
    for answer in answers:
        grouped.setdefault(answer.question_id, []).append(answer)
    selected = []
    for question_id, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda item: item.method_id)
        selected.append(Random(f"{seed}:{question_id}").choice(ordered))
    return sorted(selected, key=lambda item: (int(item.question_id) if item.question_id.isdigit() else item.question_id))


def select_balanced_answer_accuracy_answers(
    answers: list[WorkbenchAnswer],
    seed: int,
    target_counts: dict[str, int] | None = None,
    allowed_question_ids: set[str] | None = None,
    excluded_question_ids: set[str] | None = None,
) -> list[WorkbenchAnswer]:
    """Select one reproducible machine answer per question with requested gold counts.

    The chosen answer method is random only within methods bearing the requested
    answer-accuracy gold label. A small backtracking search preserves distinct
    Question IDs across label groups rather than silently changing the balance.
    """
    target_counts = target_counts or {"yes": 5, "no": 5}
    if set(target_counts) - ANSWER_LABELS or any(not isinstance(count, int) or count < 0 for count in target_counts.values()):
        raise ValueError("target_counts must contain non-negative yes/no counts")
    excluded_question_ids = excluded_question_ids or set()
    eligible: dict[str, dict[str, list[WorkbenchAnswer]]] = {label: {} for label in target_counts}
    for answer in answers:
        if allowed_question_ids is not None and answer.question_id not in allowed_question_ids:
            continue
        if answer.question_id in excluded_question_ids or answer.answer_accuracy_gold not in eligible:
            continue
        eligible[answer.answer_accuracy_gold].setdefault(answer.question_id, []).append(answer)

    rng = Random(seed)
    tasks = [label for label, count in target_counts.items() for _ in range(count)]
    # Make the label order itself reproducible, rather than giving yes/no an
    # unintentional priority when their eligible question pools overlap.
    rng.shuffle(tasks)
    choices = {label: sorted(question_ids) for label, question_ids in eligible.items()}
    for question_ids in choices.values():
        rng.shuffle(question_ids)

    def choose(index: int, used: set[str], selected: list[tuple[str, str]]) -> list[tuple[str, str]] | None:
        if index == len(tasks):
            return selected
        label = tasks[index]
        for question_id in choices[label]:
            if question_id not in used:
                found = choose(index + 1, used | {question_id}, [*selected, (label, question_id)])
                if found is not None:
                    return found
        return None

    selected_pairs = choose(0, set(), [])
    if selected_pairs is None:
        available = {label: len(question_ids) for label, question_ids in eligible.items()}
        raise ValueError(f"cannot select distinct questions for requested answer-accuracy balance {target_counts}; available question counts: {available}")
    selected: list[WorkbenchAnswer] = []
    for label, question_id in selected_pairs:
        methods = sorted(eligible[label][question_id], key=lambda item: item.method_id)
        selected.append(rng.choice(methods))
    return selected


def load_frozen_test_set(path: str | Path, answers: list[WorkbenchAnswer]) -> list[WorkbenchAnswer]:
    """Resolve the committed frozen ID manifest against a local MedAESQA copy."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    records = manifest.get("records")
    return _resolve_saved_records(records, answers, "saved batch")


def load_evaluation_split(path: str | Path, split_name: str, answers: list[WorkbenchAnswer]) -> tuple[list[WorkbenchAnswer], dict[str, Any]]:
    """Resolve one named ten-record evaluation split from the shared manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or split_name not in splits or not isinstance(splits[split_name], dict):
        raise ValueError(f"unknown evaluation split: {split_name}")
    split = splits[split_name]
    selected = _resolve_saved_records(split.get("records"), answers, split_name)
    if len(selected) != MAX_BATCH_SIZE:
        raise ValueError(f"{split_name} must contain exactly {MAX_BATCH_SIZE} records; found {len(selected)}")
    target_counts = split.get("answer_accuracy_target_counts")
    if target_counts is not None:
        observed = Counter(answer.answer_accuracy_gold for answer in selected)
        if observed != Counter(target_counts):
            raise ValueError(f"{split_name} answer-accuracy counts {dict(observed)} do not match required {target_counts}")
    return selected, {"selection_seed": manifest.get("selection_seed"), **split}


def load_all_evaluation_splits(path: str | Path, answers: list[WorkbenchAnswer]) -> dict[str, tuple[list[WorkbenchAnswer], dict[str, Any]]]:
    """Load and verify the three disjoint ten-record Verifier splits."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("evaluation split manifest has no splits object")
    required = ("prompt-development", "validation", "final-test")
    loaded = {name: load_evaluation_split(path, name, answers) for name in required}
    question_ids = [answer.question_id for selected, _ in loaded.values() for answer in selected]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("evaluation splits share one or more question IDs")
    return loaded


def _resolve_saved_records(records: Any, answers: list[WorkbenchAnswer], label: str) -> list[WorkbenchAnswer]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} manifest must contain at least one record")
    if len(records) > MAX_BATCH_SIZE:
        raise ValueError(f"{label} exceeds the {MAX_BATCH_SIZE}-answer workbench cap")
    by_id = {answer.example_id: answer for answer in answers}
    selected: list[WorkbenchAnswer] = []
    seen_questions: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("frozen test-set record must be an object")
        question_id, method_id = str(record.get("question_id", "")), str(record.get("method_id", ""))
        example_id = f"{question_id}:{method_id}"
        if example_id not in by_id:
            raise ValueError(f"{label} record {example_id} is not present in the local MedAESQA data")
        if question_id in seen_questions:
            raise ValueError(f"{label} repeats question_id {question_id}")
        selected.append(by_id[example_id])
        seen_questions.add(question_id)
    return selected


def resolve_answer(answers: list[WorkbenchAnswer], question_id: str, method_id: str) -> WorkbenchAnswer:
    """Find one exact MedAESQA machine answer from typed identifiers."""
    question_id, method_id = question_id.strip(), method_id.strip().upper()
    for answer in answers:
        if answer.question_id == question_id and answer.method_id == method_id:
            return answer
    raise ValueError(f"No MedAESQA machine answer exists for question_id={question_id!r}, method_id={method_id!r}")


def sanitize_prompt_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Enforce the Gemini input allowlist at the final serialization boundary."""
    generated = payload.get("machine_generated_answer")
    generated = generated if isinstance(generated, dict) else {}
    sentences = generated.get("answer_sentences")
    clean_sentences = []
    for sentence in sentences if isinstance(sentences, list) else []:
        if not isinstance(sentence, dict):
            continue
        citations = sentence.get("citations")
        clean_citations = []
        for citation in citations if isinstance(citations, list) else []:
            if not isinstance(citation, dict):
                continue
            clean_citations.append({
                "citation_id": str(citation.get("citation_id", "")),
                "cited_pmid": str(citation.get("cited_pmid", "")),
                "evidence_support": citation.get("evidence_support"),
            })
        clean_sentences.append({
            "answer_sentence_id": str(sentence.get("answer_sentence_id", "")),
            "answer_sentence": str(sentence.get("answer_sentence", "")),
            "citations": clean_citations,
        })
    return {
        "question_id": str(payload.get("question_id", "")),
        "question": str(payload.get("question", "")),
        "machine_generated_answer": {
            "method_id": str(generated.get("method_id", "")),
            "answer": str(generated.get("answer", "")),
            "answer_sentences": clean_sentences,
        },
    }


def _revised_prompt_for(payload: dict[str, Any]) -> str:
    sanitized = sanitize_prompt_input(payload)
    return """You are a MedAESQA verifier. Return valid JSON only: no Markdown, code fences, or text outside the JSON object.

The input contains a question and a complete machine-generated answer with sentences and citations. It deliberately excludes expert answers, human labels, gold fields, scores, and annotation metadata. Use only the supplied question, answer, sentence text, citation metadata, and evidence_support excerpts.

Assess internally in this order: read the question and complete answer; identify what the question asks; assess whole-answer acceptability and factual accuracy; assess each sentence's relevance in the context of the complete answer; then assess each citation with a non-null evidence_support excerpt against its attached claim. Keep these three judgments separate. Do not return private reasoning; give only the brief reasons requested in the JSON schema.

Answer accuracy:
This is a whole-answer judgment of whether the answer is acceptable, including whether it answers the question at least partially and whether it contains a material factual error.

- Predict "yes" when the answer provides meaningful information responsive to at least one part of the question and does not contain a material factual error that makes the answer unacceptable. The answer may be incomplete; do not require it to cover every possible detail or option.
- Predict "no" when the answer does not meaningfully answer any part of the question, or when a material false or misleading claim makes the answer unacceptable.
- Read the question as written. Do not silently add a narrower requirement the question does not state. For example, do not assume that a question about a "drug interaction with high blood pressure" specifically asks about an adverse interaction unless the wording or context establishes that.
- If the question has more than one reasonable interpretation, do not treat a reasonable answer to one interpretation as wrong solely because you assumed another. Briefly note the ambiguity if it affects your judgment.
- A statement that the provided context lacks information is not, by itself, a meaningful answer to the medical question. Judge the complete response: it may still be acceptable if it also gives relevant, accurate information.
- Do not mark an answer "no" solely because it omits details, gives a partial answer, has weak citations, has a null evidence excerpt, or uses imperfect formatting. Missing evidence is not proof a claim is false.
- Do not mark an answer "yes" solely because some individual statements are true if the answer as a whole does not meaningfully address any part of the question or includes a material false claim.
- Use general medical knowledge as appropriate, but do not invent facts or requirements. Briefly identify the main reason for the label.

Sentence relevance:
Choose one label for each sentence based on its role in the complete answer:

- "required": The assertion is necessary for the answer to be complete.
- "borderline": The assertion is relevant or useful to know, but the answer would still be complete without it.
- "unnecessary": The assertion is not needed and does not meaningfully help answer the question, such as generic information or repetition. An assertion can also be unnecessary if it recommends something that does not fit the situation described in the question.
- "inappropriate": Following the assertion could harm or mislead the patient.

For the required-versus-borderline decision, ask internally: “If this sentence were removed, would the answer lose information needed to answer the question completely, or only useful extra context?” Use "required" for the former and "borderline" for the latter.

Do not label a sentence "inappropriate" merely because it is off-topic, unsupported, or inaccurate. Use that label when the assertion could harm or mislead the patient if followed. Do not label a sentence "required" merely because it sounds cautious or medically relevant. Let the wording of the question determine what information is needed.

Evidence relation:
Use one of these labels for each citation that has a non-null evidence_support excerpt:

- "supporting": The excerpt directly supports or clearly implies the attached claim.
- "contradicting": The excerpt states something that conflicts with the claim or gives its opposite.
- "neutral": The excerpt is about the same topic but does not establish whether the claim is true or false.
- "not relevant": The excerpt does not address the attached claim.
- "invalid citation": Use only when the supplied input establishes that the citation itself is invalid or unusable. A missing excerpt, weak support, or disagreement with the claim does not by itself establish an invalid citation.

Assess meaning, not just matching words. Check the claim’s important details, including who or what it concerns, the outcome, population, scope, direction, and certainty. A related topic or study aim without a result is not automatically support. Do not infer causation from association.

The paper defines evidence relations for the cited document. You receive only an evidence_support excerpt, so base your prediction on that excerpt and do not claim to have checked the rest of the document. If the excerpt supports the main claim but leaves out a secondary detail, do not reject it for that omission alone.

If evidence_support is null, do not invent or infer an evidence relation from the missing excerpt. Omit that citation from citation_assessments. In particular, null evidence_support alone does not mean "invalid citation." If no citation for a sentence has a non-null evidence_support excerpt, return an empty citation_assessments array for that sentence.

Reasons:
Keep each reason brief and tied to the question, candidate sentence, or supplied evidence excerpt. For citation reasons, identify the relevant claim or excerpt detail. Do not provide private chain-of-thought.

Output:
Preserve question_id, method_id, answer_sentence_id, citation_id, and cited_pmid exactly. Include every supplied sentence in its original order. Include one citation assessment for each citation with a non-null evidence_support excerpt. Use exactly these keys and no others:
{
  "question_id": "string",
  "method_id": "string",
  "answer_accuracy": {"prediction": "yes|no", "reason": "string"},
  "sentences": [
    {
      "answer_sentence_id": "string",
      "relevance_prediction": "required|borderline|unnecessary|inappropriate",
      "relevance_reason": "string",
      "citation_assessments": [
        {
          "citation_id": "string",
          "cited_pmid": "string",
          "evidence_relation_prediction": "supporting|contradicting|neutral|not relevant|invalid citation",
          "reason": "string"
        }
      ]
    }
  ]
}

INPUT JSON:
""" + json.dumps(sanitized, ensure_ascii=False)


def prompt_for(payload: dict[str, Any], prompt_version: str = PROMPT_VERSION) -> str:
    """Render the single retained Verifier prompt with sanitized input."""
    if prompt_version == REVISED_PROMPT_VERSION:
        return _revised_prompt_for(payload)
    raise ValueError(f"Unknown Verifier prompt version: {prompt_version}")


def parse_workbench_prediction(answer: WorkbenchAnswer, raw_output: str) -> WorkbenchPrediction:
    try:
        data = json.loads(_unfence(raw_output))
        if not isinstance(data, dict) or data.get("question_id") != answer.question_id or data.get("method_id") != answer.method_id:
            raise ValueError("question_id or method_id does not match selected record")
        accuracy = data.get("answer_accuracy", {})
        if _normal(accuracy.get("prediction"), ANSWER_LABELS) is None or not isinstance(accuracy.get("reason"), str):
            raise ValueError("invalid answer_accuracy prediction")
        expected_sentences = {item["answer_sentence_id"]: item for item in answer.sentences}
        returned = data.get("sentences")
        if not isinstance(returned, list) or {item.get("answer_sentence_id") for item in returned} != set(expected_sentences):
            raise ValueError("sentence IDs do not match selected record")
        for sentence in returned:
            if _normal(sentence.get("relevance_prediction"), RELEVANCE_LABELS) is None or not isinstance(sentence.get("relevance_reason"), str):
                raise ValueError("invalid relevance prediction")
            expected_citations = {c["citation_id"]: c for c in expected_sentences[sentence["answer_sentence_id"]]["citations"] if c["evidence_support"] is not None}
            observed = sentence.get("citation_assessments")
            if not isinstance(observed, list) or {c.get("citation_id") for c in observed} != set(expected_citations):
                raise ValueError("citation IDs do not match evidence-bearing selected citations")
            for citation in observed:
                expected = expected_citations[citation["citation_id"]]
                if citation.get("cited_pmid") != expected["cited_pmid"] or _normal(citation.get("evidence_relation_prediction"), EVIDENCE_LABELS) is None or not isinstance(citation.get("reason"), str):
                    raise ValueError("invalid citation assessment")
        return WorkbenchPrediction(answer.example_id, data, raw_output)
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        return WorkbenchPrediction(answer.example_id, None, raw_output, str(exc))


def _unfence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        if newline < 0 or not text.endswith("```"):
            raise ValueError("unterminated JSON code fence")
        text = text[newline + 1:-3].strip()
    return text


def compare(answer: WorkbenchAnswer, prediction: WorkbenchPrediction) -> dict[str, Any]:
    """Keep answer, relevance, and evidence comparisons separate; unavailable stays skipped."""
    result = {"answer_accuracy": [], "sentence_relevance": [], "evidence_relation": [], "malformed": bool(prediction.malformed_error)}
    if prediction.data is None:
        return result
    data = prediction.data
    result["answer_accuracy"].append(_pair(answer.answer_accuracy_gold, data["answer_accuracy"]["prediction"], answer.example_id))
    by_id = {item["answer_sentence_id"]: item for item in data["sentences"]}
    for sentence in answer.sentences:
        predicted = by_id[sentence["answer_sentence_id"]]
        result["sentence_relevance"].append(_pair(sentence["relevance_gold"], predicted["relevance_prediction"], sentence["answer_sentence_id"]))
        citations = {item["citation_id"]: item for item in predicted["citation_assessments"]}
        for citation in sentence["citations"]:
            if citation["evidence_support"] is None:
                result["evidence_relation"].append(_pair(None, None, citation["citation_id"]))
            else:
                result["evidence_relation"].append(_pair(
                    citation["evidence_relation_gold"],
                    citations[citation["citation_id"]]["evidence_relation_prediction"],
                    citation["citation_id"],
                ))
    return result


def _pair(gold: str | None, predicted: str | None, identifier: str) -> dict[str, Any]:
    return {"id": identifier, "gold": gold, "predicted": predicted, "status": "skipped" if gold is None else ("correct" if gold == predicted else "incorrect")}


def cache_key(answer: WorkbenchAnswer, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    source = json.dumps({"model": model, "version": prompt_version, "prompt": prompt_for(answer.prompt_input(), prompt_version),
                         "input": answer.prompt_input()}, sort_keys=True)
    return hashlib.sha256(source.encode()).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists(): return {}
    return {row["key"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")


def append_run_log(path: Path, run: dict[str, Any]) -> None:
    """Persist selected IDs and configuration without saving human labels in the prompt input."""
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run) + "\n")


def task_counts(comparisons: list[dict[str, Any]], task: str) -> dict[str, int]:
    pairs = [pair for result in comparisons for pair in result[task]]
    return dict(Counter(pair["status"] for pair in pairs))


def task_error_rate(comparisons: list[dict[str, Any]], task: str) -> float | None:
    """Wrong / scored. Skipped and malformed items never enter the denominator."""
    counts = task_counts(comparisons, task)
    scored = counts.get("correct", 0) + counts.get("incorrect", 0)
    return None if scored == 0 else counts.get("incorrect", 0) / scored


def task_agreement_statistics(comparisons: list[dict[str, Any]], task: str) -> dict[str, Any]:
    """Calculate task-local paired agreement, confusion counts, and unweighted kappa.

    Only a pair with labels valid for this task enters agreement or kappa. Skipped
    pairs and malformed responses remain visible but never enter the denominator.
    """
    if task not in TASK_LABELS:
        raise ValueError(f"unknown task: {task}")
    labels = TASK_LABELS[task]
    confusion = {gold: {prediction: 0 for prediction in labels} for gold in labels}
    pairs = [pair for result in comparisons for pair in result[task]]
    valid_pairs: list[tuple[str, str]] = []
    missing_gold = missing_prediction = invalid_gold = invalid_prediction = skipped = 0
    for pair in pairs:
        gold, prediction = pair.get("gold"), pair.get("predicted")
        if pair.get("status") == "skipped" or (gold is None and prediction is None):
            skipped += 1
        if gold is None:
            missing_gold += 1
            continue
        if gold not in labels:
            invalid_gold += 1
            continue
        if prediction is None:
            missing_prediction += 1
            continue
        if prediction not in labels:
            invalid_prediction += 1
            continue
        valid_pairs.append((gold, prediction))
        confusion[gold][prediction] += 1
    scored = len(valid_pairs)
    correct = sum(gold == prediction for gold, prediction in valid_pairs)
    malformed_responses = sum(bool(result.get("malformed")) for result in comparisons)
    kappa, kappa_reason = _cohen_kappa(valid_pairs, labels)
    return {
        "labels": labels,
        "scored_pairs": scored,
        "correct": correct,
        "incorrect": scored - correct,
        "exact_agreement": _safe_fraction(correct, scored),
        "error_rate": _safe_fraction(scored - correct, scored),
        "kappa": kappa,
        "kappa_reason": kappa_reason,
        "skipped_pairs": skipped,
        "unscored_pairs": len(pairs) - scored,
        "missing_gold_pairs": missing_gold,
        "missing_prediction_pairs": missing_prediction,
        "invalid_gold_label_pairs": invalid_gold,
        "invalid_prediction_label_pairs": invalid_prediction,
        "malformed_responses": malformed_responses,
        "confusion_matrix": confusion,
    }


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _cohen_kappa(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> tuple[float | None, str | None]:
    if not pairs:
        return None, "Not evaluated: no valid gold/prediction pairs."
    gold, prediction = zip(*pairs)
    try:
        from sklearn.metrics import cohen_kappa_score
        value = float(cohen_kappa_score(gold, prediction, labels=list(labels)))
    except ImportError:
        # The project declares scikit-learn; this exact fallback keeps offline
        # scoring available in lightweight environments where it is not installed.
        total = len(pairs)
        observed = sum(actual == predicted for actual, predicted in pairs) / total
        expected = sum((gold.count(label) / total) * (prediction.count(label) / total) for label in labels)
        value = (observed - expected) / (1 - expected) if expected != 1 else math.nan
    if math.isnan(value):
        return None, "Undefined: expected chance agreement is one, so Cohen’s kappa denominator is zero."
    return value, None


def mismatch_diagnostic(task: str, status: str, malformed: bool = False) -> str | None:
    """A compact heuristic diagnostic, never a claimed human rationale."""
    if malformed:
        return "Missing Context"
    if status != "incorrect":
        return None
    if task == "sentence_relevance":
        return "Label Disagreement"
    return "Reasoning Failure"


def export_review_summary(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Create a downloadable, task-organized comparison export without model calls."""
    exported: dict[str, list[dict[str, Any]]] = {
        "answer_accuracy": [], "sentence_relevance": [], "evidence_relation": [],
    }
    for result in results:
        answer: WorkbenchAnswer = result["answer"]
        prediction: WorkbenchPrediction = result["prediction"]
        comparison = compare(answer, prediction)
        common = {
            "question_id": answer.question_id,
            "method_id": answer.method_id,
            "question": answer.question,
            "machine_generated_answer": answer.candidate_answer,
        }
        if prediction.data is None:
            for task in exported:
                exported[task].append({
                    **common,
                    "status": "malformed",
                    "human_label": None,
                    "gemini_prediction": None,
                    "gemini_justification": None,
                    "malformed_error": prediction.malformed_error,
                })
            continue

        accuracy = prediction.data["answer_accuracy"]
        accuracy_pair = comparison["answer_accuracy"][0]
        exported["answer_accuracy"].append({
            **common,
            "human_label": accuracy_pair["gold"],
            "gemini_prediction": accuracy_pair["predicted"],
            "gemini_justification": accuracy["reason"],
            "status": accuracy_pair["status"],
        })

        relevance_pairs = {pair["id"]: pair for pair in comparison["sentence_relevance"]}
        evidence_pairs = {pair["id"]: pair for pair in comparison["evidence_relation"]}
        response_sentences = {item["answer_sentence_id"]: item for item in prediction.data["sentences"]}
        for sentence in answer.sentences:
            sentence_id = sentence["answer_sentence_id"]
            response_sentence = response_sentences[sentence_id]
            relevance_pair = relevance_pairs[sentence_id]
            sentence_common = {
                **common,
                "answer_sentence_id": sentence_id,
                "answer_sentence": sentence["answer_sentence"],
            }
            exported["sentence_relevance"].append({
                **sentence_common,
                "human_label": relevance_pair["gold"],
                "gemini_prediction": relevance_pair["predicted"],
                "gemini_justification": response_sentence["relevance_reason"],
                "status": relevance_pair["status"],
            })
            response_citations = {item["citation_id"]: item for item in response_sentence["citation_assessments"]}
            for citation in sentence["citations"]:
                citation_id = citation["citation_id"]
                evidence_pair = evidence_pairs[citation_id]
                response_citation = response_citations.get(citation_id)
                exported["evidence_relation"].append({
                    **sentence_common,
                    "citation_id": citation_id,
                    "cited_pmid": citation["cited_pmid"],
                    "evidence_support": citation["evidence_support"],
                    "human_label": evidence_pair["gold"],
                    "gemini_prediction": response_citation["evidence_relation_prediction"] if response_citation else None,
                    "gemini_justification": response_citation["reason"] if response_citation else None,
                    "status": evidence_pair["status"],
                })
    return exported
