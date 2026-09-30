"""Offline MedAESQA scoring for the FaithfulMed Verifier.

Label sources: Gupta, Bartels, and Demner-Fushman (2025), and the upstream
``src/medaesqa_eval.py``. Upstream counts ``is_answer_accurate == 'yes'`` as
an acceptable answer and reports citation support and contradiction separately.
This module deliberately does not turn relevance, neutral, not-relevant,
missing, or invalid citations into factual-support labels.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ANSWER_ACCURACY_LABELS = {"yes": True, "no": False}
CITATION_SUPPORT_LABELS = {"supporting": True, "contradicting": False}


@dataclass(frozen=True)
class MedAESQAExample:
    """One answer or cited answer-sentence instance, with review provenance."""

    evaluation_level: Literal["answer", "citation"]
    question_id: str
    method_id: str
    question: str
    candidate_text: str
    gold_label: bool | None
    sentence_id: str | None = None
    citation_id: str | None = None
    evidence_snippets: tuple[str, ...] = ()
    gold_label_raw: str | None = None
    unavailable_reason: str | None = None

    @property
    def example_id(self) -> str:
        parts = [self.question_id, self.method_id]
        if self.sentence_id is not None:
            parts.append(self.sentence_id)
        if self.citation_id is not None:
            parts.append(self.citation_id)
        return ":".join(parts)


@dataclass(frozen=True)
class VerifierPrediction:
    """Saved prediction. Malformed output is never counted as a wrong label."""

    example_id: str
    predicted_label: bool | None
    raw_output: Any = None
    malformed_error: str | None = None


@dataclass(frozen=True)
class VerifierMetrics:
    status: str
    evaluation_level: str
    positive_class: str
    evaluated_count: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_ids: tuple[str, ...]
    false_negative_ids: tuple[str, ...]
    malformed_ids: tuple[str, ...]
    unavailable_gold_ids: tuple[str, ...]


def _label(value: Any, mapping: dict[str, bool]) -> tuple[bool | None, str | None, str | None]:
    if not isinstance(value, str):
        return None, None, "missing or non-string label"
    normalized = value.strip().lower()
    if normalized in mapping:
        return mapping[normalized], normalized, None
    return None, normalized, f"label {normalized!r} is not evaluable for this metric"


def load_medaesqa_examples(
    path: str | Path, *, level: Literal["answer", "citation"]
) -> list[MedAESQAExample]:
    """Flatten actual MedAESQA JSON without using labels as Verifier inputs.

    Answer-level ``yes/no`` means acceptable/not acceptable, not "contains an
    unsupported claim." At citation level only ``supporting/contradicting`` are
    binary factual-support targets. Neutral and not-relevant stay unavailable.
    """
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("MedAESQA JSON must contain a list of question records")
    examples: list[MedAESQAExample] = []
    for record in records:
        question_id, question = str(record["question_id"]), str(record["question"])
        for method_id, answer_item in record["machine_generated_answers"].items():
            if level == "answer":
                gold, raw, reason = _label(answer_item.get("is_answer_accurate"), ANSWER_ACCURACY_LABELS)
                examples.append(MedAESQAExample(
                    "answer", question_id, str(method_id), question, str(answer_item["answer"]), gold,
                    gold_label_raw=raw, unavailable_reason=reason,
                ))
                continue
            for sentence in answer_item.get("answer_sentences") or []:
                assessments = sentence.get("citation_assessment")
                if not isinstance(assessments, list):
                    continue  # There is no cited evidence pair to evaluate.
                for citation in assessments:
                    gold, raw, reason = _label(citation.get("evidence_relation"), CITATION_SUPPORT_LABELS)
                    support = citation.get("evidence_support")
                    snippets = (str(support),) if isinstance(support, str) and support.strip() else ()
                    citation_id = citation.get("cited_pmid")
                    examples.append(MedAESQAExample(
                        "citation", question_id, str(method_id), question,
                        str(sentence.get("answer_sentence", "")), gold,
                        str(sentence.get("answer_sentence_id")),
                        str(citation_id) if citation_id is not None else None,
                        snippets, raw, reason,
                    ))
    return examples


def parse_answer_prediction(example_id: str, raw_output: str | dict[str, Any]) -> VerifierPrediction:
    """Read the answer-level schema; ``faithful`` predicts answer acceptability."""
    try:
        data = json.loads(_json_text(raw_output)) if isinstance(raw_output, str) else raw_output
        if not isinstance(data, dict) or not isinstance(data.get("faithful"), bool):
            raise ValueError("answer prediction requires a JSON object with boolean faithful")
        return VerifierPrediction(example_id, data["faithful"], data)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return VerifierPrediction(example_id, None, raw_output, str(exc))


def parse_citation_prediction(example_id: str, raw_output: str | dict[str, Any]) -> VerifierPrediction:
    """Read citation-only output; an unknown verdict is malformed for binary scoring."""
    try:
        data = json.loads(_json_text(raw_output)) if isinstance(raw_output, str) else raw_output
        verdict = data.get("citation_verdict") if isinstance(data, dict) else None
        label, _, reason = _label(verdict, CITATION_SUPPORT_LABELS)
        if reason:
            raise ValueError(f"citation_verdict: {reason}")
        return VerifierPrediction(example_id, label, data)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return VerifierPrediction(example_id, None, raw_output, str(exc))


def _json_text(raw_output: str) -> str:
    """Accept JSON returned in the Markdown fence LLMs commonly add."""
    text = raw_output.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline == -1 or not text.endswith("```"):
            raise ValueError("unterminated JSON code fence")
        text = text[first_newline + 1 : -3].strip()
    return text


def save_predictions(path: str | Path, predictions: Iterable[VerifierPrediction]) -> None:
    """Save model results, allowing later scoring without another model call."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), sort_keys=True) + "\n")


def load_predictions(path: str | Path) -> list[VerifierPrediction]:
    """Load predictions previously saved by :func:`save_predictions`."""
    with Path(path).open(encoding="utf-8") as handle:
        return [VerifierPrediction(**json.loads(line)) for line in handle if line.strip()]


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def score_predictions(
    examples: Iterable[MedAESQAExample], predictions: Iterable[VerifierPrediction]
) -> VerifierMetrics:
    """Score one MedAESQA level; unavailable labels remain unavailable."""
    example_list, prediction_list = list(examples), list(predictions)
    gold_by_id = {item.example_id: item for item in example_list}
    predictions_by_id = {item.example_id: item for item in prediction_list}
    if len(gold_by_id) != len(example_list) or len(predictions_by_id) != len(prediction_list):
        raise ValueError("example IDs must be unique")
    if gold_by_id.keys() != predictions_by_id.keys():
        raise ValueError("gold/prediction ID mismatch")
    levels = {item.evaluation_level for item in example_list}
    if len(levels) > 1:
        raise ValueError("score answer and citation examples separately")
    level = next(iter(levels), "not_evaluated")
    tp = fp = tn = fn = evaluated = 0
    false_positive_ids: list[str] = []
    false_negative_ids: list[str] = []
    malformed_ids: list[str] = []
    unavailable_ids: list[str] = []
    for example_id, example in gold_by_id.items():
        prediction = predictions_by_id[example_id]
        if example.gold_label is None:
            unavailable_ids.append(example_id)
        elif prediction.malformed_error or prediction.predicted_label is None:
            malformed_ids.append(example_id)
        else:
            evaluated += 1
            if prediction.predicted_label and example.gold_label:
                tp += 1
            elif prediction.predicted_label:
                fp += 1; false_positive_ids.append(example_id)
            elif example.gold_label:
                fn += 1; false_negative_ids.append(example_id)
            else:
                tn += 1
    precision, recall = _safe_ratio(tp, tp + fp), _safe_ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None else (
        0.0 if precision + recall == 0 else round(2 * precision * recall / (precision + recall), 6)
    )
    positive = "MedAESQA answer label 'yes' (acceptable)" if level == "answer" else "MedAESQA evidence_relation 'supporting'"
    return VerifierMetrics(
        "evaluated" if evaluated else "not_evaluated", level, positive, evaluated, tp, fp, tn, fn,
        _safe_ratio(tp + tn, evaluated), precision, recall, f1, tuple(false_positive_ids),
        tuple(false_negative_ids), tuple(malformed_ids), tuple(unavailable_ids),
    )
