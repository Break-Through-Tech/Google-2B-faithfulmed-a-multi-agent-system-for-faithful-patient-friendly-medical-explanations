"""Verifier-only MedAESQA workbench helpers; all functions are offline except an injected caller."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

PROMPT_VERSION = "medaesqa-answer-v2"
ANSWER_LABELS = {"yes", "no"}
RELEVANCE_LABELS = {"required", "borderline", "unnecessary", "inappropriate"}
EVIDENCE_LABELS = {"supporting", "contradicting", "neutral", "not relevant", "invalid citation"}


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
        """Only fields permitted in the Gemini request; no human labels or expert answer."""
        sentences = []
        for sentence in self.sentences:
            citations = []
            for citation in sentence["citations"]:
                citations.append({key: citation[key] for key in ("citation_id", "cited_pmid", "evidence_support")})
            sentences.append({"answer_sentence_id": sentence["answer_sentence_id"], "answer_sentence": sentence["answer_sentence"], "citations": citations})
        return {"question_id": self.question_id, "question": self.question, "method_id": self.method_id,
                "candidate_answer": self.candidate_answer, "sentences": sentences}

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


def prompt_for(payload: dict[str, Any]) -> str:
    return (
        "You are a MedAESQA verifier. Return JSON only—no Markdown. You receive no human labels, expert answer, or nuggets.\n\n"
        "Assess the complete candidate answer. For every supplied sentence, predict one relevance label: required, borderline, unnecessary, or inappropriate. "
        "For every cited item WITH an evidence_support excerpt, predict one relation: supporting, contradicting, neutral, not relevant, or invalid citation. "
        "For cited items with null evidence_support, return no citation prediction. Do not invent missing citations.\n\n"
        "Return exactly: {\"question_id\": string, \"method_id\": string, \"answer_accuracy\": {\"prediction\": \"yes|no\", \"reason\": string}, "
        "\"sentences\": [{\"answer_sentence_id\": string, \"relevance_prediction\": string, \"relevance_reason\": string, \"citation_assessments\": [{\"citation_id\": string, \"cited_pmid\": string, \"evidence_relation_prediction\": string, \"reason\": string}]}]}. Keep every reason brief and evidence-based; do not reveal private reasoning.\n\nINPUT:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


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
            expected_citations = {c["citation_id"]: c for c in expected_sentences[sentence["answer_sentence_id"]]["citations"] if c["evidence_support"]}
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
            if not citation["evidence_support"]:
                result["evidence_relation"].append(_pair(None, None, citation["citation_id"]))
            else:
                result["evidence_relation"].append(_pair(citation["evidence_relation_gold"], citations[citation["citation_id"]]["evidence_relation_prediction"], citation["citation_id"]))
    return result


def _pair(gold: str | None, predicted: str | None, identifier: str) -> dict[str, Any]:
    return {"id": identifier, "gold": gold, "predicted": predicted, "status": "skipped" if gold is None else ("correct" if gold == predicted else "incorrect")}


def cache_key(answer: WorkbenchAnswer, model: str) -> str:
    source = json.dumps({"model": model, "version": PROMPT_VERSION, "input": answer.prompt_input()}, sort_keys=True)
    return hashlib.sha256(source.encode()).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists(): return {}
    return {row["key"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")


def task_counts(comparisons: list[dict[str, Any]], task: str) -> dict[str, int]:
    pairs = [pair for result in comparisons for pair in result[task]]
    return dict(Counter(pair["status"] for pair in pairs))
