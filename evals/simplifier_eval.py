"""Evaluation helpers for the FaithfulMed Simplifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import textstat


@dataclass(frozen=True)
class SimplifierExample:
    """One source text and its Simplifier output."""

    example_id: str
    source_text: str
    candidate_text: str


@dataclass(frozen=True)
class SimplifierExampleMetrics:
    """Readability metrics for one Simplifier example."""

    example_id: str
    original_fk_grade: float
    simplified_fk_grade: float
    fk_grade_reduction: float
    original_smog: float
    simplified_smog: float
    smog_reduction: float
    original_jargon_density: float
    simplified_jargon_density: float
    jargon_reduction: float
    original_word_count: int
    simplified_word_count: int
    meets_readability_target: bool


@dataclass(frozen=True)
class SimplifierMetrics:
    """Aggregate Simplifier evaluation results."""

    status: str
    n: int
    mean_original_fk_grade: float | None
    mean_simplified_fk_grade: float | None
    mean_fk_grade_reduction: float | None
    mean_original_smog: float | None
    mean_simplified_smog: float | None
    mean_smog_reduction: float | None
    mean_original_jargon_density: float | None
    mean_simplified_jargon_density: float | None
    mean_jargon_reduction: float | None
    mean_original_word_count: float | None
    mean_simplified_word_count: float | None
    readability_target_rate: float | None
    malformed_ids: tuple[str, ...]


def readability_scores(text: str) -> dict[str, float | int]:
    """Calculate readability metrics for one text."""

    words = max(textstat.lexicon_count(text, removepunct=True), 1)

    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "jargon_density": textstat.difficult_words(text) / words,
        "word_count": words,
    }


def evaluate_example(
    example: SimplifierExample,
) -> SimplifierExampleMetrics:
    """Evaluate one Simplifier output against its source text."""

    if not isinstance(example.source_text, str) or not example.source_text.strip():
        raise ValueError("Simplifier source text is empty")
    if not isinstance(example.candidate_text, str) or not example.candidate_text.strip():
        raise ValueError("Simplifier output is empty")

    original = readability_scores(example.source_text)
    simplified = readability_scores(example.candidate_text)

    return SimplifierExampleMetrics(
        example_id=example.example_id,
        original_fk_grade=original["flesch_kincaid_grade"],
        simplified_fk_grade=simplified["flesch_kincaid_grade"],
        fk_grade_reduction=(
            original["flesch_kincaid_grade"]
            - simplified["flesch_kincaid_grade"]
        ),
        original_smog=original["smog_index"],
        simplified_smog=simplified["smog_index"],
        smog_reduction=(
            original["smog_index"]
            - simplified["smog_index"]
        ),
        original_jargon_density=original["jargon_density"],
        simplified_jargon_density=simplified["jargon_density"],
        jargon_reduction=(
            original["jargon_density"]
            - simplified["jargon_density"]
        ),
        original_word_count=original["word_count"],
        simplified_word_count=simplified["word_count"],
        meets_readability_target=(
            simplified["flesch_kincaid_grade"] <= 8.0
        ),
    )


def score_examples(
    examples: Iterable[SimplifierExample],
) -> list[SimplifierExampleMetrics]:
    """Evaluate all Simplifier examples."""

    results = []

    for example in examples:
        try:
            results.append(evaluate_example(example))
        except ValueError:
            continue

    return results


def _mean(values: list[float | int]) -> float | None:
    """Return a mean, or ``None`` when no valid values were supplied."""

    return sum(values) / len(values) if values else None


def summarize_examples(examples: Iterable[SimplifierExample]) -> SimplifierMetrics:
    """Summarize readability results while preserving malformed example IDs.

    This is intentionally a readability-only summary. Faithfulness is checked by
    the downstream Verifier; a shorter or easier output is not proof that it
    preserved the medical content.
    """

    results: list[SimplifierExampleMetrics] = []
    malformed_ids: list[str] = []
    for example in examples:
        try:
            results.append(evaluate_example(example))
        except ValueError:
            malformed_ids.append(example.example_id)

    if not results:
        return SimplifierMetrics(
            status="not_evaluated",
            n=0,
            mean_original_fk_grade=None,
            mean_simplified_fk_grade=None,
            mean_fk_grade_reduction=None,
            mean_original_smog=None,
            mean_simplified_smog=None,
            mean_smog_reduction=None,
            mean_original_jargon_density=None,
            mean_simplified_jargon_density=None,
            mean_jargon_reduction=None,
            mean_original_word_count=None,
            mean_simplified_word_count=None,
            readability_target_rate=None,
            malformed_ids=tuple(malformed_ids),
        )

    return SimplifierMetrics(
        status="partial" if malformed_ids else "ok",
        n=len(results),
        mean_original_fk_grade=_mean([item.original_fk_grade for item in results]),
        mean_simplified_fk_grade=_mean([item.simplified_fk_grade for item in results]),
        mean_fk_grade_reduction=_mean([item.fk_grade_reduction for item in results]),
        mean_original_smog=_mean([item.original_smog for item in results]),
        mean_simplified_smog=_mean([item.simplified_smog for item in results]),
        mean_smog_reduction=_mean([item.smog_reduction for item in results]),
        mean_original_jargon_density=_mean([item.original_jargon_density for item in results]),
        mean_simplified_jargon_density=_mean([item.simplified_jargon_density for item in results]),
        mean_jargon_reduction=_mean([item.jargon_reduction for item in results]),
        mean_original_word_count=_mean([item.original_word_count for item in results]),
        mean_simplified_word_count=_mean([item.simplified_word_count for item in results]),
        readability_target_rate=sum(item.meets_readability_target for item in results) / len(results),
        malformed_ids=tuple(malformed_ids),
    )
