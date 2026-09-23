"""Small, offline-friendly evaluation helpers for the FaithfulMed Verifier."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabeledVerifierExample:
    """One source/explanation pair with an independent human label."""

    example_id: str
    source_text: str
    candidate_text: str
    gold_unsupported: bool | None


@dataclass(frozen=True)
class VerifierPrediction:
    """One saved Verifier prediction or one malformed model response."""

    example_id: str
    predicted_unsupported: bool | None
    raw_output: Any = None
    malformed_error: str | None = None


@dataclass(frozen=True)
class VerifierMetrics:
    """Binary unsupported-claim metrics and IDs retained for review."""

    status: str
    positive_class: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_ids: tuple[str, ...]
    false_negative_ids: tuple[str, ...]
    malformed_ids: tuple[str, ...]
    missing_gold_ids: tuple[str, ...]


def parse_prediction(example_id: str, raw_output: str | dict[str, Any]) -> VerifierPrediction:
    """Validate the current four-field Verifier response without inventing labels."""

    try:
        data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        if not isinstance(data, dict):
            raise ValueError("output must be a JSON object")
        required = {"faithful", "unsupported_claims", "omissions", "reading_level_ok"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"missing field(s): {', '.join(sorted(missing))}")
        if not isinstance(data["faithful"], bool) or not isinstance(data["reading_level_ok"], bool):
            raise ValueError("faithful and reading_level_ok must be booleans")
        for field_name in ("unsupported_claims", "omissions"):
            values = data[field_name]
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{field_name} must be a list of non-empty strings")
        has_any_faithfulness_error = bool(data["unsupported_claims"] or data["omissions"])
        if data["faithful"] == has_any_faithfulness_error:
            raise ValueError("faithful conflicts with unsupported_claims or omissions")
        return VerifierPrediction(
            example_id=example_id,
            predicted_unsupported=bool(data["unsupported_claims"]),
            raw_output=data,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return VerifierPrediction(
            example_id=example_id,
            predicted_unsupported=None,
            raw_output=raw_output,
            malformed_error=str(exc),
        )


async def generate_predictions(
    examples: Iterable[LabeledVerifierExample],
    run_one: Callable[[str, str], Awaitable[str]],
) -> list[VerifierPrediction]:
    """Run an injected Verifier callable; scoring can be repeated later without this step."""

    predictions = []
    for example in examples:
        raw_output = await run_one(example.source_text, example.candidate_text)
        predictions.append(parse_prediction(example.example_id, raw_output))
    return predictions


def save_predictions(path: str | Path, predictions: Iterable[VerifierPrediction]) -> None:
    """Save predictions as JSON Lines so they can be scored again without a model call."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), sort_keys=True) + "\n")


def load_predictions(path: str | Path) -> list[VerifierPrediction]:
    """Load predictions previously written by save_predictions()."""

    predictions = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                predictions.append(VerifierPrediction(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid prediction JSON on line {line_number}: {path}") from exc
    return predictions


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def score_predictions(
    examples: Iterable[LabeledVerifierExample],
    predictions: Iterable[VerifierPrediction],
) -> VerifierMetrics:
    """Score unsupported-claim detection against independent human labels."""

    example_list = list(examples)
    prediction_list = list(predictions)
    gold_by_id = {example.example_id: example for example in example_list}
    prediction_by_id = {prediction.example_id: prediction for prediction in prediction_list}
    if len(gold_by_id) != len(example_list) or len(prediction_by_id) != len(prediction_list):
        raise ValueError("example IDs must be unique")
    missing_predictions = sorted(gold_by_id.keys() - prediction_by_id.keys())
    unexpected_predictions = sorted(prediction_by_id.keys() - gold_by_id.keys())
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            f"gold/prediction ID mismatch; missing predictions={missing_predictions}, "
            f"unexpected predictions={unexpected_predictions}"
        )

    tp = fp = tn = fn = 0
    false_positive_ids: list[str] = []
    false_negative_ids: list[str] = []
    malformed_ids: list[str] = []
    missing_gold_ids: list[str] = []
    evaluated_count = 0

    for example_id, example in gold_by_id.items():
        prediction = prediction_by_id[example_id]
        if example.gold_unsupported is None:
            missing_gold_ids.append(example_id)
            continue
        if prediction.malformed_error or prediction.predicted_unsupported is None:
            malformed_ids.append(example_id)
            continue
        evaluated_count += 1
        if prediction.predicted_unsupported and example.gold_unsupported:
            tp += 1
        elif prediction.predicted_unsupported:
            fp += 1
            false_positive_ids.append(example_id)
        elif example.gold_unsupported:
            fn += 1
            false_negative_ids.append(example_id)
        else:
            tn += 1

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = round(2 * precision * recall / (precision + recall), 6)

    return VerifierMetrics(
        status="evaluated" if evaluated_count else "not_evaluated",
        positive_class="human label says the generated claim or explanation is unsupported",
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_ids=tuple(false_positive_ids),
        false_negative_ids=tuple(false_negative_ids),
        malformed_ids=tuple(malformed_ids),
        missing_gold_ids=tuple(missing_gold_ids),
    )
