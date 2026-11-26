"""Dataset loading, caching, and metrics for the end-to-end system demo."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from agents.extractor import EXTRACTOR_INSTRUCTION
from agents.medical_lookup import lookup_config_from_env, reference_definitions_for_prompt
from agents.factual_evaluator import (
    FACTUAL_EVALUATOR_PROMPT,
    factual_evaluator_request,
    indexed_text_spans,
    run_factual_evaluator,
)
from agents.production_pipeline import MAX_REVISION_CYCLES, PipelineRunResult, run_production_pipeline
from agents.readability import READABILITY_INSTRUCTION
from agents.refiner import _REFINER_INSTRUCTION
from agents.simplifier import SIMPLIFIER_INSTRUCTION
from agents.verifier import PRODUCTION_VERIFIER_PROMPT
from evals.readability_eval import evaluate_single

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = ROOT / "eval_results" / "system_demo_runs.jsonl"
VALID_SPLITS = ("scratch", "dev", "test")
VALID_DATASETS = ("MTSamples", "PLABA")
PIPELINE_IMPLEMENTATION_VERSION = "system-pipeline-v7"  # v7: source-grounded closed glossary + verified final text
METRIC_IMPLEMENTATION_VERSION = "system-metrics-v4-markdown-headings"

# These exact v7 prompt snapshots cover two reviewed safety-only prompt patches:
# medication-modifier grounding and the later Readability preservation rules.
# Any other prompt change still invalidates the cache unless it is reviewed and
# explicitly added. Force-running a sample always bypasses compatibility reuse.
V7_PRE_MEDICATION_GUARD_PROMPT_HASHES = {
    "extractor": "f298200225e8d4ec236a3961ae8c54af31b058f95eac65385031ba942f991125",
    "factual_evaluator": "78a730c69f7a6b78afbd12a233ff4d4addf8b07810455c11cf3fd59554e1fba2",
    "production_verifier": "ece08a8d389e21cb3a543b98d631d7108d27dab5dc8e06b77e034573fa01d4ea",
    "readability": "bd6cf1c0da4c76f012cb98d45ec28ad5bb6bb68be764d7918b68ef67aca68101",
    "refiner": "136cc8af3b95e58391e1cdf21c759fc7f3da5ed9090f84f1545362847f378a8c",
    "simplifier": "4c734f3461c6b9b1da4a841c10102282f8c8340f7fd9bde274d8af9a5bbe7f8c",
}
V7_MEDICATION_GUARD_PROMPT_HASHES = {
    "extractor": "f298200225e8d4ec236a3961ae8c54af31b058f95eac65385031ba942f991125",
    "factual_evaluator": "78a730c69f7a6b78afbd12a233ff4d4addf8b07810455c11cf3fd59554e1fba2",
    "production_verifier": "cecc6c05fe890f686e10b224ff1573a284fc949b1b226143bf8cdcde1d323eb4",
    "readability": "b80a21aa4e04c6a4d68f3b58edb7821af869e94ea887a7297f94677d8953af32",
    "refiner": "ab9d8804fffa2863370280e1cc4d2ab732108fcc4d5d16ee58977429df974965",
    "simplifier": "71f9c7e8625b16b938553e9b2a81e9eba2f7abd4632cd0662038a11c8c822a54",
}
V7_READABILITY_REVISION_PROMPT_HASHES = {
    **V7_MEDICATION_GUARD_PROMPT_HASHES,
    "readability": "930534335916bce4ff4d000f5bcb390c52fcbf9299e972efe0168a55b92b50bc",
}
FACTUAL_LABELS = {"supported", "unsupported", "contradicted", "uncertain"}
FACTUAL_STATUSES = {"scored", "no_claims", "insufficient_source"}
FACTUAL_CLAIM_TYPES = {"patient_specific", "general_term_explanation"}
FACTUAL_EVIDENCE_TYPES = {"medical_report", "umls_definition", "both", "none"}
FACTUAL_PARSER_VERSION = "factual-parser-v2-safe-empty-reference-ids"


@dataclass(frozen=True)
class SystemExample:
    dataset: str
    split: str
    example_id: str
    source_text: str
    references: dict[str, Any]
    source_file: str

    def metadata(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "example_id": self.example_id,
            "source_file": self.source_file,
        }


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_mtsamples(split: str, root: Path = ROOT) -> list[SystemExample]:
    """Load only adjudicated MTSamples records from a split."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    folder = root / "data" / "MTSamples" / split / "adjudicated"
    examples: list[SystemExample] = []
    for path in sorted(folder.glob("*_gold.json")):
        record = _load_json(path)
        source = record.get("source_text")
        if not isinstance(source, str) or not source.strip():
            continue
        facts = record.get("facts") if isinstance(record.get("facts"), list) else []
        examples.append(SystemExample(
            dataset="MTSamples",
            split=split,
            example_id=str(record.get("example_id") or path.stem.removesuffix("_gold")),
            source_text=source.strip(),
            references={"adjudicated_facts": facts},
            source_file=str(path.relative_to(root)),
        ))
    return examples


def _ordered_text(value: Any) -> str:
    if isinstance(value, dict):
        def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
            key = str(item[0])
            return (int(key), key) if key.isdigit() else (10**9, key)
        return " ".join(str(text).strip() for _, text in sorted(value.items(), key=sort_key) if str(text).strip())
    return str(value).strip() if value is not None else ""


def _parse_plaba_record(path: Path, split: str, root: Path) -> SystemExample:
    data = _load_json(path)
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"{path} must contain exactly one top-level PLABA record.")
    record_id, articles = next(iter(data.items()))
    if not isinstance(articles, dict) or not articles:
        raise ValueError(f"{path} has no PLABA articles.")
    source_parts: list[str] = []
    adaptations: list[dict[str, str]] = []
    for article_id, article in articles.items():
        if not isinstance(article, dict):
            continue
        title = str(article.get("Title") or "").strip()
        abstract = _ordered_text(article.get("abstract"))
        if abstract:
            source_parts.append(f"Title: {title}\nAbstract: {abstract}" if title else abstract)
        raw_adaptations = article.get("adaptations")
        if isinstance(raw_adaptations, dict):
            for adaptation_id, adaptation in raw_adaptations.items():
                text = _ordered_text(adaptation)
                if text:
                    adaptations.append({
                        "article_id": str(article_id),
                        "adaptation_id": str(adaptation_id),
                        "text": text,
                    })
    source_text = "\n\n".join(source_parts).strip()
    if not source_text:
        raise ValueError(f"{path} has no professional-language source text.")
    return SystemExample(
        dataset="PLABA",
        split=split,
        example_id=path.stem,
        source_text=source_text,
        references={"plain_language_adaptations": adaptations, "record_id": str(record_id)},
        source_file=str(path.relative_to(root)),
    )


def load_plaba(split: str, root: Path = ROOT) -> list[SystemExample]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    folder = root / "data" / "PLABA" / split
    return [_parse_plaba_record(path, split, root) for path in sorted(folder.glob("*.json"))]


def load_system_examples(
    split: str,
    datasets: tuple[str, ...] | list[str],
    root: Path = ROOT,
) -> list[SystemExample]:
    invalid = set(datasets) - set(VALID_DATASETS)
    if invalid:
        raise ValueError(f"Unknown datasets: {sorted(invalid)}")
    examples: list[SystemExample] = []
    if "MTSamples" in datasets:
        examples.extend(load_mtsamples(split, root))
    if "PLABA" in datasets:
        examples.extend(load_plaba(split, root))
    return sorted(examples, key=lambda item: (item.dataset, item.example_id))


def select_examples(examples: list[SystemExample], count: int | None) -> list[SystemExample]:
    """Select deterministically while alternating datasets when both are present."""
    if count is None or count >= len(examples):
        return list(examples)
    groups: dict[str, list[SystemExample]] = {}
    for example in examples:
        groups.setdefault(example.dataset, []).append(example)
    selected: list[SystemExample] = []
    index = 0
    names = sorted(groups)
    while len(selected) < count:
        added = False
        for name in names:
            if index < len(groups[name]) and len(selected) < count:
                selected.append(groups[name][index])
                added = True
        if not added:
            break
        index += 1
    return selected


def _strip_json_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.S)
    return match.group(1).strip() if match else text.strip()


_QUOTE_PUNCTUATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
})
_MARKDOWN_FORMATTING = {"*", "_", "`"}


def _normalized_text_with_spans(text: str) -> tuple[str, list[int], list[int]]:
    """Normalize harmless formatting while retaining offsets into the original text."""
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, original in enumerate(text):
        normalized = unicodedata.normalize("NFKC", original).translate(_QUOTE_PUNCTUATION)
        for character in normalized:
            if character in _MARKDOWN_FORMATTING:
                continue
            if character.isspace():
                if characters and characters[-1] != " ":
                    characters.append(" ")
                    starts.append(index)
                    ends.append(index + 1)
                elif characters and characters[-1] == " ":
                    ends[-1] = index + 1
                continue
            characters.append(character)
            starts.append(index)
            ends.append(index + 1)
    if characters and characters[-1] == " ":
        characters.pop()
        starts.pop()
        ends.pop()
    return "".join(characters), starts, ends


def _resolve_quote(quote: str, text: str, label: str) -> tuple[str, str]:
    """Return an exact source span, allowing only one unique normalized match."""
    if quote in text:
        return quote, "exact"
    normalized_quote, _, _ = _normalized_text_with_spans(quote.strip())
    normalized_text, starts, ends = _normalized_text_with_spans(text)
    if not normalized_quote:
        raise ValueError(f"{label} is empty after normalization.")
    matches: list[int] = []
    offset = 0
    while True:
        match = normalized_text.find(normalized_quote, offset)
        if match < 0:
            break
        matches.append(match)
        offset = match + 1
    if not matches:
        raise ValueError(f"{label} does not quote its target text exactly or by safe formatting normalization.")
    if len(matches) > 1:
        raise ValueError(f"{label} has {len(matches)} formatting-normalized matches and is ambiguous.")
    start = matches[0]
    resolved = text[starts[start]:ends[start + len(normalized_quote) - 1]]
    return resolved, "normalized_unique"


def parse_factual_evaluation(
    raw: str,
    final_explanation: str,
    source_text: str | None = None,
    reference_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data, _, _ = decode_factual_evaluation_json(raw)
    if not isinstance(data, dict) or data.get("status") not in FACTUAL_STATUSES:
        raise ValueError("Factual evaluator returned an invalid status.")
    if not isinstance(data.get("status_reason"), str) or not isinstance(data.get("claims"), list):
        raise ValueError("Factual evaluator returned an invalid result shape.")
    if data["status"] != "scored" and data["claims"]:
        raise ValueError("A no-data factual result must have an empty claims list.")
    if data["status"] == "scored" and not data["claims"]:
        raise ValueError("A scored factual result must contain at least one claim.")
    returned_claims = data["claims"]
    valid_claims: list[dict[str, Any]] = []
    skipped_claims: list[dict[str, Any]] = []
    explanation_spans = {item["span_id"]: item["text"] for item in indexed_text_spans(final_explanation, "E")}
    source_spans = {
        item["span_id"]: item["text"]
        for item in indexed_text_spans(source_text or "", "S")
    }
    definition_by_id = {
        item["source_id"]: item
        for item in reference_definitions or []
        if isinstance(item, dict) and isinstance(item.get("source_id"), str)
    }
    for index, raw_claim in enumerate(returned_claims, start=1):
        validation_stage = "claim_schema"
        validated_claim_quote: str | None = None
        try:
            if not isinstance(raw_claim, dict) or raw_claim.get("classification") not in FACTUAL_LABELS:
                raise ValueError("invalid classification")
            if not isinstance(raw_claim.get("claim_id"), str) or not raw_claim["claim_id"].strip():
                raise ValueError("missing or invalid claim_id")
            if not isinstance(raw_claim.get("clinically_meaningful"), bool):
                raise ValueError("missing or invalid clinical-meaning flag")
            if not isinstance(raw_claim.get("reason"), str):
                raise ValueError("missing reason")
            claim = copy.deepcopy(raw_claim)
            if "claim_span_id" in raw_claim or "source_span_ids" in raw_claim:
                validation_stage = "claim_alignment"
                claim_span_id = raw_claim.get("claim_span_id")
                if not isinstance(claim_span_id, str) or claim_span_id not in explanation_spans:
                    raise ValueError(f"invalid claim_span_id: {claim_span_id!r}")
                if not isinstance(raw_claim.get("claim_text"), str) or not raw_claim["claim_text"].strip():
                    raise ValueError("missing or invalid claim_text")
                validated_claim_quote = explanation_spans[claim_span_id]
                claim["claim_quote"] = validated_claim_quote
                claim["claim_quote_match"] = "span_id"

                validation_stage = "evidence_alignment"
                source_ids = raw_claim.get("source_span_ids")
                if not isinstance(source_ids, list) or any(not isinstance(item, str) for item in source_ids):
                    raise ValueError("source_span_ids must be a list of strings")
                if len(source_ids) != len(set(source_ids)):
                    raise ValueError("source_span_ids contains duplicate IDs")
                invalid_ids = [item for item in source_ids if item not in source_spans]
                if invalid_ids:
                    raise ValueError(f"invalid source_span_ids: {invalid_ids}")
                uses_typed_evidence = any(
                    field in raw_claim
                    for field in ("claim_type", "evidence_type", "reference_definition_ids")
                )
                if uses_typed_evidence:
                    claim_type = raw_claim.get("claim_type")
                    evidence_type = raw_claim.get("evidence_type")
                    reference_ids = raw_claim.get("reference_definition_ids")
                    if claim_type not in FACTUAL_CLAIM_TYPES:
                        raise ValueError(f"invalid claim_type: {claim_type!r}")
                    if evidence_type not in FACTUAL_EVIDENCE_TYPES:
                        raise ValueError(f"invalid evidence_type: {evidence_type!r}")
                    # Gemini occasionally omits this empty array on claims that
                    # explicitly use only report evidence (or no evidence). That
                    # omission is safe to normalize because these evidence types
                    # assert that no UMLS definition was used. Never infer IDs for
                    # umls_definition or both: those claims must identify the exact
                    # supplied definition that grounds them.
                    if (
                        reference_ids is None
                        and evidence_type in {"medical_report", "none"}
                    ):
                        reference_ids = []
                        claim["reference_definition_ids"] = []
                    if not isinstance(reference_ids, list) or any(
                        not isinstance(item, str) for item in reference_ids
                    ):
                        raise ValueError("reference_definition_ids must be a list of strings")
                    if len(reference_ids) != len(set(reference_ids)):
                        raise ValueError("reference_definition_ids contains duplicate IDs")
                    invalid_reference_ids = [
                        item for item in reference_ids if item not in definition_by_id
                    ]
                    if invalid_reference_ids:
                        raise ValueError(
                            f"invalid reference_definition_ids: {invalid_reference_ids}"
                        )
                    needs_report = evidence_type in {"medical_report", "both"}
                    needs_definition = evidence_type in {"umls_definition", "both"}
                    if needs_report != bool(source_ids):
                        raise ValueError(
                            f"evidence_type {evidence_type!r} does not match source_span_ids"
                        )
                    if needs_definition != bool(reference_ids):
                        raise ValueError(
                            f"evidence_type {evidence_type!r} does not match reference_definition_ids"
                        )
                    if claim_type == "patient_specific" and reference_ids:
                        raise ValueError(
                            "patient_specific claims cannot cite UMLS definitions"
                        )
                    if (
                        claim["classification"] in {"supported", "contradicted"}
                        and evidence_type == "none"
                    ):
                        raise ValueError(
                            f"{claim['classification']} claim requires aligned evidence"
                        )
                    claim["reference_definitions"] = [
                        definition_by_id[item] for item in reference_ids
                    ]
                elif claim["classification"] in {"supported", "contradicted"} and not source_ids:
                    raise ValueError(f"{claim['classification']} claim requires at least one source span")
                claim["source_quotes"] = [
                    {"span_id": item, "text": source_spans[item]} for item in source_ids
                ]
                claim["source_quote"] = "\n".join(source_spans[item] for item in source_ids) or None
                claim["source_quote_match"] = "span_ids" if source_ids else "not_provided"
            else:
                # Backward-compatible validation for historical quote-based cache records.
                validation_stage = "claim_alignment"
                quote = raw_claim.get("claim_quote")
                if not isinstance(quote, str) or not quote.strip():
                    raise ValueError("missing or invalid claim_quote")
                resolved_quote, claim_match = _resolve_quote(
                    quote, final_explanation, f"Claim {index} claim_quote"
                )
                validated_claim_quote = resolved_quote
                if claim_match != "exact":
                    claim["evaluator_claim_quote"] = quote
                    claim["claim_quote"] = resolved_quote
                claim["claim_quote_match"] = claim_match

                validation_stage = "evidence_alignment"
                source_quote = claim.get("source_quote")
                if source_quote is not None and not isinstance(source_quote, str):
                    raise ValueError("invalid source_quote")
                if source_text is not None and source_quote:
                    resolved_source, source_match = _resolve_quote(
                        source_quote, source_text, f"Claim {index} source_quote"
                    )
                    if source_match != "exact":
                        claim["evaluator_source_quote"] = source_quote
                        claim["source_quote"] = resolved_source
                    claim["source_quote_match"] = source_match
                elif source_quote is None:
                    claim["source_quote_match"] = "not_provided"
            valid_claims.append(claim)
        except ValueError as exc:
            skipped_claims.append({
                "claim_index": index,
                "claim_id": raw_claim.get("claim_id") if isinstance(raw_claim, dict) else None,
                "validation_stage": validation_stage,
                "validated_claim_quote": validated_claim_quote,
                "error": str(exc),
                "raw_claim": raw_claim,
            })
    data["claims"] = valid_claims
    data["claims_returned"] = len(returned_claims)
    data["skipped_claims"] = skipped_claims
    return data


def decode_factual_evaluation_json(raw: str) -> tuple[dict[str, Any], str, bool]:
    """Decode one evaluator object, recovering a complete trailing retry.

    Some streamed Gemini responses contain a malformed partial attempt followed
    by a complete JSON object. Recovery is deliberately narrow: only a suffix
    that parses through the factual-evaluator schema is accepted.
    """
    stripped = _strip_json_fence(raw)
    try:
        data = json.loads(stripped)
        return data, stripped, False
    except json.JSONDecodeError as original_error:
        starts = [index for index, character in enumerate(stripped) if character == "{"]
        for start in reversed(starts):
            candidate = stripped[start:].strip()
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(data, dict)
                and data.get("status") in FACTUAL_STATUSES
                and isinstance(data.get("status_reason"), str)
                and isinstance(data.get("claims"), list)
            ):
                return data, candidate, True
        raise original_error


def factual_metrics(result: dict[str, Any] | None) -> dict[str, Any]:
    skipped = len(result.get("skipped_claims", [])) if result else 0
    returned = int(result.get("claims_returned", 0)) if result else 0
    coverage = ((returned - skipped) / returned * 100) if returned else None
    if not result or result.get("status") != "scored" or not result.get("claims"):
        return {
            "claims_returned": returned,
            "claims_assessed": 0,
            "claims_skipped": skipped,
            "claim_validation_coverage": coverage,
            "supported_claims": 0,
            "uncertain_claims": 0,
            "factual_fidelity_estimate": None,
            "clinically_meaningful_hallucination": None,
        }
    claims = result["claims"]
    supported = sum(claim["classification"] == "supported" for claim in claims)
    uncertain = sum(claim["classification"] == "uncertain" for claim in claims)
    hallucination_detected = any(
        claim["clinically_meaningful"]
        and claim["classification"] in {"unsupported", "contradicted"}
        for claim in claims
    )
    # A validated positive can be reported even when another claim was skipped. A
    # negative cannot: the skipped claim could itself be a meaningful hallucination.
    hallucination = True if hallucination_detected else (None if skipped else False)
    return {
        "claims_returned": returned,
        "claims_assessed": len(claims),
        "claims_skipped": skipped,
        "claim_validation_coverage": coverage,
        "supported_claims": supported,
        "uncertain_claims": uncertain,
        "factual_fidelity_estimate": supported / len(claims) * 100,
        "clinically_meaningful_hallucination": hallucination,
    }


def refresh_cached_factual_evaluation(record: dict[str, Any]) -> dict[str, Any]:
    """Reparse cached evaluator JSON with the current deterministic validator.

    This never calls a model. It lets safe parser repairs update cached metrics
    without regenerating the explanation or rerunning the factual evaluator.
    """
    evaluator = record.get("factual_evaluator")
    if not isinstance(evaluator, dict):
        return record
    if evaluator.get("parser_version") == FACTUAL_PARSER_VERSION:
        return record
    raw_output = evaluator.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return record
    request = evaluator.get("request") or {}
    definitions = request.get("reference_definitions") or []
    try:
        parsed = parse_factual_evaluation(
            raw_output,
            str(record.get("final_explanation") or ""),
            str(record.get("source_text") or ""),
            definitions,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        evaluator["parser_version"] = FACTUAL_PARSER_VERSION
        evaluator["reparse_error"] = f"{type(exc).__name__}: {exc}"
        return record
    evaluator["parsed_output"] = parsed
    evaluator["error"] = None
    evaluator.pop("reparse_error", None)
    evaluator["parser_version"] = FACTUAL_PARSER_VERSION
    evaluator["reparsed_from_cached_raw"] = True
    record["factual_metrics"] = factual_metrics(parsed)
    return record


def output_metrics(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return evaluate_single(text), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def refresh_cached_readability_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute cheap deterministic metrics without rerunning generation."""
    text = record.get("final_explanation") or record.get("final_output")
    if not isinstance(text, str) or not text.strip():
        return record
    if (
        record.get("metric_version") == METRIC_IMPLEMENTATION_VERSION
        and isinstance(record.get("readability_metrics"), dict)
    ):
        return record
    metrics, error = output_metrics(text)
    record["readability_metrics"] = metrics
    record["readability_error"] = error
    record["metric_version"] = METRIC_IMPLEMENTATION_VERSION
    return record


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        refresh_cached_readability_metrics(record)
    readability = [record["readability_metrics"] for record in records if record.get("readability_metrics")]
    factual = [record.get("factual_metrics") or {} for record in records]
    scored_factual = [item for item in factual if item.get("claims_assessed", 0) > 0]
    total_claims = sum(item["claims_assessed"] for item in scored_factual)
    total_returned = sum(item.get("claims_returned", item.get("claims_assessed", 0) + item.get("claims_skipped", 0)) for item in factual)
    supported_claims = sum(item["supported_claims"] for item in scored_factual)
    hallucination_scored = [item for item in scored_factual if item.get("clinically_meaningful_hallucination") is not None]
    return {
        "outputs_total": len(records),
        "readability_scored": len(readability),
        "mean_flesch_kincaid_grade": _mean(readability, "flesch_kincaid_grade"),
        "percent_at_or_below_grade_8": _percent(readability, "meets_target"),
        "mean_smog": _mean(readability, "smog_index"),
        "mean_jargon_density": _mean(readability, "jargon_density"),
        "mean_word_count": _mean(readability, "word_count"),
        "refusal_rate": _percent(readability, "is_refusal"),
        "factual_outputs_scored": len(scored_factual),
        "claims_assessed": total_claims,
        "claims_returned": total_returned,
        "claims_skipped": sum(item.get("claims_skipped", 0) for item in factual),
        "claim_validation_coverage": total_claims / total_returned * 100 if total_returned else None,
        "uncertain_claims": sum(item.get("uncertain_claims", 0) for item in scored_factual),
        "factual_fidelity_estimate": supported_claims / total_claims * 100 if total_claims else None,
        "hallucination_outputs_scored": len(hallucination_scored),
        "hallucination_rate": (
            sum(bool(item["clinically_meaningful_hallucination"]) for item in hallucination_scored)
            / len(hallucination_scored) * 100
            if hallucination_scored else None
        ),
    }


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return sum(values) / len(values) if values else None


def _percent(items: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(item[key]) for item in items if item.get(key) is not None]
    return sum(values) / len(values) * 100 if values else None


def prompt_hashes() -> dict[str, str]:
    prompts = {
        "extractor": EXTRACTOR_INSTRUCTION,
        "simplifier": SIMPLIFIER_INSTRUCTION,
        "production_verifier": PRODUCTION_VERIFIER_PROMPT,
        "refiner": _REFINER_INSTRUCTION,
        "readability": READABILITY_INSTRUCTION,
        "factual_evaluator": FACTUAL_EVALUATOR_PROMPT,
    }
    return {name: hashlib.sha256(prompt.encode("utf-8")).hexdigest() for name, prompt in prompts.items()}


def prompt_hashes_cache_compatible(
    stored: dict[str, Any], expected: dict[str, str],
) -> bool:
    """Allow only the two explicitly reviewed v7 prompt transitions."""
    if stored == expected:
        return True
    return (
        PIPELINE_IMPLEMENTATION_VERSION == "system-pipeline-v7"
        and stored in (
            V7_PRE_MEDICATION_GUARD_PROMPT_HASHES,
            V7_MEDICATION_GUARD_PROMPT_HASHES,
        )
        and expected == V7_READABILITY_REVISION_PROMPT_HASHES
    )


def pipeline_config() -> dict[str, Any]:
    """Settings that change pipeline output, including which lookup sources exist."""
    return {"max_revision_cycles": MAX_REVISION_CYCLES, "reference_lookup": lookup_config_from_env()}


def cache_key(example: SystemExample, pipeline_model: str, evaluator_model: str) -> str:
    identity = {
        **example.metadata(),
        "source_hash": hashlib.sha256(example.source_text.encode("utf-8")).hexdigest(),
        "reference_hash": hashlib.sha256(json.dumps(example.references, sort_keys=True).encode("utf-8")).hexdigest(),
        "pipeline_model": pipeline_model,
        "evaluator_model": evaluator_model,
        "prompt_hashes": prompt_hashes(),
        "pipeline_version": PIPELINE_IMPLEMENTATION_VERSION,
        "pipeline_config": pipeline_config(),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


class AppendOnlyRunCache:
    def __init__(self, path: Path = DEFAULT_CACHE_PATH):
        self.path = path

    def find(self, key: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        found = None
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("cache_key") == key and record.get("run_status") == "completed":
                    found = record
        return found

    def records(self, *, refresh_readability: bool = True) -> list[dict[str, Any]]:
        """Return saved runs in append order, skipping malformed lines."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("run_id"), str):
                    if refresh_readability:
                        refresh_cached_readability_metrics(record)
                    records.append(record)
        return records

    def find_compatible(
        self,
        example: SystemExample,
        pipeline_model: str,
        evaluator_model: str,
    ) -> dict[str, Any] | None:
        """Reuse a prior run covered by an explicit v7 compatibility rule."""
        expected_prompts = prompt_hashes()
        for record in reversed(self.records(refresh_readability=False)):
            metadata_matches = all(
                record.get(key) == value for key, value in example.metadata().items()
            )
            stored_prompts = record.get("prompt_hashes") or {}
            if (
                metadata_matches
                and record.get("run_status") == "completed"
                and record.get("source_text") == example.source_text
                and record.get("post_generation_references") == example.references
                and record.get("pipeline_model") == pipeline_model
                and record.get("evaluator_model") == evaluator_model
                and record.get("pipeline_version") == PIPELINE_IMPLEMENTATION_VERSION
                and prompt_hashes_cache_compatible(stored_prompts, expected_prompts)
            ):
                return record
        return None

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


PipelineRunner = Callable[..., Awaitable[PipelineRunResult]]
EvaluatorRunner = Callable[..., Awaitable[str]]


async def run_system_example(
    example: SystemExample,
    pipeline_model: str,
    evaluator_model: str,
    *,
    cache: AppendOnlyRunCache | None = None,
    force: bool = False,
    persist: bool = True,
    on_pipeline_event: Callable[[dict[str, Any]], None] | None = None,
    on_evaluator_start: Callable[[], None] | None = None,
    on_evaluator_text: Callable[[str, str], None] | None = None,
    pipeline_runner: PipelineRunner = run_production_pipeline,
    evaluator_runner: EvaluatorRunner = run_factual_evaluator,
) -> dict[str, Any]:
    key = cache_key(example, pipeline_model, evaluator_model)
    if cache and not force:
        cached = cache.find(key) or cache.find_compatible(
            example, pipeline_model, evaluator_model
        )
        if cached:
            result = copy.deepcopy(cached)
            refresh_cached_readability_metrics(result)
            refresh_cached_factual_evaluation(result)
            result["cached_record_source"] = result.get("result_source")
            result["result_source"] = "cache"
            if result.get("prompt_hashes") != prompt_hashes():
                result["cache_compatibility"] = "v7_pre_medication_guard"
            return result

    pipeline = await pipeline_runner(example.source_text, pipeline_model, on_event=on_pipeline_event)
    final = pipeline.final_explanation
    readability = None
    readability_error = None
    raw_evaluation = None
    parsed_evaluation = None
    evaluator_error = None
    evaluation_request = None
    if final:
        readability, readability_error = output_metrics(final)
        reference_definitions = reference_definitions_for_prompt(
            ((pipeline.trace.get("results") or {}).get("reference_definitions") or {})
        )
        evaluation_request = factual_evaluator_request(
            example.source_text, final, example.references, reference_definitions,
        )
        try:
            if on_evaluator_start:
                on_evaluator_start()
            raw_evaluation = await evaluator_runner(
                example.source_text,
                final,
                example.references,
                evaluator_model,
                reference_definitions=reference_definitions,
                on_text=on_evaluator_text,
            )
            parsed_evaluation = parse_factual_evaluation(
                raw_evaluation, final, example.source_text, reference_definitions,
            )
        except Exception as exc:
            evaluator_error = f"{type(exc).__name__}: {exc}"

    record = {
        "schema": "faithfulmed_system_run_v3",
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_key": key,
        "result_source": "fresh_forced" if force else "fresh",
        **example.metadata(),
        "pipeline_model": pipeline_model,
        "evaluator_model": evaluator_model,
        "prompt_hashes": prompt_hashes(),
        "pipeline_version": PIPELINE_IMPLEMENTATION_VERSION,
        "pipeline_config": pipeline_config(),
        "metric_version": METRIC_IMPLEMENTATION_VERSION,
        "source_text": example.source_text,
        "post_generation_references": example.references,
        "pipeline_trace": pipeline.trace,
        "run_status": pipeline.trace.get("status", "unknown"),
        "accepted": pipeline.accepted,
        "final_explanation": final,
        "readability_metrics": readability,
        "readability_error": readability_error,
        "factual_evaluator": {
            "request": evaluation_request,
            "raw_output": raw_evaluation,
            "parsed_output": parsed_evaluation,
            "error": evaluator_error,
            "parser_version": FACTUAL_PARSER_VERSION,
        },
        "factual_metrics": factual_metrics(parsed_evaluation),
    }
    if cache and persist:
        cache.append(record)
    return record


def record_for_download(record: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy; run records intentionally contain no credentials."""
    return json.loads(json.dumps(record, ensure_ascii=False))
