"""Adapt cached single-agent outputs to the full-system evaluation workspace.

The original baseline JSONL remains immutable.  Factual-evaluator results and
source-similarity results are written to a separate append-only cache so the
42 expensive generation calls never need to be repeated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from agents.baseline import run_baseline
from agents.factual_evaluator import (
    FACTUAL_EVALUATOR_PROMPT,
    factual_evaluator_request,
    run_factual_evaluator,
)
from evals.source_matching import (
    BatchEmbedder,
    EmbeddingCache,
    EmbeddingSettings,
    build_source_matches,
)
from evals.readability_eval import evaluate_single
from evals.system_demo import factual_metrics, parse_factual_evaluation


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_RESULTS_PATH = ROOT / "runs" / "baseline.jsonl"
DEFAULT_BASELINE_ENRICHMENT_PATH = ROOT / "eval_results" / "baseline_demo_runs.jsonl"
DEFAULT_BASELINE_REVIEW_PATH = ROOT / "eval_results" / "baseline_claim_reviews.jsonl"
BASELINE_ADAPTER_VERSION = "baseline-system-adapter-v1"

EvaluatorRunner = Callable[..., Awaitable[str]]
BaselineRunner = Callable[..., Awaitable[str]]


def baseline_cache_key(record: dict[str, Any], evaluator_model: str) -> str:
    """Identify one immutable baseline output plus its evaluator configuration."""
    identity = {
        "adapter_version": BASELINE_ADAPTER_VERSION,
        "split": record.get("split"),
        "example_id": record.get("example_id"),
        "source_hash": hashlib.sha256(
            str(record.get("source_text") or "").encode("utf-8")
        ).hexdigest(),
        "output_hash": hashlib.sha256(
            str(record.get("final_explanation") or record.get("generated_text") or "").encode("utf-8")
        ).hexdigest(),
        "baseline_model": record.get("baseline_model") or record.get("model"),
        "evaluator_model": evaluator_model,
        "factual_evaluator_prompt_hash": hashlib.sha256(
            FACTUAL_EVALUATOR_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


class BaselineEnrichmentCache:
    """Append-only factuality and source-match snapshots for baseline outputs."""

    def __init__(self, path: Path = DEFAULT_BASELINE_ENRICHMENT_PATH):
        self.path = path

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("cache_key"), str):
                    records.append(record)
        return records

    def find(self, key: str) -> dict[str, Any] | None:
        found = None
        for record in self.records():
            if record.get("cache_key") == key:
                found = record
        return copy.deepcopy(found) if found else None

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _baseline_record(raw: dict[str, Any], evaluator_model: str) -> dict[str, Any]:
    generated = str(raw.get("generated_text") or "")
    source = str(raw.get("source_text") or "")
    provisional = {
        "schema": "faithfulmed_baseline_run_v1",
        "result_source": "baseline_generation_cache",
        "dataset": "MTSamples",
        "split": raw.get("split"),
        "example_id": str(raw.get("example_id")),
        "source_file": raw.get("source_file"),
        "baseline_model": raw.get("model"),
        "pipeline_model": raw.get("model"),
        "evaluator_model": evaluator_model,
        "source_text": source,
        "post_generation_references": {},
        "final_explanation": generated,
        "readability_metrics": copy.deepcopy(raw.get("metrics")),
        "readability_error": None,
        "accepted": None,
        "run_status": "cached_baseline_generation",
        "pipeline_trace": {
            "status": "cached_baseline_generation",
            "stages": [{
                "stage_id": "baseline_generation",
                "agent": "Single-agent baseline",
                "status": "loaded_from_cache",
                "input": {"source_file": raw.get("source_file")},
                "output": generated,
            }],
        },
        "factual_evaluator": {
            "request": None,
            "raw_output": None,
            "parsed_output": None,
            "error": None,
        },
        "factual_metrics": factual_metrics(None),
        "baseline_generation_record": copy.deepcopy(raw),
    }
    key = baseline_cache_key(provisional, evaluator_model)
    provisional["cache_key"] = key
    provisional["run_id"] = f"baseline-pending-{key[:20]}"
    return provisional


def load_baseline_records(
    evaluator_model: str,
    *,
    results_path: Path = DEFAULT_BASELINE_RESULTS_PATH,
    cache: BaselineEnrichmentCache | None = None,
) -> list[dict[str, Any]]:
    """Load the latest successful generation per split/example/model.

    Any compatible enrichment snapshot replaces the provisional display record.
    Malformed, failed, or empty generation records are intentionally excluded.
    """
    if not results_path.exists():
        return []
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict) or raw.get("status") != "success":
                continue
            if not str(raw.get("source_text") or "").strip():
                continue
            if not str(raw.get("generated_text") or "").strip():
                continue
            identity = (
                str(raw.get("split")), str(raw.get("example_id")), str(raw.get("model")),
            )
            if identity not in latest:
                order.append(identity)
            latest[identity] = raw

    enrichment_cache = cache or BaselineEnrichmentCache()
    cached_events = enrichment_cache.records()
    records: list[dict[str, Any]] = []
    for identity in order:
        provisional = _baseline_record(latest[identity], evaluator_model)
        enriched = None
        for event in cached_events:
            if (
                event.get("split") == provisional.get("split")
                and event.get("example_id") == provisional.get("example_id")
                and event.get("baseline_model") == provisional.get("baseline_model")
                and event.get("source_text") == provisional.get("source_text")
                and event.get("evaluator_model") == evaluator_model
                and event.get("cache_key") == baseline_cache_key(event, evaluator_model)
            ):
                enriched = copy.deepcopy(event)
        if enriched:
            enriched["cached_record_source"] = enriched.get("result_source")
            enriched["result_source"] = "baseline_enrichment_cache"
            records.append(enriched)
        else:
            records.append(provisional)
    return records


def has_factual_evaluation(record: dict[str, Any]) -> bool:
    evaluator = record.get("factual_evaluator") or {}
    return (
        not evaluator.get("error")
        and isinstance(evaluator.get("parsed_output"), dict)
    )


def has_compatible_source_matching(
    record: dict[str, Any], settings: EmbeddingSettings,
) -> bool:
    matching = record.get("source_matching")
    return bool(
        isinstance(matching, dict)
        and matching.get("model") == settings.model
        and matching.get("task_type") == settings.task_type
        and matching.get("output_dimensionality") == settings.output_dimensionality
        and matching.get("explanation_sentences")
    )


async def enrich_baseline_record(
    record: dict[str, Any],
    evaluator_model: str,
    *,
    cache: BaselineEnrichmentCache,
    embedding_cache: EmbeddingCache,
    embedding_settings: EmbeddingSettings,
    embed_batch: BatchEmbedder,
    force_evaluator: bool = False,
    force_source_matching: bool = False,
    evaluator_runner: EvaluatorRunner = run_factual_evaluator,
    on_evaluator_text: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Add only missing factual and embedding analyses to a cached generation."""
    key = baseline_cache_key(record, evaluator_model)
    saved = cache.find(key)
    working = copy.deepcopy(
        record if force_evaluator or force_source_matching else (saved or record)
    )
    working["cache_key"] = key
    working["evaluator_model"] = evaluator_model
    if not saved or force_evaluator:
        # A forced evaluator run can produce a different claim set. Give it a
        # new review identity so older human decisions cannot attach by C1/C2.
        working["run_id"] = uuid.uuid4().hex
    changed = False

    if force_evaluator or not has_factual_evaluation(working):
        source = str(working.get("source_text") or "")
        explanation = str(working.get("final_explanation") or "")
        request = factual_evaluator_request(source, explanation, {})
        raw_output = None
        parsed_output = None
        error = None
        try:
            raw_output = await evaluator_runner(
                source, explanation, {}, evaluator_model, on_text=on_evaluator_text,
            )
            parsed_output = parse_factual_evaluation(raw_output, explanation, source)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        working["factual_evaluator"] = {
            "request": request,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "error": error,
        }
        working["factual_metrics"] = factual_metrics(parsed_output)
        changed = True

    if force_source_matching or not has_compatible_source_matching(working, embedding_settings):
        try:
            matching = build_source_matches(
                str(working.get("source_text") or ""),
                str(working.get("final_explanation") or ""),
                report_id=(
                    f"baseline:{working.get('split')}:{working.get('example_id')}:"
                    f"{working.get('run_id')}"
                ),
                cache=embedding_cache,
                embed_batch=embed_batch,
                settings=embedding_settings,
            )
            working["source_matching"] = matching
            working.pop("source_matching_error", None)
        except Exception as exc:
            working["source_matching_error"] = f"{type(exc).__name__}: {exc}"
        changed = True

    if not changed and saved:
        working["cached_record_source"] = working.get("result_source")
        working["result_source"] = "baseline_enrichment_cache"
        return working

    generation_was_forced = working.get("generation_mode") == "forced"
    working.update({
        "schema": "faithfulmed_baseline_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_source": (
            "baseline_fresh_generation" if generation_was_forced
            else "baseline_enriched_forced" if (force_evaluator or force_source_matching)
            else "baseline_enriched"
        ),
        "adapter_version": BASELINE_ADAPTER_VERSION,
    })
    cache.append(working)
    return working


async def run_forced_baseline(
    record: dict[str, Any],
    evaluator_model: str,
    *,
    cache: BaselineEnrichmentCache,
    embedding_cache: EmbeddingCache,
    embedding_settings: EmbeddingSettings,
    embed_batch: BatchEmbedder,
    baseline_runner: BaselineRunner = run_baseline,
    evaluator_runner: EvaluatorRunner = run_factual_evaluator,
    on_evaluator_text: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Generate and fully analyze a new baseline answer for one source report.

    Source vectors are still obtained through the shared embedding cache, so
    unchanged source sentences are not sent to the embedding API again when a
    compatible vector already exists. The new explanation is always embedded,
    and the factual evaluator is always called.
    """
    source = str(record.get("source_text") or "")
    model = str(record.get("baseline_model") or record.get("pipeline_model") or "")
    generated = await baseline_runner(source, model=model)
    if not isinstance(generated, str) or not generated.strip():
        raise ValueError("The forced baseline generation returned empty text.")
    generated = generated.strip()

    readability = None
    readability_error = None
    try:
        readability = evaluate_single(generated)
    except Exception as exc:
        readability_error = f"{type(exc).__name__}: {exc}"

    fresh = copy.deepcopy(record)
    for field in (
        "source_matching", "source_matching_error", "factual_evaluator",
        "factual_metrics", "cached_record_source", "created_at_utc",
    ):
        fresh.pop(field, None)
    fresh.update({
        "run_id": uuid.uuid4().hex,
        "result_source": "baseline_generation_fresh",
        "generation_mode": "forced",
        "baseline_model": model,
        "pipeline_model": model,
        "evaluator_model": evaluator_model,
        "final_explanation": generated,
        "readability_metrics": readability,
        "readability_error": readability_error,
        "run_status": "fresh_baseline_generation",
        "accepted": None,
        "pipeline_trace": {
            "status": "fresh_baseline_generation",
            "stages": [{
                "stage_id": "baseline_generation",
                "agent": "Single-agent baseline",
                "status": "completed",
                "input": {"source_file": record.get("source_file")},
                "output": generated,
            }],
        },
        "baseline_generation_record": {
            "split": record.get("split"),
            "example_id": record.get("example_id"),
            "source_file": record.get("source_file"),
            "source_text": source,
            "model": model,
            "generated_text": generated,
            "metrics": readability,
            "status": "success",
            "error": readability_error,
            "generation_mode": "forced",
        },
    })
    fresh["cache_key"] = baseline_cache_key(fresh, evaluator_model)
    return await enrich_baseline_record(
        fresh,
        evaluator_model,
        cache=cache,
        embedding_cache=embedding_cache,
        embedding_settings=embedding_settings,
        embed_batch=embed_batch,
        force_evaluator=True,
        force_source_matching=True,
        evaluator_runner=evaluator_runner,
        on_evaluator_text=on_evaluator_text,
    )
