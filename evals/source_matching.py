"""Post-run sentence embedding and source-similarity matching.

This module is deliberately independent of the production agent pipeline. It
uses embeddings only to surface likely source sentences for inspection; cosine
similarity is not evidence of factual support.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "source_matching.json"
DEFAULT_CACHE_PATH = ROOT / "eval_results" / "source_embeddings.sqlite3"


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str = "gemini-embedding-001"
    task_type: str = "SEMANTIC_SIMILARITY"
    output_dimensionality: int = 768


@dataclass(frozen=True)
class SentenceSpan:
    sentence_id: str
    index: int
    text: str


BatchEmbedder = Callable[[list[str], EmbeddingSettings], list[list[float]]]


def load_source_matching_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the small source-matching config with validated safe defaults."""
    defaults: dict[str, Any] = {
        "embedding_model": "gemini-embedding-001",
        "embedding_task_type": "SEMANTIC_SIMILARITY",
        "embedding_output_dimensionality": 768,
        "embedding_match_threshold": 0.70,
    }
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Source-matching config must be a JSON object.")
        defaults.update(loaded)
    threshold = float(defaults["embedding_match_threshold"])
    dimensionality = int(defaults["embedding_output_dimensionality"])
    if not 0 <= threshold <= 1:
        raise ValueError("embedding_match_threshold must be between 0 and 1.")
    if dimensionality <= 0:
        raise ValueError("embedding_output_dimensionality must be positive.")
    defaults["embedding_match_threshold"] = threshold
    defaults["embedding_output_dimensionality"] = dimensionality
    return defaults


def settings_from_config(config: dict[str, Any]) -> EmbeddingSettings:
    return EmbeddingSettings(
        model=str(config["embedding_model"]),
        task_type=str(config["embedding_task_type"]),
        output_dimensionality=int(config["embedding_output_dimensionality"]),
    )


def split_sentences(text: str, prefix: str) -> list[SentenceSpan]:
    """Split nonempty lines into ordered sentence substrings.

    Returned text is copied from the input (apart from surrounding whitespace),
    so punctuation and Markdown markers remain available to the UI.
    """
    if not isinstance(text, str):
        return []
    pieces: list[str] = []
    boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘’“”(#*\-])")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pieces.extend(part for part in boundary.split(line) if part)
    return [
        SentenceSpan(f"{prefix}{index:03d}", index - 1, piece)
        for index, piece in enumerate(pieces, start=1)
    ]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_key(text: str, settings: EmbeddingSettings) -> str:
    payload = "\x1f".join((
        text_hash(text), settings.model, settings.task_type,
        str(settings.output_dimensionality),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Persistent SQLite cache keyed only by text and embedding settings."""

    def __init__(self, path: Path = DEFAULT_CACHE_PATH):
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterable[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    text_hash TEXT NOT NULL,
                    exact_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    output_dimensionality INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding_usages (
                    cache_key TEXT NOT NULL,
                    report_id TEXT,
                    role TEXT NOT NULL,
                    sentence_id TEXT NOT NULL,
                    sentence_index INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    UNIQUE(cache_key, report_id, role, sentence_id, sentence_index),
                    FOREIGN KEY(cache_key) REFERENCES embeddings(cache_key)
                );
            """)

    def get(self, text: str, settings: EmbeddingSettings) -> list[float] | None:
        key = embedding_key(text, settings)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT vector_json FROM embeddings WHERE cache_key = ?", (key,)
            ).fetchone()
        return [float(value) for value in json.loads(row["vector_json"])] if row else None

    def put(self, text: str, settings: EmbeddingSettings, vector: Iterable[float]) -> None:
        values = [float(value) for value in vector]
        if len(values) != settings.output_dimensionality:
            raise ValueError(
                f"Embedding has {len(values)} values; expected "
                f"{settings.output_dimensionality}."
            )
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO embeddings
                   (cache_key, text_hash, exact_text, model, task_type,
                    output_dimensionality, vector_json, created_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    embedding_key(text, settings), text_hash(text), text,
                    settings.model, settings.task_type,
                    settings.output_dimensionality, json.dumps(values), now,
                ),
            )

    def record_usage(
        self,
        text: str,
        settings: EmbeddingSettings,
        *,
        report_id: str | None,
        role: str,
        sentence_id: str,
        sentence_index: int,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO embedding_usages
                   (cache_key, report_id, role, sentence_id, sentence_index, created_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    embedding_key(text, settings), report_id, role, sentence_id,
                    sentence_index, datetime.now(timezone.utc).isoformat(),
                ),
            )

    def clear(self) -> int:
        with self._connection() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            connection.execute("DELETE FROM embedding_usages")
            connection.execute("DELETE FROM embeddings")
        return count

    def stats(self) -> dict[str, Any]:
        with self._connection() as connection:
            vectors = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            usages = int(connection.execute("SELECT COUNT(*) FROM embedding_usages").fetchone()[0])
        return {"path": str(self.path), "vectors": vectors, "usages": usages}

    def inspect_entries(self, limit: int = 100, *, include_vectors: bool = False) -> list[dict[str, Any]]:
        vector_column = ", e.vector_json" if include_vectors else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"""SELECT e.cache_key, e.text_hash, e.exact_text, e.model,
                           e.task_type, e.output_dimensionality, e.created_at_utc,
                           u.report_id, u.role, u.sentence_id, u.sentence_index
                           {vector_column}
                    FROM embeddings e
                    LEFT JOIN embedding_usages u ON u.cache_key = e.cache_key
                    ORDER BY e.created_at_utc DESC, u.report_id, u.sentence_index
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        entries: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if include_vectors:
                item["vector"] = json.loads(item.pop("vector_json"))
            entries.append(item)
        return entries

    def export_jsonl(self) -> str:
        entries = self.inspect_entries(limit=1_000_000, include_vectors=True)
        return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in entries)


def google_batch_embedder(client: Any) -> BatchEmbedder:
    """Adapt a Google GenAI client to the injectable batch-embedder contract."""
    def embed(texts: list[str], settings: EmbeddingSettings) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        # Gemini accepts at most 100 EmbedContent requests in one batch. Long
        # reports can exceed that after source and explanation sentence splits.
        for start in range(0, len(texts), 100):
            response = client.models.embed_content(
                model=settings.model,
                contents=texts[start:start + 100],
                config=types.EmbedContentConfig(
                    task_type=settings.task_type,
                    output_dimensionality=settings.output_dimensionality,
                ),
            )
            vectors.extend(
                [float(value) for value in item.values]
                for item in response.embeddings
            )
        return vectors
    return embed


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compare embeddings with different dimensions.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def build_source_matches(
    source_text: str,
    final_explanation: str,
    *,
    report_id: str | None,
    cache: EmbeddingCache,
    embed_batch: BatchEmbedder,
    settings: EmbeddingSettings,
) -> dict[str, Any]:
    """Embed missing sentences once and retain every local cosine score."""
    source = split_sentences(source_text, "S")
    explanation = split_sentences(final_explanation, "E")
    if not source or not explanation:
        raise ValueError("Source matching requires source and explanation sentences.")

    all_spans = [("source", item) for item in source] + [
        ("explanation", item) for item in explanation
    ]
    vectors: dict[str, list[float]] = {}
    origins: dict[str, str] = {}
    missing_texts: list[str] = []
    missing_seen: set[str] = set()
    for _, span in all_spans:
        if span.text in vectors or span.text in missing_seen:
            continue
        cached = cache.get(span.text, settings)
        if cached is not None:
            vectors[span.text] = cached
            origins[span.text] = "cache"
        else:
            missing_texts.append(span.text)
            missing_seen.add(span.text)

    if missing_texts:
        returned = embed_batch(missing_texts, settings)
        if len(returned) != len(missing_texts):
            raise ValueError(
                f"Embedding API returned {len(returned)} vectors for "
                f"{len(missing_texts)} sentences."
            )
        for text, vector in zip(missing_texts, returned):
            cache.put(text, settings, vector)
            vectors[text] = [float(value) for value in vector]
            origins[text] = "new"

    for role, span in all_spans:
        cache.record_usage(
            span.text, settings, report_id=report_id, role=role,
            sentence_id=span.sentence_id, sentence_index=span.index,
        )

    source_rows = [
        {
            "source_sentence_id": span.sentence_id,
            "source_sentence_index": span.index,
            "source_text": span.text,
            "embedding_source": origins[span.text],
        }
        for span in source
    ]
    results: list[dict[str, Any]] = []
    for explanation_span in explanation:
        matches = []
        for source_span in source:
            matches.append({
                "source_sentence_id": source_span.sentence_id,
                "source_sentence_index": source_span.index,
                "source_text": source_span.text,
                "embedding_source": origins[source_span.text],
                "score": cosine_similarity(
                    vectors[explanation_span.text], vectors[source_span.text]
                ),
            })
        matches.sort(key=lambda item: (-item["score"], item["source_sentence_index"]))
        results.append({
            "explanation_sentence_id": explanation_span.sentence_id,
            "explanation_sentence_index": explanation_span.index,
            "explanation_text": explanation_span.text,
            "embedding_source": origins[explanation_span.text],
            "ranked_source_matches": matches,
        })

    return {
        "schema": "faithfulmed_source_similarity_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_id": report_id,
        "model": settings.model,
        "task_type": settings.task_type,
        "output_dimensionality": settings.output_dimensionality,
        "interpretation": (
            "Cosine-similarity candidates only; scores are not factual-support "
            "probabilities or faithfulness percentages."
        ),
        "source_sentences": source_rows,
        "explanation_sentences": results,
        "embedding_counts": {
            "new": sum(origin == "new" for origin in origins.values()),
            "cache": sum(origin == "cache" for origin in origins.values()),
        },
    }


def apply_match_threshold(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Relabel/filter existing scores without embedding or mutating the input."""
    if not 0 <= threshold <= 1:
        raise ValueError("Source match threshold must be between 0 and 1.")
    rendered = copy.deepcopy(result)
    rendered["threshold_used"] = float(threshold)
    for item in rendered.get("explanation_sentences", []):
        ranked = item.pop("ranked_source_matches", [])
        if not ranked:
            item["best_match"] = None
            item["other_possible_matches"] = []
            continue
        best = ranked[0]
        best["threshold_status"] = (
            "Above threshold" if best["score"] >= threshold
            else "Below threshold — closest match shown"
        )
        item["best_match"] = best
        others = []
        for match in ranked[1:]:
            if match["score"] >= threshold:
                match["threshold_status"] = "Above threshold"
                others.append(match)
        item["other_possible_matches"] = others
    return rendered


def source_match_metrics(
    results: Iterable[dict[str, Any]],
    threshold: float,
    *,
    ambiguity_margin: float = 0.05,
) -> dict[str, Any]:
    """Summarize sentence-level similarity scores at a chosen threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("Source match threshold must be between 0 and 1.")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be nonnegative.")
    total = above = multiple = ambiguous = 0
    closest_scores: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for item in result.get("explanation_sentences", []):
            ranked = item.get("ranked_source_matches") or []
            if not ranked and item.get("best_match"):
                ranked = [item["best_match"], *(item.get("other_possible_matches") or [])]
            scores = sorted(
                (float(match["score"]) for match in ranked if match.get("score") is not None),
                reverse=True,
            )
            if not scores:
                continue
            total += 1
            closest_scores.append(scores[0])
            above_threshold = [score for score in scores if score >= threshold]
            if above_threshold:
                above += 1
            if len(above_threshold) >= 2:
                multiple += 1
                if above_threshold[0] - above_threshold[1] <= ambiguity_margin:
                    ambiguous += 1
    below = total - above
    percentage = lambda count: count / total * 100 if total else None
    return {
        "sentences": total,
        "above_threshold": above,
        "above_threshold_rate": percentage(above),
        "below_threshold": below,
        "below_threshold_rate": percentage(below),
        "multiple_match_sentences": multiple,
        "multiple_match_rate": percentage(multiple),
        "ambiguous_match_sentences": ambiguous,
        "ambiguous_match_rate": percentage(ambiguous),
        "minimum_closest_match_score": min(closest_scores) if closest_scores else None,
        "threshold": threshold,
        "ambiguity_margin": ambiguity_margin,
    }
