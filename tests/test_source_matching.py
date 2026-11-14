"""Offline tests for post-run embedding source matching."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evals.source_matching import (
    EmbeddingCache,
    EmbeddingSettings,
    apply_match_threshold,
    build_source_matches,
    google_batch_embedder,
    source_match_metrics,
    split_sentences,
)
from agentic_system_demo import hoverable_explanation_html


class RecordingEmbedder:
    def __init__(self, values: dict[str, list[float]] | None = None):
        self.values = values or {}
        self.calls: list[tuple[list[str], EmbeddingSettings]] = []

    def __call__(self, texts: list[str], settings: EmbeddingSettings) -> list[list[float]]:
        self.calls.append((list(texts), settings))
        return [
            list(self.values.get(text, [1.0] + [0.0] * (settings.output_dimensionality - 1)))
            for text in texts
        ]


class SourceMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cache = EmbeddingCache(Path(self.directory.name) / "embeddings.sqlite3")
        self.settings = EmbeddingSettings(output_dimensionality=2)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_same_report_reuses_cached_source_sentence_embeddings(self) -> None:
        embedder = RecordingEmbedder()
        first = build_source_matches(
            "Source one. Source two.", "First explanation.", report_id="R1",
            cache=self.cache, embed_batch=embedder, settings=self.settings,
        )
        second = build_source_matches(
            "Source one. Source two.", "Different explanation.", report_id="R1-again",
            cache=self.cache, embed_batch=embedder, settings=self.settings,
        )

        self.assertEqual(len(embedder.calls), 2)
        self.assertEqual(embedder.calls[0][0], ["Source one.", "Source two.", "First explanation."])
        self.assertEqual(embedder.calls[1][0], ["Different explanation."])
        self.assertTrue(all(item["embedding_source"] == "new" for item in first["source_sentences"]))
        self.assertTrue(all(item["embedding_source"] == "cache" for item in second["source_sentences"]))

    def test_identical_explanation_sentence_reuses_cached_embedding(self) -> None:
        embedder = RecordingEmbedder()
        build_source_matches(
            "First source.", "Identical explanation.", report_id="R1",
            cache=self.cache, embed_batch=embedder, settings=self.settings,
        )
        second = build_source_matches(
            "New source.", "Identical explanation.", report_id="R2",
            cache=self.cache, embed_batch=embedder, settings=self.settings,
        )

        self.assertEqual(embedder.calls[1][0], ["New source."])
        self.assertEqual(
            second["explanation_sentences"][0]["embedding_source"], "cache"
        )

    def test_threshold_changes_only_relabel_existing_scores(self) -> None:
        embedder = RecordingEmbedder({
            "Source.": [1.0, 0.0],
            "Explanation.": [0.8, 0.6],
        })
        result = build_source_matches(
            "Source.", "Explanation.", report_id="R1",
            cache=self.cache, embed_batch=embedder, settings=self.settings,
        )
        calls_before = len(embedder.calls)
        low = apply_match_threshold(result, 0.70)
        high = apply_match_threshold(result, 0.90)

        self.assertEqual(len(embedder.calls), calls_before)
        self.assertEqual(low["explanation_sentences"][0]["best_match"]["threshold_status"],
                         "Above threshold")
        self.assertEqual(high["explanation_sentences"][0]["best_match"]["threshold_status"],
                         "Below threshold — closest match shown")
        self.assertNotIn("threshold_used", result)

    def test_changed_embedding_settings_cause_cache_misses(self) -> None:
        embedder = RecordingEmbedder()
        settings = [
            EmbeddingSettings("model-a", "SEMANTIC_SIMILARITY", 2),
            EmbeddingSettings("model-b", "SEMANTIC_SIMILARITY", 2),
            EmbeddingSettings("model-a", "CLUSTERING", 2),
            EmbeddingSettings("model-a", "SEMANTIC_SIMILARITY", 3),
        ]
        for index, setting in enumerate(settings):
            build_source_matches(
                "Source.", "Explanation.", report_id=f"R{index}",
                cache=self.cache, embed_batch=embedder, settings=setting,
            )

        self.assertEqual(len(embedder.calls), 4)
        self.assertTrue(all(call[0] == ["Source.", "Explanation."] for call in embedder.calls))
        self.assertEqual(self.cache.stats()["vectors"], 8)

    def test_each_explanation_sentence_uses_highest_scoring_source(self) -> None:
        embedder = RecordingEmbedder({
            "Alpha source.": [1.0, 0.0],
            "Beta source.": [0.0, 1.0],
            "Mostly alpha.": [0.8, 0.2],
        })
        result = build_source_matches(
            "Alpha source. Beta source.", "Mostly alpha.", report_id="R1",
            cache=self.cache, embed_batch=embedder, settings=self.settings,
        )
        rendered = apply_match_threshold(result, 0.70)
        best = rendered["explanation_sentences"][0]["best_match"]

        self.assertEqual(best["source_sentence_id"], "S001")
        self.assertEqual(best["source_text"], "Alpha source.")
        self.assertAlmostEqual(best["score"], 0.9701425, places=6)

    def test_sentence_split_preserves_exact_order_and_text(self) -> None:
        spans = split_sentences("  First sentence. Second sentence?\n- Final bullet.  ", "S")
        self.assertEqual(
            [(item.sentence_id, item.index, item.text) for item in spans],
            [
                ("S001", 0, "First sentence."),
                ("S002", 1, "Second sentence?"),
                ("S003", 2, "- Final bullet."),
            ],
        )

    def test_similarity_summary_recalculates_at_threshold(self) -> None:
        result = {
            "explanation_sentences": [
                {"ranked_source_matches": [{"score": 0.79}, {"score": 0.75}]},
                {"ranked_source_matches": [{"score": 0.60}, {"score": 0.20}]},
            ]
        }
        at_seventy = source_match_metrics([result], 0.70, ambiguity_margin=0.05)
        at_eighty = source_match_metrics([result], 0.80, ambiguity_margin=0.05)

        self.assertEqual(at_seventy["above_threshold_rate"], 50)
        self.assertEqual(at_seventy["below_threshold_rate"], 50)
        self.assertEqual(at_seventy["multiple_match_rate"], 50)
        self.assertEqual(at_seventy["ambiguous_match_rate"], 50)
        self.assertEqual(at_seventy["minimum_closest_match_score"], 0.60)
        self.assertEqual(at_eighty["above_threshold_rate"], 0)
        self.assertEqual(at_eighty["below_threshold_rate"], 100)

    def test_hover_html_preserves_markdown_blocks_and_escapes_html(self) -> None:
        explanation = "**Heading**\n\n* **Bold:** First sentence.\n* Second item with <script>."
        rendered = {
            "explanation_sentences": [
                {
                    "explanation_sentence_id": "E001", "explanation_text": "**Heading**",
                    "best_match": {"source_text": "Heading source.", "score": .9,
                                   "threshold_status": "Above threshold"},
                    "other_possible_matches": [],
                },
                {
                    "explanation_sentence_id": "E002",
                    "explanation_text": "* **Bold:** First sentence.",
                    "best_match": {"source_text": "First source.", "score": .8,
                                   "threshold_status": "Above threshold"},
                    "other_possible_matches": [],
                },
                {
                    "explanation_sentence_id": "E003",
                    "explanation_text": "* Second item with <script>.",
                    "best_match": {"source_text": "Second source.", "score": .7,
                                   "threshold_status": "Above threshold"},
                    "other_possible_matches": [],
                },
            ]
        }

        output = hoverable_explanation_html(explanation, rendered)
        self.assertIn("<strong>Heading</strong>", output)
        self.assertIn("<ul>", output)
        self.assertEqual(output.count("<li>"), 2)
        self.assertIn("<strong>Bold:</strong>", output)
        self.assertNotIn("**Heading**", output)
        self.assertIn("&lt;script&gt;", output)
        self.assertNotIn("<script>", output)

    def test_google_embedder_chunks_requests_at_api_batch_limit(self) -> None:
        calls = []

        class Models:
            def embed_content(self, *, model, contents, config):
                calls.append(list(contents))
                return SimpleNamespace(embeddings=[
                    SimpleNamespace(values=[float(index), 1.0])
                    for index, _text in enumerate(contents)
                ])

        client = SimpleNamespace(models=Models())
        embed = google_batch_embedder(client)
        texts = [f"sentence {index}" for index in range(205)]

        vectors = embed(texts, self.settings)

        self.assertEqual([len(batch) for batch in calls], [100, 100, 5])
        self.assertEqual(len(vectors), 205)


if __name__ == "__main__":
    unittest.main()
