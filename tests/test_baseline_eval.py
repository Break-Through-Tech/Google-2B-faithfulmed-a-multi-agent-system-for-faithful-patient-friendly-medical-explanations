
"""
FaithfulMed — Single Gemini Baseline evaluation tests
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

NOTEBOOKS = Path(__file__).resolve().parent.parent / "evals"
sys.path.append(str(NOTEBOOKS))

import baseline_eval

from baseline_eval import (
    load_samples,
    save_record,
    load_completed_ids,
    evaluate_split,
)


class BaselineEvaluationTests(unittest.TestCase):

    def setUp(self) -> None:
        """Create temporary files for each test."""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.root = Path(self.temp_dir.name)
        self.annotations_dir = self.root / "annotations"
        self.results_file = self.root / "runs" / "results.jsonl"

        # Replace the real paths so tests don't touch project data.
        self.annotations_patch = patch.object(
            baseline_eval, "ANNOTATIONS_DIR", self.annotations_dir
        )
        self.results_patch = patch.object(
            baseline_eval, "RESULTS_FILE", self.results_file
        )

        self.annotations_patch.start()
        self.results_patch.start()

        self.addCleanup(self.annotations_patch.stop)
        self.addCleanup(self.results_patch.stop)

        self.root_patch = patch.object(
            baseline_eval, "ROOT", self.root
        )
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def create_sample(
        self,
        split: str,
        filename: str,
        data: dict,
    ) -> None:
        """Create a sample JSON file in a temporary dataset."""

        folder = self.annotations_dir / split / "adjudicated"
        folder.mkdir(parents=True, exist_ok=True)

        (folder / filename).write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def test_load_samples(self) -> None:
        """Loads sample IDs and source text from JSON."""

        self.create_sample(
            "dev",
            "sample_001.json",
            {
                "example_id": "sample_001",
                "source_text": "The patient should drink water.",
                "source_url": "https://example.com",
            },
        )

        samples = load_samples("dev")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["example_id"], "sample_001")
        self.assertEqual(
            samples[0]["source_text"],
            "The patient should drink water.",
        )
        self.assertEqual(
            samples[0]["source_file"],
            str(Path("annotations") / "dev" / "adjudicated" / "sample_001.json"),
        )

    def test_load_samples_ignores_missing_or_blank_text(self) -> None:
        """Skips samples with missing or blank source text."""

        self.create_sample(
            "dev",
            "missing.json",
            {"example_id": "missing"},
        )
        self.create_sample(
            "dev",
            "blank.json",
            {
                "example_id": "blank",
                "source_text": "   ",
            },
        )

        samples = load_samples("dev")

        self.assertEqual(samples, [])

    def test_load_samples_invalid_split(self) -> None:
        """Rejects an unsupported dataset split."""

        with self.assertRaises(ValueError):
            load_samples("invalid")

    def test_save_record(self) -> None:
        """Saves a record that can be read back from JSONL."""

        record = {
            "example_id": "sample_001",
            "status": "success",
            "metrics": {"word_count": 20},
        }

        save_record(record)

        lines = self.results_file.read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), record)

    def test_load_completed_ids(self) -> None:
        """Loads only successful samples for the requested split/model."""

        records = [
            {
                "example_id": "sample_001",
                "split": "dev",
                "model": "test-model",
                "status": "success",
            },
            {
                "example_id": "sample_002",
                "split": "dev",
                "model": "test-model",
                "status": "failed",
            },
            {
                "example_id": "sample_003",
                "split": "test",
                "model": "test-model",
                "status": "success",
            },
        ]

        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.results_file.write_text(
            "".join(json.dumps(r) + "\n" for r in records),
            encoding="utf-8",
        )

        completed = load_completed_ids("dev", "test-model")

        self.assertEqual(completed, {"sample_001"})

    def test_evaluate_split_saves_generated_output(self) -> None:
        """Tests the evaluation pipeline without calling Gemini."""

        self.create_sample(
            "dev",
            "sample_001.json",
            {
                "example_id": "sample_001",
                "source_text": "The patient should drink water.",
            },
        )

        generated_text = "The patient should drink enough water."
        metrics = {
            "flesch_kincaid_grade": 3.0,
            "smog_index": 4.0,
            "jargon_density": 0.0,
            "word_count": 7,
            "meets_target": True,
            "is_refusal": False,
        }

        with (
            patch.object(
                baseline_eval,
                "run_baseline",
                new_callable=AsyncMock,
                return_value=generated_text,
            ) as mock_agent,
            patch.object(
                baseline_eval,
                "evaluate_single",
                return_value=metrics,
            ) as mock_evaluator,
        ):
            counts = asyncio.run(
                evaluate_split(
                    "dev",
                    model="test-model",
                )
            )

        mock_agent.assert_awaited_once_with(
            "The patient should drink water.",
            model="test-model",
        )
        mock_evaluator.assert_called_once_with(generated_text)

        self.assertEqual(counts["total"], 1)
        self.assertEqual(counts["success"], 1)
        self.assertEqual(counts["failed"], 0)

        saved = json.loads(
            self.results_file.read_text(encoding="utf-8").strip()
        )

        self.assertEqual(saved["example_id"], "sample_001")
        self.assertEqual(saved["generated_text"], generated_text)
        self.assertEqual(saved["metrics"], metrics)
        self.assertEqual(saved["status"], "success")

    def test_evaluate_split_records_failure(self) -> None:
        """Records a failed model call without crashing the whole run."""

        self.create_sample(
            "dev",
            "sample_001.json",
            {
                "example_id": "sample_001",
                "source_text": "The patient should rest.",
            },
        )

        with patch.object(
            baseline_eval,
            "run_baseline",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Simulated API error"),
        ):
            counts = asyncio.run(
                evaluate_split(
                    "dev",
                    model="test-model",
                )
            )

        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["success"], 0)

        saved = json.loads(
            self.results_file.read_text(encoding="utf-8").strip()
        )

        self.assertEqual(saved["status"], "failed")
        self.assertIn("Simulated API error", saved["error"])


if __name__ == "__main__":
    unittest.main()
