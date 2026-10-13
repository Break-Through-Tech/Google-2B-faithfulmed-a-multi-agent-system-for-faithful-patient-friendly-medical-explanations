"""
FaithfulMed — ADK pipeline wiring tests.

Offline only. These check the pipeline's stage order and the draft-mode Verifier
prompt. Importing notebooks.adk_pipeline does not import ADK (ADK is imported
lazily inside the builders), so no Gemini call or ADK dependency is needed here.
"""

from __future__ import annotations

import unittest

from pathlib import Path
import sys

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
sys.path.append(str(NOTEBOOKS))

from adk_pipeline import (
    DRAFT_VERIFIER_INSTRUCTION,
    MAX_REFINE_ITERATIONS,
    pipeline_plan,
)


class AdkPipelineWiringTests(unittest.TestCase):

    def test_loop_is_bounded(self) -> None:
        self.assertEqual(MAX_REFINE_ITERATIONS, 2)

    def test_pipeline_plan_order(self) -> None:
        self.assertEqual(
            pipeline_plan(),
            [
                "Extractor",
                "Simplifier",
                "Verifier",
                "Refiner",
                "Verifier",
                "Refiner",
                "Readability",
            ],
        )

    def test_refine_loop_runs_verifier_before_refiner(self) -> None:
        plan = pipeline_plan()
        first_verifier = plan.index("Verifier")
        first_refiner = plan.index("Refiner")
        self.assertLess(first_verifier, first_refiner)

    def test_draft_verifier_prompt_reads_source_and_draft(self) -> None:
        self.assertIn("{source_text}", DRAFT_VERIFIER_INSTRUCTION)
        self.assertIn("{draft}", DRAFT_VERIFIER_INSTRUCTION)

    def test_draft_verifier_prompt_requests_refiner_fields(self) -> None:
        for field in ("faithful", "unsupported_claims", "omissions", "reading_level_ok"):
            self.assertIn(field, DRAFT_VERIFIER_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
