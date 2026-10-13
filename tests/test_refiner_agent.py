"""
FaithfulMed — Refiner agent helper tests.

Offline and synthetic only. These exercise the Refiner's pure-Python helpers
(`_verdict_to_text`, `describe_refinement`, `RefinerResult`). They do not import
ADK and never call Gemini; `agents.refiner` imports ADK lazily inside the agent
builder, so importing the module here is safe.
"""

from __future__ import annotations

import json
import unittest

from agents.refiner import (
    RefinerResult,
    _verdict_to_text,
    describe_refinement,
)


class RefinerAgentHelperTests(unittest.TestCase):

    # ---- _verdict_to_text ----

    def test_verdict_dict_becomes_json(self) -> None:
        verdict = {"faithful": False, "unsupported_claims": ["you have cancer"], "omissions": []}
        text = _verdict_to_text(verdict)
        self.assertEqual(json.loads(text), verdict)

    def test_verdict_string_passes_through(self) -> None:
        raw = '{"faithful": true, "unsupported_claims": [], "omissions": []}'
        self.assertEqual(_verdict_to_text(raw), raw)

    def test_verdict_other_coerced_to_str(self) -> None:
        self.assertEqual(_verdict_to_text(None), "None")

    # ---- describe_refinement ----

    def test_describe_unchanged(self) -> None:
        draft = "You have high blood pressure."
        result = describe_refinement(draft, draft)
        self.assertIsInstance(result, RefinerResult)
        self.assertFalse(result.changed)
        self.assertEqual(result.minimal_edit_ratio, 1.0)
        self.assertEqual(result.revised_draft, draft)

    def test_describe_changed_stays_similar(self) -> None:
        draft = "You have high blood pressure and you also have cancer."
        revised = "You have high blood pressure."
        result = describe_refinement(draft, revised)
        self.assertTrue(result.changed)
        self.assertLess(result.minimal_edit_ratio, 1.0)
        self.assertGreater(result.minimal_edit_ratio, 0.5)
        self.assertEqual(result.draft, draft)
        self.assertEqual(result.revised_draft, revised)

    def test_describe_whitespace_only_is_unchanged(self) -> None:
        draft = "Rest and drink water."
        result = describe_refinement(draft, "  Rest and drink water.  ")
        self.assertFalse(result.changed)


if __name__ == "__main__":
    unittest.main()
