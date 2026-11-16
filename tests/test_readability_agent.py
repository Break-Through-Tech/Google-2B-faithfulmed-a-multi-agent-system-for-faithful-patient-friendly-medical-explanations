"""Offline checks for the Readability agent prompt contract."""

from __future__ import annotations

import unittest

from agents.readability import READABILITY_INSTRUCTION


class ReadabilityAgentTests(unittest.TestCase):
    def test_instruction_requires_terminal_punctuation_for_bullets(self) -> None:
        self.assertIn("end every bullet item with a period", READABILITY_INSTRUCTION)
        self.assertIn("no bullet item is missing terminal punctuation", READABILITY_INSTRUCTION)

    def test_instruction_preserves_medication_modifier_scope(self) -> None:
        self.assertIn("dose, route, and frequency attached only to that medication", READABILITY_INSTRUCTION)
        self.assertIn("Never copy", READABILITY_INSTRUCTION)
        self.assertIn("do not add one", READABILITY_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
