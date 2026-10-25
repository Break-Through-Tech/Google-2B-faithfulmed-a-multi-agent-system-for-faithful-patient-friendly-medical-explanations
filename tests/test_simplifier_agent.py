"""Offline tests for the Simplifier's input and state contract."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from agents.simplifier import SIMPLIFIER_INSTRUCTION, run_simplifier


def run_immediately(coroutine):
    """Complete a coroutine that only awaits mocked immediate work.

    This keeps the test independent of an event-loop implementation; it is not a
    substitute for running the actual Gemini-backed agent.
    """

    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("The mocked coroutine unexpectedly yielded control.")


class SimplifierAgentTests(unittest.TestCase):
    def test_instruction_has_all_state_placeholders(self) -> None:
        for placeholder in ("{source_text}", "{atoms}", "{glossary}"):
            self.assertIn(placeholder, SIMPLIFIER_INSTRUCTION)

    def test_instruction_documents_the_current_output_sections(self) -> None:
        self.assertIn("PATIENT-FRIENDLY VERSION:", SIMPLIFIER_INSTRUCTION)
        self.assertIn("REVIEW FLAGS:", SIMPLIFIER_INSTRUCTION)
        self.assertIn("End every bullet item with a period", SIMPLIFIER_INSTRUCTION)

    def test_blank_or_nonstring_inputs_are_rejected_without_a_model_call(self) -> None:
        cases = [
            ("", "[{}]", "No glossary entries provided."),
            ("Source", "", "No glossary entries provided."),
            (None, "[{}]", "No glossary entries provided."),
            ("Source", None, "No glossary entries provided."),
            ("Source", "[{}]", ""),
        ]
        for source_text, atoms, glossary in cases:
            with self.subTest(source_text=source_text, atoms=atoms, glossary=glossary):
                with self.assertRaises(ValueError):
                    run_immediately(run_simplifier(source_text, atoms, glossary=glossary))

    @patch("agents.simplifier.create_simplifier_agent", return_value="agent")
    @patch("agents.simplifier.run_agent_once", new_callable=AsyncMock)
    def test_valid_run_passes_default_glossary_in_state(self, mock_run, mock_create) -> None:
        mock_run.return_value = "PATIENT-FRIENDLY VERSION:\nExample\n\nREVIEW FLAGS:\nNone"

        result = run_immediately(run_simplifier("Source", "[facts]"))

        self.assertEqual(result, mock_run.return_value)
        mock_create.assert_called_once_with("gemini-2.0-flash")
        mock_run.assert_awaited_once_with(
            "agent",
            state={
                "source_text": "Source",
                "atoms": "[facts]",
                "glossary": "No glossary entries provided.",
            },
            message="Create the patient-friendly medical text.",
        )


if __name__ == "__main__":
    unittest.main()
