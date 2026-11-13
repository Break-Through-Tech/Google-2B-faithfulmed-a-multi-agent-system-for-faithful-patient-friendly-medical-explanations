"""Offline tests for the observable full-system demo pipeline."""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.medical_lookup import MedicalLookup
from agents.production_pipeline import MAX_REVISION_CYCLES, run_production_pipeline, split_simplifier_output
from agents.verifier import PRODUCTION_VERIFIER_PROMPT, production_verifier_prompt


def verdict(decision: str) -> str:
    return json.dumps({
        "overall_decision": decision,
        "explanation_accuracy": {"prediction": "yes" if decision == "accept" else "no", "reason": "brief"},
        "sentences": [],
        "unsupported_claims": [] if decision == "accept" else [{
            "draft_quote": "unsupported", "issue_type": "unsupported", "source_quote": None,
            "instruction_to_refiner": "Remove unsupported.",
        }],
        "omissions": [],
    })


class ProductionPipelineTests(unittest.TestCase):
    def test_production_prompt_is_separate_and_appends_only_inputs(self) -> None:
        rendered = production_verifier_prompt("Report", '[{"fact_id":"F-001"}]', "Draft")
        self.assertTrue(rendered.startswith(PRODUCTION_VERIFIER_PROMPT))
        payload = json.loads(rendered.rsplit("INPUT JSON:\n", 1)[1])
        self.assertEqual(payload["original_medical_report"], "Report")
        self.assertEqual(payload["extracted_facts"][0]["fact_id"], "F-001")
        self.assertFalse(any("gold" in key.lower() for key in payload))
        self.assertIn('"overall_decision": "accept|revise"', PRODUCTION_VERIFIER_PROMPT)
        self.assertIn("CLOSED-BOOK SOURCE FIDELITY", PRODUCTION_VERIFIER_PROMPT)
        self.assertIn("vitamin B12 to vitamin B6", PRODUCTION_VERIFIER_PROMPT)
        self.assertIn("a side effect of warfarin to aspirin", PRODUCTION_VERIFIER_PROMPT)
        self.assertIn("the left kidney to the right kidney", PRODUCTION_VERIFIER_PROMPT)
        self.assertIn('"history of hypertension" becoming "you have hypertension"', PRODUCTION_VERIFIER_PROMPT)

    def test_split_simplifier_output(self) -> None:
        wrapped = "PATIENT-FRIENDLY VERSION:\n\nYou have a history of high blood pressure.\n\nREVIEW FLAGS:\n- None."
        self.assertEqual(split_simplifier_output(wrapped), ("You have a history of high blood pressure.", "- None."))
        bold = "**PATIENT-FRIENDLY VERSION:**\nText.\n\n**REVIEW FLAGS:**\n- Source and facts disagree."
        self.assertEqual(split_simplifier_output(bold), ("Text.", "- Source and facts disagree."))
        self.assertEqual(split_simplifier_output("Plain text only."), ("Plain text only.", None))

    @patch("agents.production_pipeline.create_readability_agent", return_value=SimpleNamespace(name="Readability"))
    @patch("agents.production_pipeline.create_simplifier_agent", return_value=SimpleNamespace(name="Simplifier"))
    @patch("agents.production_pipeline.create_extractor_agent", return_value=SimpleNamespace(name="Extractor"))
    def test_review_flags_never_reach_verifier_or_final(self, *_mocks) -> None:
        raw = "PATIENT-FRIENDLY VERSION:\nPatient text.\n\nREVIEW FLAGS:\n- None."
        seen = {}

        async def executor(agent, state, message, on_text):
            seen[agent.name] = dict(state)
            text = {"Extractor": json.dumps([{"fact_id": "F-001", "fact_text": "Finding", "fact_type": "diagnosis",
                                              "source_evidence": ["Finding."]}]),
                    "Simplifier": raw}.get(agent.name, state.get("draft"))  # Readability echoes its input
            on_text(text, text)
            return text

        async def verifier_runner(source, facts, draft, model, on_text, definitions):
            seen["Verifier"] = draft
            return verdict("accept")

        result = asyncio.run(run_production_pipeline(
            "Finding.", agent_executor=executor, verifier_runner=verifier_runner, lookup=MedicalLookup.disabled()))
        self.assertEqual(seen["Verifier"], "Patient text.")
        self.assertEqual(seen["Readability"]["draft"], "Patient text.")
        self.assertEqual(result.final_explanation, "Patient text.")
        self.assertEqual(result.trace["results"]["simplifier"], raw)
        self.assertEqual(result.trace["results"]["simplifier_review_flags"], "- None.")

    @patch("agents.production_pipeline.create_readability_agent", return_value=SimpleNamespace(name="Readability"))
    @patch("agents.production_pipeline.create_refiner_agent", return_value=SimpleNamespace(name="Refiner"))
    @patch("agents.production_pipeline.create_simplifier_agent", return_value=SimpleNamespace(name="Simplifier"))
    @patch("agents.production_pipeline.create_extractor_agent", return_value=SimpleNamespace(name="Extractor"))
    def test_revision_then_accept_runs_readability(self, *_mocks) -> None:
        outputs = {
            "Extractor": json.dumps([{"fact_id": "F-001", "fact_text": "Finding", "fact_type": "diagnosis", "source_evidence": ["Finding."]}]),
            "Simplifier": "Initial draft.",
            "Refiner": "Revised draft.",
            "Readability": "Final explanation.",
        }

        async def executor(agent, state, message, on_text):
            text = outputs[agent.name]
            on_text(text, text)
            return text

        decisions = iter((verdict("revise"), verdict("accept")))

        async def verifier_runner(source, facts, draft, model, on_text, definitions):
            text = next(decisions)
            on_text(text, text)
            return text

        result = asyncio.run(run_production_pipeline(
            "Finding.", agent_executor=executor, verifier_runner=verifier_runner,
            lookup=MedicalLookup.disabled(),
        ))
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_explanation, "Final explanation.")
        self.assertEqual([item["agent"] for item in result.trace["stages"]],
                         ["Extractor", "TermLookup", "Simplifier", "Verifier", "Refiner", "Verifier", "Readability"])
        self.assertEqual(len(result.trace["results"]["verifier_refiner_iterations"]), 2)

    @patch("agents.production_pipeline.create_readability_agent", return_value=SimpleNamespace(name="Readability"))
    @patch("agents.production_pipeline.create_refiner_agent", return_value=SimpleNamespace(name="Refiner"))
    @patch("agents.production_pipeline.create_simplifier_agent", return_value=SimpleNamespace(name="Simplifier"))
    @patch("agents.production_pipeline.create_extractor_agent", return_value=SimpleNamespace(name="Extractor"))
    def test_two_revision_limit_does_not_present_unaccepted_final(self, *_mocks) -> None:
        counts = {"Refiner": 0, "Readability": 0, "Verifier": 0}

        async def executor(agent, state, message, on_text):
            if agent.name == "Extractor":
                text = json.dumps([{"fact_id": "F-001", "fact_text": "Finding", "fact_type": "diagnosis", "source_evidence": ["Finding."]}])
            elif agent.name == "Simplifier":
                text = "Draft 0."
            elif agent.name == "Refiner":
                counts["Refiner"] += 1
                text = f"Draft {counts['Refiner']}."
            else:
                counts["Readability"] += 1
                text = "Should not run."
            on_text(text, text)
            return text

        async def verifier_runner(source, facts, draft, model, on_text, definitions):
            counts["Verifier"] += 1
            text = verdict("revise")
            on_text(text, text)
            return text

        result = asyncio.run(run_production_pipeline(
            "Finding.", agent_executor=executor, verifier_runner=verifier_runner,
            lookup=MedicalLookup.disabled(),
        ))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.final_explanation)
        self.assertEqual(result.trace["status"], "not_accepted")
        self.assertEqual(counts["Refiner"], MAX_REVISION_CYCLES)
        self.assertEqual(counts["Verifier"], MAX_REVISION_CYCLES + 1)
        self.assertEqual(counts["Readability"], 0)
        self.assertEqual(result.trace["latest_draft"], "Draft 2.")

    @patch("agents.production_pipeline.create_simplifier_agent", return_value=SimpleNamespace(name="Simplifier"))
    @patch("agents.production_pipeline.create_extractor_agent", return_value=SimpleNamespace(name="Extractor"))
    def test_failure_preserves_completed_stage_trace(self, *_mocks) -> None:
        extracted = json.dumps([{
            "fact_id": "F-001",
            "fact_text": "Finding",
            "fact_type": "diagnosis",
            "source_evidence": ["Finding."],
        }])

        async def executor(agent, state, message, on_text):
            if agent.name == "Extractor":
                on_text(extracted, extracted)
                return extracted
            raise RuntimeError("simplifier unavailable")

        result = asyncio.run(run_production_pipeline(
            "Finding.", agent_executor=executor, lookup=MedicalLookup.disabled(),
        ))
        self.assertFalse(result.accepted)
        self.assertEqual(result.trace["status"], "failed")
        self.assertEqual(result.trace["results"]["extractor"], extracted)
        self.assertEqual(result.trace["stages"][0]["status"], "completed")
        self.assertEqual(result.trace["stages"][1]["status"], "skipped")
        self.assertEqual(result.trace["stages"][2]["status"], "failed")
        self.assertEqual(result.trace["error"], "simplifier unavailable")


if __name__ == "__main__":
    unittest.main()
