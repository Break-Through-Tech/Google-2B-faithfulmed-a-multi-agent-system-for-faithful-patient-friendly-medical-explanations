"""Offline tests for the FastAPI demo server. Gemini and lookups are faked."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from agents.medical_lookup import MedicalLookup
from demo import server
from evals.system_demo import SystemExample, run_system_example

SOURCE = "The patient has a history of hypertension. Continue the current treatment plan."
FACTS = json.dumps([
    {"fact_id": "F-001", "fact_text": "The patient has a history of hypertension.", "fact_type": "diagnosis",
     "source_evidence": ["The patient has a history of hypertension."]},
])
OUTPUTS = {
    "Extractor": FACTS,
    "Simplifier": "PATIENT-FRIENDLY VERSION:\nYou have high blood pressure. It is not caused by stress.\n\nREVIEW FLAGS:\n- None.",
    "Refiner": "You have high blood pressure. Keep following your treatment plan.",
    "Readability": "You have high blood pressure. Keep following your treatment plan.",
}
EVALUATION = json.dumps({
    "status": "scored", "status_reason": "ok",
    "claims": [
        {"claim_id": "C1", "claim_span_id": "E001", "claim_text": "History of hypertension.",
         "classification": "supported", "clinically_meaningful": True, "source_span_ids": ["S001"], "reason": "Stated."},
        {"claim_id": "C2", "claim_span_id": "E002", "claim_text": "Continue the plan.",
         "classification": "supported", "clinically_meaningful": True, "source_span_ids": ["S002"], "reason": "Stated."},
    ],
})


def verdict(decision: str) -> str:
    return json.dumps({
        "overall_decision": decision,
        "explanation_accuracy": {"prediction": "yes" if decision == "accept" else "no", "reason": "brief"},
        "sentences": [],
        "unsupported_claims": [] if decision == "accept" else [{
            "draft_quote": "It is not caused by stress.", "issue_type": "unsupported", "source_quote": None,
            "instruction_to_refiner": "Remove the stress claim.",
        }],
        "omissions": [],
    })


async def _emit(text: str, on_text) -> str:
    accumulated = ""
    for start in range(0, len(text), 12):
        accumulated += text[start:start + 12]
        if on_text:
            on_text(text[start:start + 12], accumulated)
    return text


async def fake_agent_stream(agent, *, state, message, on_text=None) -> str:
    if agent.name == "FactualEvaluator":
        return await _emit(EVALUATION, on_text)
    return await _emit(OUTPUTS[agent.name], on_text)


def fake_verifier(decisions: list[str]):
    async def run(source_text, facts, draft, model, *, on_text=None, reference_definitions=None) -> str:
        return await _emit(verdict(decisions.pop(0)), on_text)
    return run


async def fake_agent_once(agent, *, state, message) -> str:
    return await fake_agent_stream(agent, state=state, message=message)


def faked_model(decisions: list[str]):
    return (
        patch("agents.production_pipeline.run_agent_stream", fake_agent_stream),
        patch("agents.factual_evaluator.run_agent_stream", fake_agent_stream),
        patch("agents.factual_evaluator.run_agent_once", fake_agent_once),
        patch("agents.production_pipeline.run_production_verifier", fake_verifier(decisions)),
        patch("agents.production_pipeline.MedicalLookup.from_env", MedicalLookup.disabled),
    )


def parse_sse(body: str) -> list[dict]:
    return [json.loads(block[6:]) for block in body.split("\n\n") if block.startswith("data: ")]


class DemoServerTests(unittest.TestCase):
    def test_live_run_streams_every_stage_and_the_scored_result(self) -> None:
        patches = faked_model(["revise", "accept"])
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-only", "DEMO_LIVE": "1"}):
            response = TestClient(server.app).post("/api/run", json={"source_text": SOURCE})
        self.assertEqual(response.status_code, 200)
        events = parse_sse(response.text)
        types = [event["type"] for event in events]
        self.assertEqual(types[0], "run_started")
        self.assertEqual(types[-1], "done")
        completed = [event["stage"]["stage_id"] for event in events if event["type"] == "stage_completed"]
        self.assertEqual(completed, ["extractor", "term_lookup", "simplifier", "verifier-1", "refiner-1",
                                     "verifier-2", "readability"])
        self.assertIn("stage_chunk", types)
        self.assertIn("evaluator_chunk", types)
        # Chunks carry only new text, so concatenating them rebuilds the output.
        simplifier = "".join(e["delta"] for e in events if e["type"] == "stage_chunk" and e["stage_id"] == "simplifier")
        self.assertEqual(simplifier, OUTPUTS["Simplifier"])
        started = next(e for e in events if e["type"] == "stage_started")
        self.assertNotIn("input", started["stage"])
        result = next(e for e in events if e["type"] == "result")
        self.assertEqual(result["factual"]["factual_fidelity_estimate"], 100.0)
        self.assertEqual([c["source_span_ids"] for c in result["claims"]], [["S001"], ["S002"]])
        self.assertEqual([s["span_id"] for s in result["explanation_spans"]], ["E001", "E002"])

    def test_replay_matches_the_live_event_contract(self) -> None:
        patches = faked_model(["accept"])
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            record = asyncio.run(run_system_example(
                SystemExample("Test", "scratch", "T1", SOURCE, {}, "test"), "model", "model",
                cache=None, persist=False,
            ))
        self.assertIsNone(record["factual_evaluator"]["error"])
        record["run_id"] = "replay-test"

        async def collect() -> list[dict]:
            return [json.loads(chunk[6:]) async for chunk in server.replay_events(record, speed=8)
                    if chunk.startswith("data: ")]

        events = asyncio.run(collect())
        self.assertEqual(events[0]["meta"]["mode"], "replay")
        readability = "".join(e["delta"] for e in events if e.get("stage_id") == "readability")
        self.assertEqual(readability, OUTPUTS["Readability"])
        self.assertEqual(events[-2]["type"], "result")
        self.assertEqual(events[-1]["type"], "done")

    def test_live_run_is_refused_when_disabled(self) -> None:
        with patch.dict(os.environ, {"DEMO_LIVE": "0"}):
            response = TestClient(server.app).post("/api/run", json={"source_text": SOURCE})
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
