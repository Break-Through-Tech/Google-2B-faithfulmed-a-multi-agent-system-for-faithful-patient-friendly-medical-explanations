"""Offline tests for the leaderboard runner: retries, resume, usage capture. No model calls."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents import _report_usage, resolve_model
from agents.medical_lookup import MedicalLookup
from evals import leaderboard as lb
from evals import leaderboard_runner as runner

CATALOG = lb.load_catalog()
PACKAGES = {"google-adk": "2.14.0", "google-genai": "2.34.0", "litellm": "1.106.2"}


def note(example_id: str) -> dict:
    return {"dataset": "MTSamples", "split": "dev", "example_id": example_id, "source_file": f"{example_id}.json",
            "source_sha256": lb.sha256_text(f"Source {example_id}.")}


MANIFEST = {"schema": lb.MANIFEST_SCHEMA, "manifest_id": "unit", "purpose": "pilot", "split": "dev",
            "datasets": ["MTSamples"], "notes": [note("A"), note("B")]}
MANIFEST["manifest_sha256"] = lb.stable_hash(MANIFEST["notes"])
EXAMPLES = [SimpleNamespace(dataset="MTSamples", example_id=e, source_text=f"Source {e}.",
                            references={"adjudicated_facts": []}) for e in ("A", "B")]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ServerError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"{code} server says no")
        self.code = code


def recorder(clock: FakeClock, max_attempts: int = 3) -> runner.CallRecorder:
    async def sleep(seconds: float) -> None:
        clock.now += seconds

    return runner.CallRecorder(max_attempts=max_attempts, base_delay=1.0, clock=clock, sleep=sleep,
                               rng=__import__("random").Random(0))


class RetryTests(unittest.TestCase):
    def test_retryable_classification(self) -> None:
        self.assertTrue(runner.is_retryable(ServerError(503)))
        self.assertTrue(runner.is_retryable(ServerError(429)))
        self.assertFalse(runner.is_retryable(ServerError(400)))
        self.assertFalse(runner.is_retryable(ValueError("bad JSON")))
        self.assertTrue(runner.is_retryable(type("RateLimitError", (Exception,), {})("slow down")))
        self.assertTrue(runner.is_retryable(RuntimeError("503 UNAVAILABLE. The model is overloaded.")))

    def test_bounded_retries_record_usage_and_waits(self) -> None:
        clock = FakeClock()
        rec = recorder(clock)
        outcomes = [ServerError(503), ServerError(503), "ok"]

        async def fn(on_usage):
            clock.now += 2.0
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            on_usage({"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 1, "method": "total_minus_input"})
            return "text"

        self.assertEqual(asyncio.run(rec.call("Extractor", fn)), "text")
        self.assertEqual([c["status"] for c in rec.calls], ["error", "error", "ok"])
        self.assertEqual([c["retried"] for c in rec.calls], [True, True, False])
        summary = lb.summarize_calls(rec.calls, latency_s=clock.now)
        self.assertEqual(summary["retries"], 2)
        self.assertGreaterEqual(summary["retry_wait_s"], 3.0)  # 1 s + 2 s backoff, plus jitter
        self.assertEqual(clock.now, 6.0 + summary["retry_wait_s"])  # waits are inside the latency
        self.assertEqual(summary["tokens"], {"input": 10, "output": 5, "thinking": 1})

    def test_gives_up_after_limit_and_never_retries_other_errors(self) -> None:
        clock = FakeClock()
        rec = recorder(clock, max_attempts=2)

        async def always_503(on_usage):
            raise ServerError(503)

        with self.assertRaises(ServerError):
            asyncio.run(rec.call("Simplifier", always_503))
        self.assertEqual(len(rec.calls), 2)
        self.assertFalse(rec.calls[-1]["retried"])

        rec2 = recorder(FakeClock())

        async def bad(on_usage):
            raise ValueError("not JSON")

        with self.assertRaises(ValueError):
            asyncio.run(rec2.call("Verifier", bad))
        self.assertEqual(len(rec2.calls), 1)


class UsageCaptureTests(unittest.TestCase):
    def test_partial_events_are_not_counted(self) -> None:
        seen = []
        usage = SimpleNamespace(prompt_token_count=5, candidates_token_count=3, thoughts_token_count=0,
                                total_token_count=8)
        _report_usage(SimpleNamespace(partial=True, usage_metadata=usage), seen.append)
        _report_usage(SimpleNamespace(partial=False, usage_metadata=usage), seen.append)
        _report_usage(SimpleNamespace(partial=False, usage_metadata=None), seen.append)
        self.assertEqual(len(seen), 1)

    def test_groq_ids_route_through_litellm(self) -> None:
        try:
            from google.adk.models.lite_llm import LiteLlm
        except ImportError:
            self.skipTest("google-adk with LiteLLM is not installed")
        self.assertIsInstance(resolve_model("groq/openai/gpt-oss-120b"), LiteLlm)
        self.assertEqual(resolve_model("gemma-4-31b-it"), "gemma-4-31b-it")

    @patch("agents.production_pipeline.create_readability_agent", return_value=SimpleNamespace(name="Readability"))
    @patch("agents.production_pipeline.create_refiner_agent", return_value=SimpleNamespace(name="Refiner"))
    @patch("agents.production_pipeline.create_simplifier_agent", return_value=SimpleNamespace(name="Simplifier"))
    @patch("agents.production_pipeline.create_extractor_agent", return_value=SimpleNamespace(name="Extractor"))
    def test_every_pipeline_stage_and_revision_is_counted(self, *_mocks) -> None:
        facts = json.dumps([{"fact_id": "F-001", "fact_text": "Finding.", "fact_type": "diagnosis",
                             "source_evidence": ["Source A."]}])

        def verdict(decision):
            return json.dumps({"overall_decision": decision,
                               "explanation_accuracy": {"prediction": "yes", "reason": "r"}, "sentences": [],
                               "unsupported_claims": [], "omissions": []})

        verdicts = [verdict("revise"), verdict("accept")]
        texts = {"Extractor": facts, "Simplifier": "Draft.", "Refiner": "Revised.", "Readability": "Final."}
        models = []

        async def fake_run(agent, *, state, message, on_usage=None):
            models.append(getattr(agent, "model", None))
            text = verdicts.pop(0) if agent.name == "Verifier" else texts[agent.name]
            on_usage({"input_tokens": 100, "output_tokens": 10, "thinking_tokens": 0, "method": "total_minus_input"})
            return text

        config = lb.configuration(CATALOG, "gemma-system")
        rec = recorder(FakeClock())
        with patch.object(runner, "run_agent_once", fake_run), \
                patch.object(runner, "create_verifier_agent",
                             side_effect=lambda m: SimpleNamespace(name="Verifier", model=m)):
            result = asyncio.run(runner.generate_system(config, EXAMPLES[0], rec, MedicalLookup.disabled()))
        self.assertEqual(result["status"], lb.COMPLETED)
        self.assertEqual(result["final_output"], "Final.")
        stages = [c["stage"] for c in rec.calls]
        self.assertEqual(stages, ["Extractor", "Simplifier", "Verifier", "Refiner", "Verifier", "Readability"])
        summary = lb.summarize_calls(rec.calls, latency_s=1.0)
        self.assertEqual(summary["tokens"]["input"], 600)
        self.assertTrue(summary["usage_complete"])
        self.assertIn("gemma-4-31b-it", models)  # the Verifier ran on the configuration's model


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.attempts = Path(self.tmp.name) / "attempts.jsonl"
        self.evaluations = Path(self.tmp.name) / "evaluations.jsonl"
        self.fail_b = True
        self.judge_fails = False

    def tearDown(self) -> None:
        self.tmp.cleanup()

    async def fake_generate(self, config, example, rec, lookup):
        async def fn(on_usage):
            if example.example_id == "B" and self.fail_b:
                raise ValueError("Extractor returned invalid JSON")
            on_usage({"input_tokens": 50, "output_tokens": 20, "thinking_tokens": 0, "method": "total_minus_input"})
            return f"Plain text for {example.example_id}."
        text = await rec.call("Baseline", fn)
        return {"status": lb.COMPLETED, "final_output": text, "error": None, "latest_draft": None, "trace": None}

    async def fake_judge(self, attempt, example, evaluator, rec):
        if self.judge_fails:
            return "failed", None, None, None, "503"
        metrics = {"claims_returned": 2, "claims_assessed": 2, "claims_skipped": 0, "claim_validation_coverage": 100.0,
                   "supported_claims": 2, "uncertain_claims": 0, "factual_fidelity_estimate": 100.0,
                   "clinically_meaningful_hallucination": False}
        return "scored", "{}", {"status": "scored"}, metrics, None

    def run_once(self, **kwargs) -> dict:
        return asyncio.run(runner.run_leaderboard(
            MANIFEST, CATALOG, ["gemini-baseline"], examples=EXAMPLES, attempts_path=self.attempts,
            evaluations_path=self.evaluations, generators={"baseline": self.fake_generate}, judge=self.fake_judge,
            recorder_factory=lambda: recorder(FakeClock()), packages=PACKAGES, log=lambda _: None, **kwargs))

    def lines(self, path: Path) -> list:
        return lb.load_jsonl(path)[0]

    def test_resume_skips_done_preserves_failures_and_bounds_retries(self) -> None:
        first = self.run_once()
        self.assertEqual((first["generated"], first["generation_failed"], first["judged"]), (1, 1, 1))
        attempts = self.lines(self.attempts)
        self.assertEqual([a["status"] for a in attempts], [lb.COMPLETED, lb.FAILED])
        self.assertIn("invalid JSON", attempts[1]["error"])

        again = self.run_once()  # nothing new: A is done, B failed and --retry-failed is off
        self.assertEqual(sum(again.values()), 0)
        self.assertEqual(len(self.lines(self.attempts)), 2)

        self.run_once(retry_failed=True, max_note_attempts=2)  # second failure for B
        self.run_once(retry_failed=True, max_note_attempts=2)  # limit reached: no third attempt
        statuses = [a["status"] for a in self.lines(self.attempts)]
        self.assertEqual(statuses, [lb.COMPLETED, lb.FAILED, lb.FAILED])

        self.fail_b = False
        self.run_once(retry_failed=True, max_note_attempts=3)
        statuses = [a["status"] for a in self.lines(self.attempts)]
        self.assertEqual(statuses, [lb.COMPLETED, lb.FAILED, lb.FAILED, lb.COMPLETED])
        self.assertEqual(len(self.lines(self.evaluations)), 2)
        report = lb.build_report(MANIFEST, CATALOG, self.lines(self.attempts), self.lines(self.evaluations))
        row = next(r for r in report["rows"] if r["config_id"] == "gemini-baseline")
        self.assertEqual((row["completed"], row["failed_attempts_total"], row["duplicate_attempts_ignored"]), (2, 2, 0))

    def test_failed_judge_is_retried_without_regenerating(self) -> None:
        self.fail_b = False
        self.judge_fails = True
        self.run_once()
        self.assertEqual(len(self.lines(self.evaluations)), 2)
        self.judge_fails = False
        self.run_once()
        self.assertEqual(len(self.lines(self.attempts)), 2)  # no regeneration
        evaluations = self.lines(self.evaluations)
        self.assertEqual([e["status"] for e in evaluations], ["failed", "failed", "scored", "scored"])
        self.run_once()
        self.assertEqual(len(self.lines(self.evaluations)), 4)  # judged notes are not re-judged

    def test_records_carry_reproducibility_fields(self) -> None:
        self.fail_b = False
        self.run_once()
        attempt = self.lines(self.attempts)[0]
        for field in ("config_id", "config_type", "model_id", "provider", "manifest_id", "setup_fingerprint",
                      "status", "error", "started_at_utc"):
            self.assertIn(field, attempt)
        self.assertEqual(attempt["note"]["split"], "dev")
        self.assertEqual(attempt["setup"]["packages"], PACKAGES)
        self.assertIn("baseline_prompt_sha256", attempt["setup"])
        evaluation = self.lines(self.evaluations)[0]
        self.assertEqual(evaluation["evaluator"]["model_id"], "gemini-3.8-flash")
        self.assertEqual(evaluation["evaluator"]["judge_family"], "Gemini")
        self.assertIn("prompt_sha256", evaluation["evaluator"])


class PreflightTests(unittest.TestCase):
    def test_credentials_and_lookup_requirements(self) -> None:
        ids = [c["config_id"] for c in CATALOG["configurations"]]
        problems = runner.preflight(CATALOG, ids, {})
        self.assertIn("GOOGLE_API_KEY is not set", problems)
        self.assertIn("GROQ_API_KEY is not set", problems)
        self.assertTrue(any("UMLS_API_KEY" in p for p in problems))
        self.assertEqual(runner.preflight(CATALOG, ["gemini-baseline"], {"GOOGLE_API_KEY": "k"}), [])
        lookup = runner.build_lookup(CATALOG, {"UMLS_API_KEY": "k"})
        self.assertIsNotNone(lookup.umls)
        self.assertEqual(set(lookup.sources()), {"umls"})

    def test_pinned_versions_parse_requirements(self) -> None:
        pins = runner.pinned_versions()
        self.assertEqual(set(pins) & {"google-adk", "google-genai", "litellm"}, {"google-adk", "google-genai", "litellm"})


if __name__ == "__main__":
    unittest.main()
