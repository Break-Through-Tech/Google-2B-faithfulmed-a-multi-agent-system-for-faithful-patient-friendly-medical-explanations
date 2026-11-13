"""Offline tests for the leaderboard schema, aggregation, report, and blind review."""

from __future__ import annotations

import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents import usage_from_metadata
from evals import leaderboard as lb

CATALOG = lb.load_catalog()
PACKAGES = {"google-adk": "2.14.0", "google-genai": "2.34.0", "litellm": "1.106.2"}
EVALUATOR = lb.evaluator_setup(CATALOG)


def note(example_id: str, dataset: str = "MTSamples") -> dict:
    return {"dataset": dataset, "split": "dev", "example_id": example_id, "source_file": f"{example_id}.json",
            "source_sha256": lb.sha256_text(example_id)}


MANIFEST = {"schema": lb.MANIFEST_SCHEMA, "manifest_id": "unit", "purpose": "pilot", "split": "dev",
            "datasets": ["MTSamples"], "notes": [note("A"), note("B"), note("C")]}
MANIFEST["manifest_sha256"] = lb.stable_hash(MANIFEST["notes"])


def call(stage="Agent", inp=1000, out=2000, think=0, status="ok", usage=True, retried=False, wait=0.0):
    return {"stage": stage, "attempt": 1, "status": status, "usage_available": usage,
            "input_tokens": inp if usage else 0, "output_tokens": out if usage else 0,
            "thinking_tokens": think if usage else 0, "retried": retried, "wait_after_s": wait, "elapsed_s": 1.0}


READABLE = {"flesch_kincaid_grade": 7.0, "smog_index": 9.0, "jargon_density": 0.2, "word_count": 150,
            "meets_target": True, "is_refusal": False}


def make_attempt(config_id: str, example_id: str, status: str = lb.COMPLETED, *, calls=None, latency=10.0,
                 started="2026-11-10T10:00:00+00:00", readability=READABLE, catalog=CATALOG, packages=PACKAGES,
                 source_sha=None) -> dict:
    config = lb.configuration(catalog, config_id)
    setup = lb.generation_setup(config, catalog, packages)
    n = next(x for x in MANIFEST["notes"] if x["example_id"] == example_id)
    if source_sha:
        n = {**n, "source_sha256": source_sha}
    calls = calls if calls is not None else [call()]
    return lb.new_attempt(config, setup, MANIFEST, n, status=status,
                          error="boom" if status == lb.FAILED else None,
                          final_output="Explanation." if status == lb.COMPLETED else None,
                          generation=lb.summarize_calls(calls, latency_s=latency if status != lb.FAILED else 3.0),
                          readability_metrics=readability if status == lb.COMPLETED else None,
                          started_at_utc=started)


def make_eval(attempt: dict, *, supported=9, total=10, hallucination=False, status="scored", fingerprint=None,
              tokens=(500, 100), created="2026-11-10T11:00:00+00:00") -> dict:
    evaluator = dict(EVALUATOR)
    if fingerprint:
        evaluator["evaluator_fingerprint"] = fingerprint
    metrics = None
    if status == "scored":
        metrics = {"claims_returned": total, "claims_assessed": total, "claims_skipped": 0,
                   "claim_validation_coverage": 100.0, "supported_claims": supported, "uncertain_claims": 0,
                   "factual_fidelity_estimate": supported / total * 100,
                   "clinically_meaningful_hallucination": hallucination}
    record = lb.new_evaluation(attempt, evaluator, status=status, raw_output="{}", parsed_output=None,
                               factual_metrics=metrics,
                               usage=lb.summarize_calls([call("FactualEvaluator", *tokens)], latency_s=2.0),
                               error=None if status == "scored" else "bad")
    record["created_at_utc"] = created
    return record


def row(report: dict, config_id: str) -> dict:
    return next(r for r in report["rows"] if r["config_id"] == config_id)


class UsageNormalizationTests(unittest.TestCase):
    def test_gemini_thinking_counted_once(self) -> None:
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=50, thoughts_token_count=30,
                                total_token_count=180)
        result = usage_from_metadata(usage)
        self.assertEqual((result["input_tokens"], result["output_tokens"], result["thinking_tokens"]), (100, 80, 30))

    def test_litellm_reasoning_not_double_counted(self) -> None:
        # LiteLLM: completion (candidates) already includes the 30 reasoning tokens.
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=80, thoughts_token_count=30,
                                total_token_count=180)
        self.assertEqual(usage_from_metadata(usage)["output_tokens"], 80)

    def test_no_total_and_no_usage(self) -> None:
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2,
                                total_token_count=None)
        self.assertEqual(usage_from_metadata(usage)["output_tokens"], 7)
        self.assertIsNone(usage_from_metadata(None))
        self.assertIsNone(usage_from_metadata(SimpleNamespace()))


class CalculationTests(unittest.TestCase):
    def test_percentile_and_call_summary(self) -> None:
        self.assertEqual(lb.percentile([5, 1, 3, 2, 4], 0.9), 5)
        self.assertEqual(lb.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9), 9)
        self.assertIsNone(lb.percentile([], 0.5))
        calls = [call(status="error", usage=False, retried=True, wait=2.5), call(inp=100, out=50, think=20)]
        summary = lb.summarize_calls(calls, latency_s=12.0)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["retry_wait_s"], 2.5)
        self.assertEqual(summary["tokens"], {"input": 100, "output": 50, "thinking": 20})
        self.assertTrue(summary["usage_complete"])  # failed calls without usage do not block coverage
        self.assertFalse(lb.summarize_calls([call(usage=False)], latency_s=1.0)["usage_complete"])

    def test_price_periods_zero_and_unknown(self) -> None:
        gemini, gemma, oss = (CATALOG["models"][k] for k in ("gemini", "gemma", "gpt_oss"))
        self.assertEqual(lb.price_for(gemini, "2026-11-10"), ({"input": 0.75, "output": 3.75}, "ok"))
        self.assertEqual(lb.price_for(gemini, "2027-01-01"), ({"input": 1.5, "output": 7.5}, "ok"))
        self.assertEqual(lb.price_for(gemma, "2026-11-10"), (None, "no_paid_price"))
        self.assertEqual(lb.price_for({"pricing": {"periods": []}}, "2026-11-10"), (None, "price_unknown"))
        self.assertAlmostEqual(lb.estimate_cost({"input": 1000, "output": 2000}, {"input": 0.15, "output": 0.60}),
                               0.00015 + 0.0012)
        self.assertIsNone(lb.estimate_cost({"input": 1, "output": 1}, None))


class AggregationTests(unittest.TestCase):
    def build(self, attempts, evaluations=()):
        return lb.build_report(MANIFEST, CATALOG, list(attempts), list(evaluations))

    def test_metrics_latency_tokens_and_cost(self) -> None:
        a = make_attempt("gemini-system", "A", latency=10.0, calls=[call(inp=1000, out=2000, think=500)])
        b = make_attempt("gemini-system", "B", latency=30.0,
                         calls=[call(status="error", usage=False, retried=True, wait=4.0), call(inp=3000, out=4000)])
        c = make_attempt("gemini-system", "C", latency=20.0)
        evals = [make_eval(a, supported=9, total=10), make_eval(b, supported=10, total=10, hallucination=False),
                 make_eval(c, supported=4, total=10, hallucination=True)]
        r = row(self.build([a, b, c], evals), "gemini-system")
        self.assertEqual((r["completed"], r["missing"], r["failed"]), (3, 0, 0))
        self.assertAlmostEqual(r["fidelity_pct"], 23 / 30 * 100)
        self.assertAlmostEqual(r["hallucination_rate_pct"], 100 / 3)
        self.assertEqual(r["scoring_coverage_pct"], 100.0)
        self.assertEqual((r["latency_median_s"], r["latency_p90_s"]), (20.0, 30.0))
        self.assertEqual((r["retries_total"], r["retry_wait_total_s"]), (1, 4.0))
        self.assertEqual(r["gen_input_tokens_mean"], (1000 + 3000 + 1000) / 3)
        self.assertEqual(r["gen_thinking_tokens_mean"], 500 / 3)
        expected = [(1000 * 0.75 + 2000 * 3.75) / 1e6, (3000 * 0.75 + 4000 * 3.75) / 1e6,
                    (1000 * 0.75 + 2000 * 3.75) / 1e6]
        self.assertAlmostEqual(r["est_cost_total_usd"], sum(expected))
        self.assertEqual(r["cost_status"], "estimated")
        self.assertEqual(r["billed"], "$0 (free tier)")
        # The judge's tokens are separate from generation tokens.
        self.assertEqual((r["judge_input_tokens"], r["judge_output_tokens"], r["judge_model_calls"]), (1500, 300, 3))
        self.assertAlmostEqual(r["judge_est_cost_usd"], (1500 * 0.75 + 300 * 3.75) / 1e6)

    def test_zero_price_model_is_not_unknown_cost(self) -> None:
        r = row(self.build([make_attempt("gemma-baseline", "A")]), "gemma-baseline")
        self.assertIsNone(r["est_cost_per_note_usd"])
        self.assertEqual(r["cost_status"], "no_paid_price")
        self.assertEqual(r["billed"], "$0 (free tier)")
        self.assertIn("free_tier", json.dumps(CATALOG["models"]["gemma"]["pricing"]))

    def test_incomplete_usage_makes_cost_unavailable(self) -> None:
        a = make_attempt("gpt-oss-system", "A", calls=[call(), call(usage=False)])
        b = make_attempt("gpt-oss-system", "B")
        r = row(self.build([a, b]), "gpt-oss-system")
        self.assertEqual(r["usage_coverage"], "1/2")
        self.assertTrue(r["cost_status"].startswith("partial"))
        self.assertAlmostEqual(r["est_cost_total_usd"], (1000 * 0.15 + 2000 * 0.60) / 1e6)

    def test_duplicates_use_first_terminal_attempt(self) -> None:
        first = make_attempt("gemini-baseline", "A", latency=5.0, started="2026-11-10T09:00:00+00:00")
        rerun = make_attempt("gemini-baseline", "A", latency=99.0, started="2026-11-10T12:00:00+00:00")
        report = self.build([rerun, first])
        r = row(report, "gemini-baseline")
        self.assertEqual(r["duplicate_attempts_ignored"], 1)
        self.assertEqual(r["latency_median_s"], 5.0)
        chosen = next(n for n in report["notes"] if n["config_id"] == "gemini-baseline" and n["example_id"] == "A")
        self.assertEqual(chosen["attempt_id"], first["attempt_id"])

    def test_failures_and_missing_stay_visible(self) -> None:
        failed_only = make_attempt("gemma-system", "A", lb.FAILED, calls=[call(inp=400, out=0)])
        failed_then_ok = [make_attempt("gemma-system", "B", lb.FAILED, started="2026-11-10T09:00:00+00:00"),
                          make_attempt("gemma-system", "B", started="2026-11-10T10:00:00+00:00")]
        report = self.build([failed_only, *failed_then_ok])
        r = row(report, "gemma-system")
        self.assertEqual((r["completed"], r["failed"], r["missing"]), (1, 1, 1))
        self.assertEqual(r["failed_attempts_total"], 2)
        self.assertEqual(r["failed_attempt_input_tokens"], 1400)
        states = {n["example_id"]: n["state"] for n in report["notes"] if n["config_id"] == "gemma-system"}
        self.assertEqual(states, {"A": "failed", "B": "completed", "C": "missing"})
        self.assertTrue(any("gemma-system" in w and "1 missing and 1 failed" in w for w in report["warnings"]))
        markdown = lb.render_markdown(report)
        self.assertIn("MTSamples/A | failed", markdown)
        self.assertIn("MTSamples/C | missing", markdown)

    def test_configuration_without_runs_is_all_missing_and_unavailable(self) -> None:
        report = self.build([make_attempt("gemini-baseline", "A")])
        r = row(report, "gpt-oss-baseline")
        self.assertEqual((r["missing"], r["setup"]), (3, "none"))
        for field in ("fidelity_pct", "latency_median_s", "est_cost_per_note_usd", "fk_grade_mean"):
            self.assertIsNone(r[field])
        self.assertEqual(len(report["rows"]), 6)
        self.assertIn("| gpt-oss-baseline | unavailable | unavailable |", lb.render_markdown(report))
        csv_rows = list(csv.DictReader(io.StringIO(lb.to_csv(report["rows"], lb.ROW_FIELDS))))
        self.assertEqual(next(x for x in csv_rows if x["config_id"] == "gpt-oss-baseline")["fidelity_pct"], "")

    def test_not_accepted_counted_not_judged(self) -> None:
        a = make_attempt("gemini-system", "A", lb.NOT_ACCEPTED, latency=40.0)
        r = row(self.build([a], [make_eval(a)]), "gemini-system")
        self.assertEqual((r["not_accepted"], r["completed"], r["judge_scored"]), (1, 0, 0))
        self.assertEqual(r["latency_median_s"], 40.0)
        self.assertIsNone(r["fidelity_pct"])

    def test_missing_measurements_and_judge_selection(self) -> None:
        a = make_attempt("gemini-system", "A", readability=None)
        b = make_attempt("gemini-system", "B")
        evals = [make_eval(a, status="failed"),
                 make_eval(b, status="failed", created="2026-11-10T11:00:00+00:00"),
                 make_eval(b, supported=5, total=10, created="2026-11-10T12:00:00+00:00"),
                 make_eval(b, supported=1, total=10, fingerprint="other-judge")]
        report = self.build([a, b], evals)
        r = row(report, "gemini-system")
        self.assertEqual((r["judge_scored"], r["judge_failed"]), (1, 1))
        self.assertEqual(r["scoring_coverage_pct"], 50.0)
        self.assertEqual(r["fidelity_pct"], 50.0)  # other judge ignored
        self.assertEqual(r["readability_scored"], 1)
        self.assertTrue(any("different judge" in w for w in report["warnings"]))

    def test_setups_are_grouped_and_mismatches_flagged(self) -> None:
        old = make_attempt("gemini-system", "A", packages={**PACKAGES, "google-adk": "2.13.0"})
        new = make_attempt("gemini-system", "B")
        other_model = make_attempt("gemma-system", "A")
        report = self.build([old, new, other_model])
        rows = [r for r in report["rows"] if r["config_id"] == "gemini-system"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(any("2 different setups" in w for w in report["warnings"]))
        self.assertTrue(any("full_system rows were generated with different" in w for w in report["warnings"]))

    def test_changed_source_text_is_ignored(self) -> None:
        stale = make_attempt("gemini-baseline", "A", source_sha="0" * 64)
        r = row(self.build([stale]), "gemini-baseline")
        self.assertEqual((r["stale_attempts_ignored"], r["missing"]), (1, 3))

    def test_setup_fingerprint_tracks_model_and_shared_parts(self) -> None:
        gem = lb.generation_setup(lb.configuration(CATALOG, "gemini-system"), CATALOG, PACKAGES)
        gma = lb.generation_setup(lb.configuration(CATALOG, "gemma-system"), CATALOG, PACKAGES)
        self.assertEqual(gem["shared_setup_sha256"], gma["shared_setup_sha256"])
        self.assertNotEqual(gem["setup_fingerprint"], gma["setup_fingerprint"])
        self.assertEqual(gem["lookup"], {"umls": True})
        self.assertNotIn("factual_evaluator", gem["prompt_hashes"])
        base = lb.generation_setup(lb.configuration(CATALOG, "gemini-baseline"), CATALOG, PACKAGES)
        self.assertIn("baseline_prompt_sha256", base)

    def test_report_names_the_gemini_judge(self) -> None:
        text = lb.render_markdown(self.build([]))
        self.assertIn("Automated faithfulness judge:** gemini-3.8-flash (Gemini)", text)
        self.assertIn("Gemini is also a contestant", text)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_tamper_evident_and_test_split_is_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            edited = copy.deepcopy(MANIFEST)
            edited["notes"].pop()
            path.write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaises(ValueError):
                lb.load_manifest(path)
        with self.assertRaises(ValueError):
            lb.build_manifest("x", "test", ["MTSamples"], "pilot")
        with self.assertRaises(ValueError):
            lb.build_manifest("x", "dev", ["MTSamples"], "final_test")


class BlindReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attempts = [make_attempt(c["config_id"], n["example_id"])
                         for c in CATALOG["configurations"] for n in MANIFEST["notes"]]
        self.evaluations = [make_eval(a, hallucination=(a["note"]["example_id"] == "A")) for a in self.attempts]
        self.report = lb.build_report(MANIFEST, CATALOG, self.attempts, self.evaluations)
        fake = [SimpleNamespace(dataset="MTSamples", example_id=n["example_id"], source_text=f"Source {n['example_id']}")
                for n in MANIFEST["notes"]]
        self.patcher = patch.object(lb, "manifest_examples", return_value=fake)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()

    def draw(self, seed=7, per_config=2):
        return lb.draw_review_sample(self.report, self.attempts, MANIFEST, per_config=per_config, seed=seed,
                                     sample_id="s")

    def test_equal_sized_reproducible_and_blind(self) -> None:
        items, key = self.draw()
        self.assertEqual(len(items["items"]), 12)
        self.assertEqual(sorted(Counter_(v["config_id"] for v in key["items"].values()).values()), [2] * 6)
        self.assertEqual(self.draw()[1]["items"], key["items"])  # same seed, same sample
        self.assertNotEqual(self.draw(seed=8)[1]["items"], key["items"])
        self.assertEqual(key["seed"], 7)
        blob = json.dumps(items["items"]).lower()
        self.assertEqual(set(items["items"][0]), {"review_id", "dataset", "example_id", "source_text", "explanation"})
        for word in ("gemini", "gemma", "gpt-oss", "baseline", "system", "config", "attempt", "setup"):
            self.assertNotIn(word, blob, word)
        self.assertIn("review_id,reviewer,clinically_meaningful_hallucination,notes", lb.judgments_template(items))

    def test_shortfall_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            self.draw(per_config=4)

    def test_human_and_judge_reported_separately(self) -> None:
        items, key = self.draw(per_config=3)
        judgments = []
        for review_id, item in key["items"].items():
            label = "yes" if item["example_id"] in ("A", "B") else "no"  # human also flags B
            judgments.append({"review_id": review_id, "reviewer": "r1", "clinically_meaningful_hallucination": label})
        judgments.append({"review_id": "R-001", "reviewer": "r2", "clinically_meaningful_hallucination": "unsure"})
        report = lb.build_report(MANIFEST, CATALOG, self.attempts, self.evaluations,
                                 review_keys=[key], judgments={"s": judgments})
        summary = report["human_review"][0]
        r = next(x for x in summary["rows"] if x["config_id"] == "gemini-baseline")
        self.assertEqual((r["sampled"], r["human_judgments"], r["paired"]), (3, 3, 3))
        self.assertAlmostEqual(r["human_hallucination_rate_pct"], 200 / 3)
        self.assertAlmostEqual(r["judge_hallucination_rate_pct"], 100 / 3)
        self.assertAlmostEqual(r["agreement_pct"], 200 / 3)
        self.assertAlmostEqual(r["cohen_kappa"], 0.4)
        self.assertEqual(sum(x["human_unsure"] for x in summary["rows"]), 1)
        # Automated scores in the main table are unchanged by human judgments.
        self.assertEqual(row(report, "gemini-baseline")["hallucination_rate_pct"],
                         row(self.report, "gemini-baseline")["hallucination_rate_pct"])
        self.assertIn("Blind human review: s", lb.render_markdown(report))


def Counter_(values):
    from collections import Counter
    return Counter(values)


class HistoricalTests(unittest.TestCase):
    def test_historical_rows_are_labelled_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            system = Path(tmp) / "s.jsonl"
            baseline = Path(tmp) / "b.jsonl"
            base = {"pipeline_model": "gemini-3.8-flash", "pipeline_version": "system-pipeline-v2",
                    "prompt_hashes": {"x": "1"}, "evaluator_model": "gemini-3.8-flash", "split": "dev",
                    "dataset": "MTSamples", "final_explanation": "Text.", "readability_metrics": READABLE,
                    "factual_metrics": None,
                    "pipeline_trace": {"started_at_utc": "2026-10-01T00:00:00+00:00",
                                       "completed_at_utc": "2026-10-01T00:01:40+00:00"}}
            rows = [{**base, "example_id": "A", "created_at_utc": "1"}, {**base, "example_id": "A", "created_at_utc": "2"},
                    {**base, "example_id": "T", "split": "test", "created_at_utc": "3"}]
            system.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            baseline.write_text(json.dumps({"model": "gemini-3.8-flash", "split": "dev", "status": "success",
                                            "metrics": READABLE}), encoding="utf-8")
            hist = lb.historical_rows(("scratch", "dev"), system, baseline)
        self.assertEqual(len(hist), 2)
        self.assertIn("system-pipeline-v2", hist[0]["label"])
        self.assertEqual((hist[0]["notes"], hist[0]["duplicates_ignored"]), (1, 1))
        self.assertEqual(hist[0]["latency_median_s"], 100.0)
        self.assertIsNone(hist[1]["fidelity_pct"])
        self.assertIn("not recorded", hist[1]["original_settings"])


if __name__ == "__main__":
    unittest.main()
