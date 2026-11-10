"""Offline: the lookup smoke test passes its handoff checks in mock mode."""

from __future__ import annotations

import asyncio
import unittest

from evals.lookup_smoke import resolve_mode, run_smoke


class LookupSmokeTests(unittest.TestCase):
    def test_mock_mode_passes_handoff_steps(self) -> None:
        report = asyncio.run(run_smoke("mock", env={}))
        self.assertEqual(report["pipeline_status"], "completed")
        self.assertEqual([agent for agent, _ in report["stages"]],
                         ["Extractor", "TermLookup", "Simplifier", "Verifier", "Readability"])
        for step in ("1", "2", "3a", "3b"):
            with self.subTest(step=step):
                self.assertIs(report["steps"][step]["pass"], True, report["steps"][step])
        self.assertEqual(report["steps"]["4"]["pass"], "blocked (mock text)")

    def test_mode_resolution_never_goes_live_without_keys(self) -> None:
        self.assertEqual(resolve_mode("auto", {}), "mock")
        self.assertEqual(resolve_mode("auto", {"UMLS_API_KEY": "k"}), "umls-live")
        self.assertEqual(resolve_mode("auto", {"UMLS_API_KEY": "k", "GOOGLE_API_KEY": "g"}), "live")
        with self.assertRaises(RuntimeError):
            resolve_mode("umls-live", {})
        with self.assertRaises(RuntimeError):
            resolve_mode("live", {"UMLS_API_KEY": "k"})


if __name__ == "__main__":
    unittest.main()
