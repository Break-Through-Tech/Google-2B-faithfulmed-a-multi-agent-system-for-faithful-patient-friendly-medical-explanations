"""Streamlit integration checks for the combined evaluation workspaces."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent.parent


class AgenticSystemDemoTests(unittest.TestCase):
    def test_readability_orders_are_separate_tabs_and_post_tab_reuses_all_prefixes(self) -> None:
        app = AppTest.from_file(str(ROOT / "agentic_system_demo.py"), default_timeout=30).run()

        self.assertIn("Full agentic system", app.segmented_control[0].options)
        self.assertIn("Readability pre-verifier", app.segmented_control[0].options)
        self.assertIn("Readability post-verifier", app.segmented_control[0].options)
        app.segmented_control[0].set_value("Readability post-verifier").run(timeout=30)

        self.assertFalse(app.exception)
        metrics = {item.label: item.value for item in app.metric[:4]}
        self.assertEqual(metrics["MTSamples"], "42")
        self.assertEqual(metrics["Reusable prefixes"], "42")
        self.assertEqual(metrics["Missing prefixes"], "0")
        self.assertGreaterEqual(int(metrics["Post-verifier outputs"]), 0)

    def test_pre_verifier_tab_opens_all_cached_mtsamples(self) -> None:
        app = AppTest.from_file(str(ROOT / "agentic_system_demo.py"), default_timeout=30).run()
        app.segmented_control[0].set_value("Readability pre-verifier").run()

        self.assertEqual([(item.label, item.value) for item in app.metric[:3]], [
            ("MTSamples", "42"),
            ("Cached outputs", "42"),
            ("Missing outputs", "0"),
        ])
        app.button[0].click().run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox[0].options), 42)
        self.assertIn("Outputs ≤ grade 8", [item.label for item in app.metric])


if __name__ == "__main__":
    unittest.main()
