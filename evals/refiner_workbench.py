"""Refiner workbench page (owned by Sameen).

Isolated from the Verifier workbench: this module only renders the Refiner tab
and scores results with ``evals.refiner_eval``. It does not touch Verifier
prompts, scoring, cache files, or tests. Streamlit is imported at module load
because this file is only imported by ``workbench.py`` (a Streamlit app); the
offline test suite never imports it.
"""
from __future__ import annotations

import asyncio
import os

import streamlit as st

# Small scratch example so the page is runnable out of the box. Scratch material
# only — never paste frozen test-set content here.
_SCRATCH_SOURCE = (
    "ASSESSMENT: Allergic rhinitis. PLAN: She will try Zyrtec instead of Allegra. "
    "Samples of Nasonex two sprays in each nostril given for three weeks."
)
_SCRATCH_DRAFT = (
    "You have hay fever. You should start taking Zyrtec instead of Allegra, and you "
    "also have asthma. Use Nasonex, two sprays in each nostril, for three weeks."
)
_SCRATCH_VERDICT = (
    '{\n  "faithful": false,\n'
    '  "unsupported_claims": ["you also have asthma"],\n'
    '  "omissions": [],\n'
    '  "reading_level_ok": true\n}'
)


def _fmt_rate(value: float | None) -> str:
    """Percentage string, or n/a when the metric did not apply to this example."""
    return "n/a" if value is None else f"{value * 100:.0f}%"


def refiner_page() -> None:
    """Render the Refiner workbench tab."""
    from agents.refiner import REFINER_DEFAULT_MODEL

    st.title("FaithfulMed Refiner Workbench")
    st.caption("Exploratory scratch/dev testing only — never paste frozen test-set material.")

    with st.sidebar:
        model = st.text_input("Model", REFINER_DEFAULT_MODEL)
        st.warning("Gemini is called only after Run is clicked.")

    source = st.text_area("Medical source", _SCRATCH_SOURCE, height=140)
    draft = st.text_area("Draft explanation (from the Simplifier)", _SCRATCH_DRAFT, height=120)
    verdict_text = st.text_area(
        "Verifier feedback (JSON: faithful / unsupported_claims / omissions)",
        _SCRATCH_VERDICT,
        height=160,
    )

    has_key = bool(os.getenv("GOOGLE_API_KEY"))
    if not has_key:
        st.info("Set GOOGLE_API_KEY in .env to enable live Refiner runs.")

    if st.button("Run Refiner", type="primary", disabled=not has_key):
        from agents.refiner import refine_and_describe

        with st.spinner("Refining…"):
            result = asyncio.run(refine_and_describe(source, draft, verdict_text, model))
        st.session_state["refiner_result"] = {
            "draft": draft,
            "verdict": verdict_text,
            "revised": result.revised_draft,
        }

    data = st.session_state.get("refiner_result")
    if not data:
        return

    from evals.refiner_eval import compare_texts, evaluate_single

    metrics = evaluate_single(data["draft"], data["revised"], data["verdict"])
    diff = compare_texts(data["draft"], data["revised"])

    st.subheader("Contract check")
    top = st.columns(3)
    top[0].metric("Contract pass", "Yes" if metrics["contract_pass"] else "No")
    top[1].metric("Unsupported removed", _fmt_rate(metrics["unsupported_removed_rate"]))
    top[2].metric("Omissions restored", _fmt_rate(metrics["omissions_restored_rate"]))
    bottom = st.columns(2)
    bottom[0].metric("Minimal-edit ratio", f"{metrics['minimal_edit_ratio']:.2f}",
                     help="1.0 = identical. High means the Refiner only touched what was flagged.")
    noop = metrics["noop_respected"]
    bottom[1].metric("No-op respected", "n/a" if noop is None else ("Yes" if noop else "No"),
                     help="Only applies when the Verifier said the draft was already faithful.")

    st.subheader("Revised explanation")
    st.write(data["revised"])

    with st.expander("What changed (sentence-level diff)"):
        st.markdown("**Removed from draft**")
        st.write(diff["removed_sentences"] or "— nothing removed —")
        st.markdown("**Added in revision**")
        st.write(diff["added_sentences"] or "— nothing added —")
        st.caption(f"Character delta: {diff['char_delta']} · Word delta: {diff['word_delta']}")

    with st.expander("Original draft and raw metrics"):
        st.markdown("**Draft sent to the Refiner**")
        st.write(data["draft"])
        st.json(metrics)
