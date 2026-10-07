"""Streamlit workbench for testing the FaithfulMed Readability agent."""

from __future__ import annotations

import asyncio

import streamlit as st

from agents.readability import READABILITY_DEFAULT_MODEL, run_readability
from evals.readability_eval import evaluate_single, compare_scores


def readability_page() -> None:
    """Display the interactive Readability agent workbench."""
    st.title("FaithfulMed Readability Workbench")
    st.caption(
        "Interactive testing of the Readability agent. "
        "These results are exploratory and are not final research results."
    )

    with st.sidebar:
        model = st.text_input("Model", READABILITY_DEFAULT_MODEL)
        st.warning("Gemini is called only after Run is clicked.")

    draft = st.text_area(
        "Medical draft",
        height=250,
        placeholder="Enter a medical draft to evaluate its readability and simplify it...",
    )

    if st.button(
        "Run Readability Agent",
        type="primary",
        disabled=not draft.strip(),
    ):
        with st.spinner("Running Readability agent..."):
            revised = asyncio.run(run_readability(draft, model))

        st.session_state["readability_original"] = draft
        st.session_state["readability_revised"] = revised

    original = st.session_state.get("readability_original")
    revised = st.session_state.get("readability_revised")

    if original is None or revised is None:
        return

    # Calculate the readability results for both versions.
    before = evaluate_single(original)
    after = evaluate_single(revised)

    # Calculate the changes from the original to the revised version.
    comparison = compare_scores(original, revised)

    st.subheader("Result")

    left, right = st.columns(2)

    with left:
        st.markdown("### Original draft")
        st.write(original)

    with right:
        st.markdown("### Revised draft")
        st.write(revised)

    st.subheader("Readability metrics")

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "Flesch-Kincaid",
            f"{after['flesch_kincaid_grade']:.1f}",
            delta=f"{comparison['fk_diff']:.1f}",
            delta_color="inverse",
        )

    with columns[1]:
        st.metric(
            "SMOG",
            f"{after['smog_index']:.1f}",
            delta=f"{comparison['smog_index_diff']:.1f}",
            delta_color="inverse",
        )

    with columns[2]:
        st.metric(
            "Jargon density",
            f"{after['jargon_density']:.3f}",
            delta=f"{comparison['jargon_density_diff']:.3f}",
            delta_color="inverse",
        )

    with columns[3]:
        st.metric(
            "Word count",
            after["word_count"],
            delta=f"{comparison['word_count_diff']}",
            delta_color="inverse",
        )

    st.subheader("Readability target")

    target_left, target_right = st.columns(2)

    with target_left:
        st.markdown("**Original:**")
        if before["meets_target"]:
            st.success("Meets grade 8 or below")
        else:
            st.warning("Above grade 8")

    with target_right:
        st.markdown("**Revised:**")
        if after["meets_target"]:
            st.success("Meets grade 8 or below")
        else:
            st.warning("Above grade 8")

    st.subheader("Refusal detection")

    refusal_left, refusal_right = st.columns(2)

    with refusal_left:
        st.markdown(
            f"**Original:** {'Yes' if before['is_refusal'] else 'No'}"
        )

    with refusal_right:
        st.markdown(
            f"**Revised:** {'Yes' if after['is_refusal'] else 'No'}"
        )

    with st.expander("Detailed evaluation results"):
        st.markdown("**Original scores**")
        st.json(before)

        st.markdown("**Revised scores**")
        st.json(after)

        st.markdown("**Changes**")
        st.json(comparison)