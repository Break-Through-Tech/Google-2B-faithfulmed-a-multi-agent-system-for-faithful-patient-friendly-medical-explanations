"""FaithfulMed workbench shell; Verifier is the only implemented agent view."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st

from agents.verifier import VERIFIER_DEFAULT_MODEL, run_medaesqa_verifier
from evals.verifier_workbench import (append_cache, cache_key, compare, load_cache, load_workbench_answers,
                                      parse_workbench_prediction, prompt_for, select_representative_answers, task_counts)

ROOT = Path(__file__).resolve().parent
DATA, CACHE = ROOT / "data" / "medaesqa_v1.json", ROOT / "runs" / "verifier_workbench_cache.jsonl"
TASKS = ("answer_accuracy", "sentence_relevance", "evidence_relation")


def load_env() -> None:
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key: os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@st.cache_data(show_spinner=False)
def answers() -> list:
    return load_workbench_answers(DATA)


def detail_rows(answer, prediction, task: str) -> list[dict]:
    comparison = compare(answer, prediction)[task]
    if prediction.data is None:
        return [{"id": answer.example_id, "status": "malformed", "actual item": "Response could not be parsed", "human label": None, "Gemini prediction": None, "Gemini reason": prediction.malformed_error}]
    pairs = {row["id"]: row for row in comparison}; rows = []
    if task == "answer_accuracy":
        prediction_data = prediction.data["answer_accuracy"]
        rows.append({**pairs[answer.example_id], "actual item": answer.candidate_answer, "Gemini reason": prediction_data["reason"]})
    else:
        output_sentences = {item["answer_sentence_id"]: item for item in prediction.data["sentences"]}
        for sentence in answer.sentences:
            output = output_sentences[sentence["answer_sentence_id"]]
            if task == "sentence_relevance":
                rows.append({**pairs[sentence["answer_sentence_id"]], "actual item": sentence["answer_sentence"], "Gemini reason": output["relevance_reason"]})
            else:
                outputs = {item["citation_id"]: item for item in output["citation_assessments"]}
                for citation in sentence["citations"]:
                    row = pairs[citation["citation_id"]]
                    model = outputs.get(citation["citation_id"], {})
                    rows.append({**row, "actual item": sentence["answer_sentence"], "cited PMID": citation["cited_pmid"],
                                 "evidence excerpt": citation["evidence_support"] or "No evidence excerpt available", "Gemini reason": model.get("reason", "")})
    return rows


def verifier_page() -> None:
    if not DATA.exists(): st.error("Missing data/medaesqa_v1.json"); return
    with st.sidebar:
        size = st.slider("Answer records", 1, 5, 5, help="Five is the hard maximum. One Gemini call is made per selected answer.")
        model = st.text_input("Model", VERIFIER_DEFAULT_MODEL)
        st.warning("Gemini calls happen only after Run is clicked. Cached identical requests use no new call.")
    selected, absent = select_representative_answers(answers(), size)
    st.subheader(f"Selected deterministic sample · {len(selected)}/5")
    st.dataframe([{"question ID": a.question_id, "method ID": a.method_id, "question": a.question} for a in selected], hide_index=True, use_container_width=True)
    st.caption("Selection greedily covers the most unseen human-label categories. " + (f"Not covered: {', '.join(sorted(absent))}." if absent else "All observed categories covered."))
    st.info("Sent to Gemini: question/method, machine answer, sentence IDs/text, and cited PMID plus evidence excerpts. Human labels, expert answer, and nuggets are used only after generation.")
    if st.button(f"Run Verifier on {len(selected)} answer(s)", type="primary", disabled=not os.getenv("GOOGLE_API_KEY")):
        cached, saved, results = load_cache(CACHE), [], []
        progress = st.progress(0)
        for index, answer in enumerate(selected, 1):
            key = cache_key(answer, model)
            raw = cached[key]["raw_output"] if key in cached else asyncio.run(run_medaesqa_verifier(prompt_for(answer.prompt_input()), model))
            if key not in cached: saved.append({"key": key, "raw_output": raw})
            results.append({"answer": answer, "prediction": parse_workbench_prediction(answer, raw)})
            progress.progress(index / len(selected))
        if saved: append_cache(CACHE, saved)
        st.session_state["verifier_results"] = results
    results = st.session_state.get("verifier_results", [])
    if not results: return
    comparisons = [compare(row["answer"], row["prediction"]) for row in results]
    st.subheader("Small-sample counts")
    columns = st.columns(3)
    for column, task in zip(columns, TASKS):
        counts = task_counts(comparisons, task)
        with column:
            st.metric(task.replace("_", " ").title(), sum(counts.values()), help="Number of individual items in this task, not a final benchmark result.")
            st.caption(f"Correct: {counts.get('correct', 0)} · Incorrect: {counts.get('incorrect', 0)} · Skipped: {counts.get('skipped', 0)}")
    task = st.selectbox("Inspect one evaluation task", TASKS, format_func=lambda value: value.replace("_", " ").title(), help="Choose one task to avoid mixing answer, relevance, and citation judgments.")
    status = st.selectbox("Show items", ["all", "correct", "incorrect", "skipped", "malformed"], help="Skipped means MedAESQA has no usable label or evidence excerpt for scoring that item.")
    st.subheader("Individual comparisons")
    for result in results:
        prediction = result["prediction"]
        rows = detail_rows(result["answer"], prediction, task)
        shown = rows if status == "all" else [row for row in rows if row["status"] == status]
        if not shown: continue
        answer = result["answer"]
        with st.container(border=True):
            st.markdown(f"### Question {answer.question_id} · {answer.method_id}")
            st.markdown(f"**Question:** {answer.question}")
            st.markdown("**Machine-generated candidate answer**")
            st.write(answer.candidate_answer)
            if answer.expert_reference:
                with st.expander("Expert-curated reference — display only; not sent to Gemini"):
                    st.write(answer.expert_reference)
            for row in shown:
                st.divider()
                st.markdown(f"**{row['status'].upper()} · {row['id']}**")
                if task == "answer_accuracy":
                    st.markdown("**Candidate answer**"); st.write(row["actual item"])
                elif task == "sentence_relevance":
                    st.markdown("**Answer sentence**"); st.write(row["actual item"])
                else:
                    st.markdown(f"**Sentence ({row['id'].split(':')[0]})**"); st.write(row["actual item"])
                    st.markdown(f"**Cited PMID:** {row['cited PMID']}")
                    st.markdown("**Evidence excerpt supplied to Gemini**"); st.write(row["evidence excerpt"])
                left, middle = st.columns(2)
                left.markdown(f"**Human label:** `{row.get('gold') or 'Unscored'}`")
                middle.markdown(f"**Gemini prediction:** `{row.get('predicted') or 'Unscored'}`")
                if row.get("Gemini reason"):
                    st.markdown("**Gemini’s brief justification:**"); st.write(row["Gemini reason"])
            with st.expander("Original MedAESQA record and Gemini structured response"):
                st.markdown("**Original MedAESQA data — human annotations, never sent to Gemini**"); st.json(answer.human_record())
                st.markdown("**Gemini parsed structured response**"); st.json(prediction.data)
            with st.expander("Raw Gemini response — debugging only"):
                st.code(prediction.raw_output, language="json")


def placeholder_page(agent: str) -> None:
    st.title(f"{agent} workbench")
    st.info(f"{agent} does not have a workbench yet. This page is intentionally isolated from the Verifier. Its owner can implement an agent-specific view here without changing Verifier prompts, scoring, cache files, or tests.")
    # Teammate scaffold: keep each agent's UI and evaluation helpers in a separate module.


def main() -> None:
    load_env(); st.set_page_config(page_title="FaithfulMed Workbench", page_icon="🩺", layout="wide")
    agent = st.sidebar.radio("Agent workbench", ["Verifier", "Extractor", "Simplifier", "Refiner", "Readability"])
    if agent == "Verifier":
        st.title("FaithfulMed Verifier Workbench"); st.caption("Exploratory MedAESQA results only—not final research results.")
        verifier_page()
    elif agent == "Refiner":
        from evals.refiner_workbench import refiner_page
        refiner_page()
    else: placeholder_page(agent)


if __name__ == "__main__": main()
