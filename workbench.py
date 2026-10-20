"""FaithfulMed workbench shell; Verifier is the only implemented agent view."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from agents.verifier import VERIFIER_DEFAULT_MODEL, run_medaesqa_verifier
from evals.verifier_workbench import (append_cache, cache_key, compare, load_cache, load_workbench_answers,
                                      export_review_summary, load_all_evaluation_splits, parse_workbench_prediction, prompt_for,
                                      append_run_log, MAX_BATCH_SIZE, mismatch_diagnostic, PROMPT_VERSION_LABELS,
                                      REVISED_PROMPT_VERSION, random_answer_per_question, resolve_answer, task_agreement_statistics,
                                      KAPPA_TARGETS)
from evals.independent_review import load_independent_review, supplementary_agreement

ROOT = Path(__file__).resolve().parent
DATA, CACHE = ROOT / "data" / "medaesqa_v1.json", ROOT / "runs" / "verifier_workbench_cache.jsonl"
EVALUATION_SPLITS = ROOT / "evals" / "verifier_evaluation_splits_v2_balanced.json"
RUN_LOG = ROOT / "runs" / "verifier_workbench_runs.jsonl"
INDEPENDENT_REVIEW = ROOT / "evals" / "verifier_independent_review_v2.json"
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
        return [{"id": answer.example_id, "status": "malformed", "actual item": "Response could not be parsed", "human label": None, "Gemini prediction": None, "Gemini reason": prediction.malformed_error, "diagnostic": mismatch_diagnostic(task, "malformed", malformed=True)}]
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
    for row in rows:
        row["diagnostic"] = mismatch_diagnostic(task, row["status"])
    return rows


def cached_result_rows(candidates: list, model: str, prompt_version: str) -> list[dict]:
    """Rehydrate only matching cached responses; this function never calls Gemini."""
    cached = load_cache(CACHE)
    rows = []
    for answer in candidates:
        item = cached.get(cache_key(answer, model, prompt_version))
        if item:
            rows.append({"answer": answer, "prediction": parse_workbench_prediction(answer, item["raw_output"])})
    return rows


def supplementary_benchmark_eligibility(results: list[dict], task: str) -> dict[tuple[str, str], str | None]:
    """Use benchmark label availability only to exclude unscored supplemental items."""
    eligible: dict[tuple[str, str], str | None] = {}
    allowed = {"answer_accuracy": {"yes", "no"},
               "sentence_relevance": {"required", "borderline", "unnecessary", "inappropriate"},
               "evidence_relation": {"supporting", "contradicting", "neutral", "not relevant", "invalid citation"}}[task]
    for result in results:
        answer = result["answer"]
        if task == "answer_accuracy":
            eligible[(answer.example_id, answer.example_id)] = None if answer.answer_accuracy_gold in allowed else "Original human answer-accuracy label is missing or unscored."
            continue
        for sentence in answer.sentences:
            if task == "sentence_relevance":
                eligible[(answer.example_id, sentence["answer_sentence_id"])] = None if sentence["relevance_gold"] in allowed else "Original human sentence-relevance label is missing or unscored."
            else:
                for citation in sentence["citations"]:
                    reason = None
                    if citation["evidence_support"] is None:
                        reason = "Evidence excerpt is missing; the benchmark citation item is skipped."
                    elif citation["evidence_relation_gold"] not in allowed:
                        reason = "Original human evidence-relation label is missing or unscored."
                    eligible[(answer.example_id, citation["citation_id"])] = reason
    return eligible


def supplementary_review_view(results: list[dict], split_name: str) -> None:
    """Display saved-only Gemini/Codex comparison without touching benchmark metrics."""
    st.subheader("Supplementary Rubric Agreement")
    st.caption("Provisional Codex-generated labels. This view is independent of, and does not modify, MedAESQA benchmark labels, original scores, or Cohen’s κ. Here, ‘gold’ columns refer to the saved Codex primary label, not the MedAESQA human label. Items whose original benchmark label is missing, skipped, or unscored are excluded and display no supplementary label.")
    if not INDEPENDENT_REVIEW.exists():
        st.info("No saved independent review is available for this split.")
        return
    try:
        review = load_independent_review(INDEPENDENT_REVIEW)
        review_records = review["splits"].get(split_name, [])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f"Independent review is unavailable: {exc}")
        return
    if not review_records:
        st.info(f"No independent review records are available for {split_name}.")
        return
    prediction_by_example = {
        row["answer"].example_id: row["prediction"].data
        for row in results if row["prediction"].data is not None
    }
    malformed_responses = sum(row["prediction"].data is None for row in results)
    if not prediction_by_example:
        st.info("No saved, parseable Gemini result is loaded for this split. Run results must be available locally; this view never calls Gemini.")
        return
    summaries = {
        task: supplementary_agreement(
            review_records, prediction_by_example, task,
            supplementary_benchmark_eligibility(results, task),
        )
        for task in TASKS
    }
    rows = []
    for task, metrics in summaries.items():
        rows.append({
            "Metric": task.replace("_", " ").title(), "Split": split_name,
            "Scored": metrics["scored"],
            "Correct": metrics["exact_matches"], "Incorrect": metrics["incorrect"],
            "Exact agreement": "not calculable" if metrics["exact_agreement"] is None else f"{metrics['exact_agreement']:.1%}",
            "Error rate": "not calculable" if metrics["error_rate"] is None else f"{metrics['error_rate']:.1%}",
            "Cohen’s κ": "not calculable" if metrics["kappa"] is None else f"{metrics['kappa']:.3f}",
            "Skipped pairs": metrics["skipped_unscorable"], "Unscored total": metrics["unscored_total"],
            "Missing gold": metrics["missing_gold"], "Missing prediction": metrics["missing_prediction"],
            "Invalid gold": metrics["invalid_gold"], "Invalid prediction": metrics["invalid_prediction"],
            "Malformed responses": malformed_responses,
            "Target": KAPPA_TARGETS.get(task, "No documented agreement/κ target for this metric."),
            "Accepted-label matches": f"{metrics['accepted_matches']} ({metrics['accepted_match_rate']:.1%})" if metrics["accepted_match_rate"] is not None else "not calculable",
            "Ambiguity flagged": metrics["ambiguity_flagged"], "Needs human review": metrics["needs_human_review"],
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)
    task = st.selectbox("Inspect supplementary items", TASKS, format_func=lambda value: value.replace("_", " ").title(), key="supplementary_task")
    for row in summaries[task]["details"]:
        with st.expander(f"{row['example_id']} · {row['item_id']} · {row['status']}"):
            st.write({key: row.get(key) for key in ("gemini_prediction", "primary_label", "acceptable_labels", "rationale", "alternative_rationales", "ambiguity_flag", "needs_human_review", "status", "skip_reason")})


def verifier_page() -> None:
    if not DATA.exists() or not EVALUATION_SPLITS.exists():
        st.error("Missing data/medaesqa_v1.json or evals/verifier_evaluation_splits_v2_balanced.json"); return
    with st.sidebar:
        model = st.text_input("Model", VERIFIER_DEFAULT_MODEL)
        prompt_version = REVISED_PROMPT_VERSION
        st.caption(f"Prompt: {PROMPT_VERSION_LABELS[prompt_version]}")
        st.warning("Gemini calls happen only after Run is clicked. Cached identical requests use no new call.")
    pool = st.radio(
        "Evaluation set",
        ["prompt-development", "validation", "final-test", "Explore random answers"],
        format_func=lambda value: {
            "prompt-development": "Prompt-development (inspect and revise)",
            "validation": "Validation (compare candidate prompts)",
            "final-test": "Final-test (locked configuration only)",
            "Explore random answers": "Explore random answers",
        }[value],
        help="The three saved sets are disjoint and reproducible. The final-test set must remain unseen until the prompt, model, and settings are locked.",
    )
    all_answers = answers()
    available_question_count = len({answer.question_id for answer in all_answers})
    needed_question_count = MAX_BATCH_SIZE * 3
    if available_question_count < needed_question_count:
        st.error(f"The local MedAESQA data has only {available_question_count} distinct question IDs; {needed_question_count} are required for the three ten-record evaluation sets.")
        return
    st.subheader("Add a specific record")
    st.caption("Type an existing MedAESQA Question ID and method ID (for example, `127` and `M6`). Added records are automatically included in the next run.")
    question_column, method_column, add_column, clear_column = st.columns([2, 2, 1.4, 1.6])
    typed_question_id = question_column.text_input("Question ID", key="manual_question_id")
    typed_method_id = method_column.text_input("Method ID", key="manual_method_id", placeholder="M1–M30")
    if "verifier_manual_example_ids" not in st.session_state:
        st.session_state["verifier_manual_example_ids"] = []
    if add_column.button("Add and select", disabled=not typed_question_id.strip() or not typed_method_id.strip()):
        try:
            manual_answer = resolve_answer(all_answers, typed_question_id, typed_method_id)
            if manual_answer.example_id not in st.session_state["verifier_manual_example_ids"]:
                st.session_state["verifier_manual_example_ids"].append(manual_answer.example_id)
        except ValueError as exc:
            st.error(str(exc))
    if clear_column.button("Clear added records", disabled=not st.session_state["verifier_manual_example_ids"]):
        st.session_state["verifier_manual_example_ids"] = []
    if "verifier_answer_seed" not in st.session_state:
        st.session_state["verifier_answer_seed"] = secrets.randbelow(2**31)
    if pool == "Explore random answers" and st.button("Choose different random machine answers", help="Keeps one row per Question ID but randomly samples machine-answer methods again."):
        st.session_state["verifier_answer_seed"] = secrets.randbelow(2**31)
    try:
        saved_splits = load_all_evaluation_splits(EVALUATION_SPLITS, all_answers)
        split_metadata = saved_splits[pool][1] if pool in saved_splits else None
        candidates = (saved_splits[pool][0] if pool in saved_splits
                      else random_answer_per_question(all_answers, st.session_state["verifier_answer_seed"]))
    except ValueError as exc:
        st.error(f"Evaluation split configuration error: {exc}"); return
    by_example_id = {answer.example_id: answer for answer in all_answers}
    manual_ids = st.session_state["verifier_manual_example_ids"]
    manual_answers = [by_example_id[item] for item in manual_ids if item in by_example_id]
    candidate_ids = {answer.example_id for answer in candidates}
    candidates.extend(answer for answer in manual_answers if answer.example_id not in candidate_ids)
    st.subheader("Choose answer records to run")
    if pool in saved_splits:
        st.caption(str(split_metadata["role"]))
        st.caption("Saved ten-record set · all Question IDs are unique across prompt-development, validation, and final-test.")
    else:
        st.caption(f"Each row is a different Question ID and contains one randomly selected machine answer. Click up to {MAX_BATCH_SIZE} rows.")
    selector = st.dataframe(
        [{"question ID": answer.question_id, "method ID": answer.method_id, "question": answer.question,
          "added manually": "Yes" if answer.example_id in manual_ids else ""}
         for answer in candidates],
        hide_index=True,
        use_container_width=True,
        height=430,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"verifier_answer_selector_{pool}_{st.session_state['verifier_answer_seed']}",
    )
    selected_indices = [index for index in selector.selection.rows if 0 <= index < len(candidates)]
    selected_by_id = {candidates[index].example_id: candidates[index] for index in selected_indices}
    selected_by_id.update({answer.example_id: answer for answer in manual_answers})
    selected = list(selected_by_id.values())
    if len(selected) > MAX_BATCH_SIZE:
        st.error(f"You selected {len(selected)} records. Select {MAX_BATCH_SIZE} or fewer before running the Verifier.")
    elif selected:
        st.caption("Selected: " + ", ".join(f"{answer.question_id} · {answer.method_id}" for answer in selected))
    else:
        st.info(f"Select one to {MAX_BATCH_SIZE} rows in the table to prepare a run.")
    st.info("Sent to Gemini: question/method, machine answer, sentence IDs/text, and cited PMID plus evidence excerpts. Human labels, expert answer, and nuggets are used only after generation.")
    st.caption(f"One Gemini request is made per selected answer ({len(selected)} planned request(s)).")
    final_test_confirmed = True
    if pool == "final-test":
        final_test_confirmed = st.checkbox("I confirm the prompt version, model, and settings are locked before using final-test.")
        st.warning("If this final-test result causes a prompt, model, or setting change, mark it as development use and create a new unseen final-test split for the next final evaluation.")
    run_column, fresh_column = st.columns(2)
    run_clicked = run_column.button(
        f"Run Verifier on {len(selected)} answer(s)",
        type="primary",
        disabled=not os.getenv("GOOGLE_API_KEY") or not selected or len(selected) > MAX_BATCH_SIZE or not final_test_confirmed,
    )
    fresh_targets = selected
    fresh_clicked = fresh_column.button(
        f"Force fresh Gemini run on {len(fresh_targets)} selected answer(s)",
        disabled=not os.getenv("GOOGLE_API_KEY") or not fresh_targets or len(fresh_targets) > MAX_BATCH_SIZE or not final_test_confirmed,
        help="Ignores cached responses and writes the newly generated response as the current local cache entry.",
    )
    if run_clicked or fresh_clicked:
        run_targets, force_fresh = (fresh_targets, True) if fresh_clicked else (selected, False)
        cached, saved, results = load_cache(CACHE), [], []
        progress = st.progress(0)
        for index, answer in enumerate(run_targets, 1):
            key = cache_key(answer, model, prompt_version)
            if force_fresh or key not in cached:
                raw = asyncio.run(run_medaesqa_verifier(prompt_for(answer.prompt_input(), prompt_version), model))
                saved.append({"key": key, "raw_output": raw, "refreshed_at_utc": datetime.now(timezone.utc).isoformat()})
            else:
                item = cached[key]
                raw = item["raw_output"]
            results.append({"answer": answer, "prediction": parse_workbench_prediction(answer, raw)})
            progress.progress(index / len(run_targets))
        if saved: append_cache(CACHE, saved)
        append_run_log(RUN_LOG, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split_name": pool if pool in saved_splits else "explore",
            "prompt_version": prompt_version,
            "model": model,
            "model_settings": {"model": model},
            "pool": pool,
            "cache_mode": "force_fresh" if force_fresh else "reuse_cached",
            "selection_seed": split_metadata["selection_seed"] if split_metadata else st.session_state["verifier_answer_seed"],
            "selected_records": [{"question_id": answer.question_id, "method_id": answer.method_id} for answer in run_targets],
        })
        st.session_state["verifier_results"] = results
        st.session_state["verifier_result_ids"] = [answer.example_id for answer in run_targets]
        st.session_state["verifier_result_prompt_version"] = prompt_version
    results = st.session_state.get("verifier_results", [])
    if not results:
        if pool in saved_splits:
            supplementary_review_view(cached_result_rows(candidates, model, prompt_version), pool)
        return
    if ([answer.example_id for answer in selected] != st.session_state.get("verifier_result_ids", [])
            or prompt_version != st.session_state.get("verifier_result_prompt_version")):
        st.info("The results below are from the last completed run. Click Run Verifier to evaluate the current row selection.")
    st.caption(f"Results shown: prompt **{PROMPT_VERSION_LABELS.get(st.session_state.get('verifier_result_prompt_version'), prompt_version)}** · one Gemini request per answer.")
    comparisons = [compare(row["answer"], row["prediction"]) for row in results]
    export = export_review_summary(results)
    st.download_button(
        "Download review summary (JSON)",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="verifier_review_summary.json",
        mime="application/json",
        help="Contains separate answer_accuracy, sentence_relevance, and evidence_relation lists for the completed run. Downloading never calls Gemini.",
    )
    st.subheader("Paired agreement and Cohen’s κ")
    agreement = {name: task_agreement_statistics(comparisons, name) for name in TASKS}
    summary_rows = []
    for name in TASKS:
        metrics = agreement[name]
        kappa_display = (
            "Not evaluated" if metrics["kappa"] is None and metrics["kappa_reason"].startswith("Not evaluated")
            else "Undefined" if metrics["kappa"] is None
            else f"{metrics['kappa']:.3f}"
        )
        summary_rows.append({
            "Metric": name.replace("_", " ").title(),
            "Scored pairs": metrics["scored_pairs"],
            "Correct": metrics["correct"],
            "Incorrect": metrics["incorrect"],
            "Exact agreement": "Undefined" if metrics["exact_agreement"] is None else f"{metrics['exact_agreement']:.1%}",
            "Error rate": "Undefined" if metrics["error_rate"] is None else f"{metrics['error_rate']:.1%}",
            "Cohen’s κ": kappa_display,
            "Skipped pairs": metrics["skipped_pairs"],
            "Unscored total": metrics["unscored_pairs"],
            "Missing gold": metrics["missing_gold_pairs"],
            "Missing prediction": metrics["missing_prediction_pairs"],
            "Invalid gold": metrics["invalid_gold_label_pairs"],
            "Invalid prediction": metrics["invalid_prediction_label_pairs"],
            "Malformed responses": metrics["malformed_responses"],
            "Target": KAPPA_TARGETS.get(name, "No documented agreement/κ target for this metric."),
        })
    st.dataframe(summary_rows, hide_index=True, use_container_width=True)
    matrix_task = st.selectbox(
        "Confusion matrix metric",
        TASKS,
        format_func=lambda value: value.replace("_", " ").title(),
        help="Rows are human gold labels; columns are Gemini predictions. Only valid paired labels are counted.",
        key="kappa_confusion_task",
    )
    matrix = agreement[matrix_task]
    st.caption(matrix["kappa_reason"] or "Unweighted Cohen’s κ adjusts exact agreement for agreement expected from the observed label distributions.")
    st.caption("Rows (vertical axis) are Human gold / actual labels. Columns (horizontal axis) are Gemini predictions. Each cell counts items with that row’s human label and that column’s Gemini prediction; row ‘no’, column ‘yes’ means human gold ‘no’ and Gemini predicted ‘yes’.")
    st.dataframe(
        [{"Human gold / actual ↓  |  Gemini prediction →": gold, **matrix["confusion_matrix"][gold]} for gold in matrix["labels"]],
        hide_index=True,
        use_container_width=True,
    )
    if pool in saved_splits:
        supplementary_review_view(results or cached_result_rows(candidates, model, prompt_version), pool)
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
                if row.get("diagnostic"):
                    st.caption(f"Mismatch diagnostic (heuristic, not a human rationale): {row['diagnostic']}")
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
    agent = st.sidebar.radio("Agent workbench", ["Baseline", "Verifier", "Extractor", "Simplifier", "Refiner", "Readability"])
    if agent == "Verifier":
        st.title("FaithfulMed Verifier Workbench"); st.caption("Exploratory MedAESQA results only—not final research results.")
        verifier_page()
    elif agent == "Refiner":
        from evals.refiner_workbench import refiner_page
        refiner_page()
    elif agent == "Readability":
        from evals.readability_workbench import readability_page
        readability_page()
    elif agent == "Baseline":
        from evals.baseline_workbench import baseline_page
        baseline_page()
    else: placeholder_page(agent)


if __name__ == "__main__": main()
