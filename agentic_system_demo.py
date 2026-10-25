"""Streamlit demo and evaluation workspace for the complete FaithfulMed pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agents.factual_evaluator import indexed_text_spans
from agents.production_pipeline import MAX_REVISION_CYCLES
from agents.verifier import VERIFIER_DEFAULT_MODEL
from evals.claim_review import ReviewDecisionStore, review_metrics, reviewable_claims
from evals.system_demo import (
    AppendOnlyRunCache,
    SystemExample,
    aggregate_metrics,
    load_system_examples,
    record_for_download,
    run_system_example,
    select_examples,
)

ROOT = Path(__file__).resolve().parent


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and key:
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def inject_styles() -> None:
    st.markdown("""
    <style>
      .block-container {max-width:1180px; padding-top:1.8rem; padding-bottom:5rem;}
      .fm-kicker {font-size:.74rem; font-weight:750; letter-spacing:.15em; text-transform:uppercase; color:#167267;}
      .fm-lead {max-width:780px; color:#5d686c; font-size:1.02rem; line-height:1.6; margin-bottom:1.2rem;}
      .fm-rule {height:1px; background:#dfe5e3; margin:1rem 0 1.35rem;}
      .fm-status {padding:.7rem .85rem; border-left:3px solid #208174; background:#f3f8f7; margin:.45rem 0;}
      .fm-status small {color:#647075;}
      [data-testid="stMetric"] {background:transparent; border-top:2px solid #dce5e2; padding-top:.75rem;}
      div[data-testid="stExpander"] {border-left:0; border-right:0; border-radius:0;}
    </style>
    """, unsafe_allow_html=True)


def fmt(value: float | int | None, suffix: str = "", digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}{suffix}"


def metric_summary(records: list[dict[str, Any]]) -> None:
    summary = aggregate_metrics(records)
    st.subheader("Evaluation metrics")
    st.caption(
        "Scores use the final Readability output only. Factual fidelity is an automated, "
        "source-based estimate—not a human agreement score."
    )
    first = st.columns(4)
    first[0].metric("Outputs ≤ grade 8", fmt(summary["percent_at_or_below_grade_8"], "%"),
                    help="Challenge target: at least 80% of scored outputs.")
    first[1].metric("Mean FK grade", fmt(summary["mean_flesch_kincaid_grade"]))
    first[2].metric("Mean SMOG", fmt(summary["mean_smog"]),
                    help="Secondary readability measure; no separate challenge target.")
    first[3].metric("Mean words", fmt(summary["mean_word_count"], digits=0))
    second = st.columns(4)
    jargon = summary["mean_jargon_density"]
    second[0].metric("Jargon density proxy", fmt(jargon * 100 if jargon is not None else None, "%"),
                     help="Fraction of Dale–Chall difficult words. This is not a medical-terminology lookup.")
    second[1].metric("Refusal rate", fmt(summary["refusal_rate"], "%"),
                     help="Phrase-based detection of common refusal language such as ‘I cannot help.’")
    second[2].metric("Factual fidelity estimate", fmt(summary["factual_fidelity_estimate"], "%"),
                     help=(f"Supported claims / all {summary['claims_assessed']} validated, assessed claims; "
                           f"uncertain claims remain in the denominator and {summary['claims_skipped']} malformed claims are excluded."))
    second[3].metric("Hallucination rate", fmt(summary["hallucination_rate"], "%"),
                     help=f"Challenge target: under 10%. Based on {summary['hallucination_outputs_scored']} source-assessed outputs.")
    st.caption(
        f"Readability scored: {summary['readability_scored']}/{summary['outputs_total']} outputs · "
        f"Factuality scored: {summary['factual_outputs_scored']}/{summary['outputs_total']} outputs · "
        f"Claim validation coverage: {fmt(summary['claim_validation_coverage'], '%')} "
        f"({summary['claims_assessed']}/{summary['claims_returned']}) · "
        f"Uncertain claims: {summary['uncertain_claims']} · "
        f"Skipped malformed claims: {summary['claims_skipped']}"
    )
    st.info(
        "Challenge targets: at least 80% of outputs at or below grade 8; fewer than 10% of "
        "source-assessed outputs containing a clinically meaningful unsupported or contradicted claim. "
        "Jargon density is the Dale–Chall difficult-word proxy, not a clinical-term lookup."
    )


def review_summary(records: list[dict[str, Any]], decisions: dict[tuple[str, str], dict[str, Any]]) -> None:
    summary = review_metrics(records, decisions)
    st.subheader("Human claim-review status")
    st.caption(
        "The existing factual evaluator supplies the review queue. Clinically meaningful unsupported "
        "or contradicted claims default to Hallucination until you override or ignore them."
    )
    columns = st.columns(4)
    columns[0].metric(
        "Current hallucination rate",
        fmt(summary["hallucination_rate"], "%"),
        help="Outputs with at least one effective Hallucination label / eligible source-assessed outputs. Unreviewed candidates use Gemini's default.",
    )
    columns[1].metric(
        "No current hallucination",
        fmt(summary["no_hallucination_rate"], "%"),
        help=(
            "Output-level rate: the percentage of complete explanations with zero claims currently "
            "counted as hallucinations. Example: if 1 of 10 outputs still contains a Hallucination "
            "claim, this rate is 9/10 = 90%. Human overrides replace Gemini defaults."
        ),
    )
    columns[2].metric(
        "Candidate not-hallucination rate",
        fmt(summary["not_hallucination_claim_rate"], "%"),
        help=(
            "Claim-level rate: Not hallucination / (Hallucination + Not hallucination) across candidate "
            "claims; ignored claims are excluded. Example: if one output has 2 candidates and you mark "
            "1 Not hallucination while the other remains Hallucination, this rate is 1/2 = 50%."
        ),
    )
    columns[3].metric("Human-reviewed candidates", summary["human_reviewed"])
    st.caption(
        f"Output denominator: {summary['completed_outputs']} eligible source-assessed output(s); "
        f"{summary['unavailable_outputs']} evaluator-unavailable output(s) excluded. "
        f"Candidates — Hallucination: {summary['hallucination']} · Not hallucination: "
        f"{summary['not_hallucination']} · Ignored: {summary['ignored']}. "
        f"Gemini defaults still active: {summary['gemini_default']} · Human reviewed: {summary['human_reviewed']}. "
        f"Claim denominator: {summary['reviewed_claim_denominator']}; ignored claims excluded. "
        f"No qualifying evaluator claim: {summary['no_candidate_outputs']} output(s)."
    )


def per_example_table(records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        readable = record.get("readability_metrics") or {}
        factual = record.get("factual_metrics") or {}
        rows.append({
            "Dataset": record.get("dataset"), "Split": record.get("split"),
            "Example": record.get("example_id"), "Result": record.get("result_source"),
            "Accepted": record.get("accepted"), "FK grade": readable.get("flesch_kincaid_grade"),
            "≤ grade 8": readable.get("meets_target"), "SMOG": readable.get("smog_index"),
            "Jargon proxy %": (
                readable["jargon_density"] * 100 if readable.get("jargon_density") is not None else None
            ), "Words": readable.get("word_count"),
            "Refusal": readable.get("is_refusal"),
            "Fidelity estimate %": factual.get("factual_fidelity_estimate"),
            "Hallucination": factual.get("clinically_meaningful_hallucination"),
            "Claim coverage %": factual.get("claim_validation_coverage"),
            "Uncertain claims": factual.get("uncertain_claims"),
            "Skipped claims": factual.get("claims_skipped", 0),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_trace(record: dict[str, Any]) -> None:
    trace = record.get("pipeline_trace") or {}
    for stage in trace.get("stages", []):
        iteration = stage.get("iteration")
        suffix = f" · iteration {iteration}" if iteration is not None else ""
        with st.expander(f"{stage.get('agent', 'Stage')}{suffix} · {stage.get('status', 'unknown')}", expanded=False):
            st.markdown("**Input**")
            st.json(stage.get("input"))
            st.markdown("**Output**")
            st.code(stage.get("output") or "No output", wrap_lines=True)
            if stage.get("parsed_output") is not None:
                st.markdown("**Parsed output**")
                st.json(stage["parsed_output"])
            if stage.get("error"):
                st.error(stage["error"])
    evaluator = record.get("factual_evaluator") or {}
    with st.expander("Independent factual evaluator", expanded=False):
        st.markdown("**Input**")
        st.json(evaluator.get("request") or {
            "original_source_spans": indexed_text_spans(record.get("source_text") or "", "S"),
            "final_explanation_spans": indexed_text_spans(record.get("final_explanation") or "", "E"),
            "post_generation_references": record.get("post_generation_references") or {},
        })
        st.markdown("**Raw output**")
        st.code(evaluator.get("raw_output") or "No output", language="json", wrap_lines=True)
        if evaluator.get("error"):
            st.error(evaluator["error"])
def record_with_reviews(record: dict[str, Any], store: ReviewDecisionStore) -> dict[str, Any]:
    complete = record_for_download(record)
    run_decisions = store.for_run(record.get("run_id", ""))
    complete["manual_claim_reviews"] = list(run_decisions.values())
    complete["human_review_metrics"] = review_metrics([record], store.latest())
    return complete


def render_run_history(cache: AppendOnlyRunCache, store: ReviewDecisionStore) -> None:
    history = list(reversed(cache.records()))
    with st.expander(f"Saved run history ({len(history)})", expanded=False):
        if not history:
            st.caption("No saved dataset runs yet.")
            return
        options = list(range(len(history)))
        selected = st.selectbox(
            "Saved run",
            options,
            format_func=lambda index: (
                f"{history[index].get('created_at_utc', '')} · {history[index].get('dataset', '')} · "
                f"{history[index].get('split', '')} · {history[index].get('example_id', '')} · "
                f"{history[index].get('run_id', '')[:8]}"
            ),
            key="saved_run_history",
        )
        record = history[selected]
        decision_count = len(store.for_run(record.get("run_id", "")))
        st.caption(
            f"Run ID: `{record.get('run_id')}` · saved manual decisions: {decision_count} · "
            f"result source: {record.get('result_source', 'unknown')}"
        )
        if st.button("Open saved run", use_container_width=True):
            reopened = json.loads(json.dumps(record))
            reopened["cached_record_source"] = reopened.get("result_source")
            reopened["result_source"] = "history"
            st.session_state["faithfulmed_system_records"] = [reopened]
            st.rerun()


def render_human_claim_review(
    records: list[dict[str, Any]],
    store: ReviewDecisionStore,
) -> None:
    labels = [f"{record['dataset']} · {record['split']} · {record['example_id']}" for record in records]
    selected = st.selectbox(
        "Inspect screened output", range(len(records)),
        format_func=lambda index: labels[index], key="human_claim_record",
    )
    record = records[selected]
    evaluator = record.get("factual_evaluator") or {}
    st.caption(
        f"Run ID: `{record.get('run_id', 'unavailable')}` · Candidates come directly from the "
        "existing factual evaluator and are not human gold labels."
    )
    with st.expander("Original source used by the pipeline and factual evaluator", expanded=False):
        st.code(record.get("source_text") or "No source text", wrap_lines=True)
    adaptations = (record.get("post_generation_references") or {}).get("plain_language_adaptations")
    if adaptations:
        with st.expander("PLABA human adaptations (reference only)", expanded=False):
            st.caption("These were kept out of generation and are not required wording or source evidence.")
            for adaptation in adaptations:
                st.write(adaptation.get("text", ""))
    if evaluator.get("error"):
        st.error(f"Factual evaluation unavailable: {evaluator['error']}")
        with st.expander("Raw evaluator response", expanded=False):
            st.code(evaluator.get("raw_output") or "No output", language="json", wrap_lines=True)
        return
    parsed = evaluator.get("parsed_output")
    if not isinstance(parsed, dict) or parsed.get("status") != "scored":
        st.info((parsed or {}).get("status_reason") or "This run has no scored factual evaluation.")
        return
    claims = reviewable_claims(record)
    if not claims:
        st.success("The factual evaluator returned no clinically meaningful unsupported or contradicted claims.")
        st.caption("This is an automated evaluator result, not proof that the explanation is hallucination-free.")
        return
    decisions = store.for_run(record["run_id"])
    for claim in claims:
        claim_id = claim["claim_id"]
        saved_status = (decisions.get(claim_id) or {}).get("status")
        status = saved_status or "hallucination"
        status_label = status.replace("_", " ").title()
        if saved_status is None:
            status_label += " — Gemini default, not yet human-reviewed"
        with st.container(border=True):
            st.markdown(f"#### {claim_id} · Evaluator: {claim['classification'].title()}")
            st.markdown("**Exact claim text**")
            st.code(claim.get("claim_quote") or claim.get("claim_text") or "No claim text", wrap_lines=True)
            st.markdown("**Gemini evaluator reason**")
            st.write(claim.get("reason") or "No reason returned.")
            st.markdown("**Related source material**")
            sources = claim.get("source_quotes") or []
            if sources:
                for source in sources:
                    st.markdown(f"**{source.get('span_id', 'Source')}**")
                    st.code(source.get("text") or "", wrap_lines=True)
            elif claim.get("source_quote"):
                st.code(claim["source_quote"], wrap_lines=True)
            else:
                st.caption("The evaluator did not identify a supporting or contradicting source span.")
            st.markdown(f"**Current decision:** {status_label}")
            buttons = st.columns(3)
            choices = (
                ("Hallucination", "hallucination"),
                ("Not hallucination", "not_hallucination"),
                ("Ignore", "ignore"),
            )
            for column, (label, value) in zip(buttons, choices):
                if column.button(
                    label,
                    key=f"review_{record['run_id']}_{claim_id}_{value}",
                    type="primary" if status == value else "secondary",
                    use_container_width=True,
                ):
                    updated = store.latest()
                    updated[(record["run_id"], claim_id)] = {"status": value}
                    store.append(
                        record["run_id"], claim_id, value,
                        metric_snapshot=review_metrics(records, updated),
                    )
                    st.rerun()


def render_results(records: list[dict[str, Any]], review_store: ReviewDecisionStore) -> None:
    if not records:
        return
    decisions = review_store.latest()
    review_summary(records, decisions)
    metric_summary(records)
    per_example_table(records)
    overview, human_review, claims, traces = st.tabs([
        "Overview", "Human claim review", "Automatic factual evaluation", "Pipeline traces"
    ])
    labels = [f"{record['dataset']} · {record['split']} · {record['example_id']}" for record in records]
    with overview:
        selected = st.selectbox("Inspect output", range(len(records)), format_func=lambda index: labels[index], key="overview_record")
        record = records[selected]
        source_label = "Loaded from matching cache" if record.get("result_source") == "cache" else "Generated in this run"
        st.caption(source_label)
        if record.get("final_explanation"):
            st.markdown("#### Final patient-friendly explanation")
            st.write(record["final_explanation"])
        elif record.get("pipeline_trace", {}).get("status") == "not_accepted":
            st.warning("Latest draft — not accepted by the production Verifier")
            st.write(record["pipeline_trace"].get("latest_draft", ""))
        else:
            st.error(record.get("pipeline_trace", {}).get("error") or "This run has no final explanation.")
        if record.get("readability_error"):
            st.warning(f"Readability metrics unavailable: {record['readability_error']}")
    with human_review:
        render_human_claim_review(records, review_store)
    with claims:
        selected = st.selectbox("Inspect claims", range(len(records)), format_func=lambda index: labels[index], key="claim_record")
        record = records[selected]
        evaluator = record.get("factual_evaluator") or {}
        parsed = evaluator.get("parsed_output")
        st.caption("Automated provisional labels from the independent source-fidelity evaluator.")
        if parsed and parsed.get("status") == "scored":
            if parsed.get("claims"):
                st.dataframe(pd.DataFrame(parsed["claims"]), hide_index=True, use_container_width=True)
            else:
                st.warning("No evaluator claims passed quote and schema validation, so factuality is unavailable.")
            if parsed.get("skipped_claims"):
                st.warning(
                    f"{len(parsed['skipped_claims'])} claim(s) were excluded from the fidelity denominator "
                    "because their returned fields, span IDs, or legacy quotes could not be validated."
                )
                with st.expander("Inspect skipped claims", expanded=False):
                    st.json(parsed["skipped_claims"])
        elif evaluator.get("error"):
            st.error(f"Factuality unavailable: {evaluator['error']}")
            with st.expander("Raw evaluator output", expanded=False):
                st.code(evaluator.get("raw_output") or "No output", language="json", wrap_lines=True)
        else:
            st.info((parsed or {}).get("status_reason") or "No factuality assessment is available.")
    with traces:
        selected = st.selectbox("Inspect trace", range(len(records)), format_func=lambda index: labels[index], key="trace_record")
        render_trace(records[selected])
        st.download_button(
            "Download selected run trace", data=json.dumps(record_with_reviews(records[selected], review_store), ensure_ascii=False, indent=2),
            file_name=f"{records[selected]['example_id']}_faithfulmed_run.json", mime="application/json",
        )


def render_live(
    container: Any,
    label: str,
    stages: dict[str, dict[str, Any]],
    evaluator_output: str | None = None,
) -> None:
    with container.container():
        st.markdown(f"#### Running {label}")
        display_stages = [
            "Extractor", "Simplifier", "Verifier / Refiner", "Readability",
            "Factual evaluator",
        ]
        states = {name: "waiting" for name in display_stages}
        for record in stages.values():
            name = "Verifier / Refiner" if record.get("agent") in {"Verifier", "Refiner"} else record.get("agent")
            if name in states:
                states[name] = record.get("status", "running")
        if evaluator_output is not None:
            states["Factual evaluator"] = "running"
        columns = st.columns(len(display_stages))
        for column, name in zip(columns, display_stages):
            with column:
                st.markdown(f"**{name}**")
                st.caption(states[name].replace("_", " ").title())
        if stages:
            latest = list(stages.values())[-1]
            iteration = latest.get("iteration")
            suffix = f" · iteration {iteration}" if iteration is not None else ""
            messages = {
                "Extractor": "Extractor is reading the source.",
                "Simplifier": "Simplifier is drafting the patient-friendly explanation.",
                "Verifier": "Verifier is checking the current draft.",
                "Refiner": "Refiner is applying targeted corrections.",
                "Readability": "Readability is improving the accepted draft.",
            }
            st.caption(messages.get(latest.get("agent"), f"{latest.get('agent')} is running."))
            st.markdown(
                f'<div class="fm-status"><strong>{latest.get("agent")}{suffix}</strong><br><small>{latest.get("status", "running").title()}</small></div>',
                unsafe_allow_html=True,
            )
            if latest.get("output"):
                st.code(latest["output"], wrap_lines=True)
        if evaluator_output is not None:
            st.caption("The independent factual evaluator is checking the final output against the source.")
            st.markdown('<div class="fm-status"><strong>Independent factual evaluator</strong><br><small>Running</small></div>', unsafe_allow_html=True)
            if evaluator_output:
                st.code(evaluator_output, language="json", wrap_lines=True)


async def execute_examples(
    examples: list[SystemExample], pipeline_model: str, evaluator_model: str,
    force: bool, persist: bool, live_placeholder: Any,
) -> list[dict[str, Any]]:
    cache = AppendOnlyRunCache() if persist else None
    records: list[dict[str, Any]] = []
    progress = st.progress(0, text="Preparing selected examples")
    for position, example in enumerate(examples, start=1):
        stages: dict[str, dict[str, Any]] = {}
        evaluator_live: str | None = None
        label = f"{example.dataset} · {example.example_id} ({position}/{len(examples)})"

        def on_event(event: dict[str, Any]) -> None:
            record = event.get("record")
            if record:
                stages[record["stage_id"]] = record
            elif event.get("stage_id") in stages:
                stages[event["stage_id"]]["output"] = event.get("output", "")
            render_live(live_placeholder, label, stages, evaluator_live)

        def on_evaluator_text(_delta: str, accumulated: str) -> None:
            nonlocal evaluator_live
            evaluator_live = accumulated
            render_live(live_placeholder, label, stages, evaluator_live)

        def on_evaluator_start() -> None:
            nonlocal evaluator_live
            evaluator_live = ""
            render_live(live_placeholder, label, stages, evaluator_live)

        record = await run_system_example(
            example, pipeline_model, evaluator_model, cache=cache, force=force, persist=persist,
            on_pipeline_event=on_event, on_evaluator_start=on_evaluator_start,
            on_evaluator_text=on_evaluator_text,
        )
        records.append(record)
        progress.progress(position / len(examples), text=f"Completed {position} of {len(examples)}")
    live_placeholder.empty()
    progress.empty()
    return records


def main() -> None:
    load_env()
    st.set_page_config(page_title="FaithfulMed System Demo", page_icon="🩺", layout="wide")
    inject_styles()
    st.markdown('<div class="fm-kicker">FaithfulMed · system evaluation</div>', unsafe_allow_html=True)
    st.title("End-to-end patient-friendly explanation pipeline")
    st.markdown(
        '<div class="fm-lead">Run a pasted medical report or a de-identified dataset split through Extractor, Simplifier, the bounded Verifier–Refiner loop, Readability, and an independent factual evaluation.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Models")
        pipeline_model = st.text_input("Pipeline model", VERIFIER_DEFAULT_MODEL)
        evaluator_model = st.text_input("Factual evaluator model", VERIFIER_DEFAULT_MODEL)
        st.caption(f"Maximum: {MAX_REVISION_CYCLES} Verifier-to-Refiner revision cycles.")
        st.divider()
        st.caption("Individual-agent workbench: `python -m streamlit run workbench.py`")

    cache = AppendOnlyRunCache()
    review_store = ReviewDecisionStore()
    render_run_history(cache, review_store)

    mode = st.radio("Input source", ["Dataset batch", "Paste report"], horizontal=True)
    examples: list[SystemExample] = []
    force = False
    persist = mode == "Dataset batch"
    run_clicked = False

    if mode == "Dataset batch":
        controls = st.columns([1, 1.4, 1])
        split = controls[0].selectbox("Split", ["scratch", "dev", "test"])
        datasets = controls[1].multiselect("Dataset", ["MTSamples", "PLABA"], default=["MTSamples", "PLABA"])
        run_all = controls[2].checkbox("Run all examples", value=False)
        if split == "test":
            st.warning("Test is for final evaluation. Do not use its results to tune prompts or model settings.")
        available = load_system_examples(split, datasets) if datasets else []
        mts_count = sum(item.dataset == "MTSamples" for item in available)
        plaba_count = sum(item.dataset == "PLABA" for item in available)
        if "MTSamples" in datasets and mts_count == 0:
            st.warning(
                f"No adjudicated MTSamples records were found in "
                f"`data/MTSamples/{split}/adjudicated`."
            )
        st.caption(f"Available: {mts_count} MTSamples · {plaba_count} PLABA")
        count = len(available) if run_all else min(5, len(available))
        if not run_all and available:
            count = st.number_input("Number of examples", min_value=1, max_value=len(available), value=count, step=1)
        defaults = select_examples(available, None if run_all else int(count))
        by_label = {f"{item.dataset} · {item.example_id}": item for item in available}
        chosen = st.multiselect(
            "Selected examples", list(by_label),
            default=[f"{item.dataset} · {item.example_id}" for item in defaults],
            key=f"selected_{split}_{'_'.join(datasets)}_{count}_{run_all}",
        )
        examples = [by_label[label] for label in chosen]
        left, right = st.columns([1, 1])
        normal = left.button("Run selected examples", type="primary", disabled=not examples, use_container_width=True)
        force = right.button(
            "Force new run", disabled=not examples, use_container_width=True,
            help="Bypasses matching cache entries, makes fresh Gemini calls, and appends new records without deleting prior runs.",
        )
        run_clicked = normal or force
    else:
        report = st.text_area(
            "Medical report", height=300, placeholder="Paste a complete medical report here…",
            help="Pasted reports remain in Streamlit session state and are not automatically written to disk.",
        )
        if report.strip():
            digest = hashlib.sha256(report.strip().encode("utf-8")).hexdigest()[:12]
            examples = [SystemExample("Pasted", "session", f"pasted-{digest}", report.strip(), {}, "session")]
        run_clicked = st.button("Run pasted report", type="primary", disabled=not examples)
        st.caption("Use de-identified or approved demonstration text. This action may consume Gemini API quota.")

    live_placeholder = st.empty()
    if run_clicked:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is missing. Add it to `.env`, then rerun this page.")
        else:
            records = asyncio.run(execute_examples(
                examples, pipeline_model, evaluator_model, force, persist, live_placeholder,
            ))
            st.session_state["faithfulmed_system_records"] = records

    records = st.session_state.get("faithfulmed_system_records", [])
    if records:
        st.markdown('<div class="fm-rule"></div>', unsafe_allow_html=True)
        render_results(records, review_store)
        st.download_button(
            "Download current batch (JSON)",
            data=json.dumps(
                [record_with_reviews(record, review_store) for record in records],
                ensure_ascii=False,
                indent=2,
            ),
            file_name="faithfulmed_system_batch.json", mime="application/json",
        )


if __name__ == "__main__":
    main()
