"""Streamlit demo and evaluation workspace for the complete FaithfulMed pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st
from markdown_it import MarkdownIt

from agents.factual_evaluator import indexed_text_spans
from agents.production_pipeline import MAX_REVISION_CYCLES
from agents.verifier import VERIFIER_DEFAULT_MODEL
from evals.baseline_system import (
    DEFAULT_BASELINE_RESULTS_PATH,
    DEFAULT_BASELINE_REVIEW_PATH,
    BaselineEnrichmentCache,
    enrich_baseline_record,
    has_compatible_source_matching,
    has_factual_evaluation,
    load_baseline_records,
    run_forced_baseline,
)
from evals.claim_review import ReviewDecisionStore, review_metrics, reviewable_claims
from evals.system_demo import (
    AppendOnlyRunCache,
    SystemExample,
    aggregate_metrics,
    load_system_examples,
    record_for_download,
    refresh_cached_factual_evaluation,
    run_system_example,
    select_examples,
)
from evals.source_matching import (
    EmbeddingCache,
    EmbeddingSettings,
    apply_match_threshold,
    build_source_matches,
    google_batch_embedder,
    load_source_matching_config,
    settings_from_config,
    source_match_metrics,
    split_sentences,
)

ROOT = Path(__file__).resolve().parent
INLINE_MARKDOWN = MarkdownIt("commonmark", {"html": False})
AMBIGUOUS_MATCH_MARGIN = 0.05


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
      .fm-explanation {font-size:1rem; line-height:1.75; margin:.4rem 0 1rem;}
      .fm-explanation p {margin:.45rem 0 .9rem;}
      .fm-explanation ul, .fm-explanation ol {margin:.35rem 0 1rem; padding-left:1.55rem;}
      .fm-explanation li {margin:.28rem 0; padding-left:.15rem;}
      .fm-explanation h1, .fm-explanation h2, .fm-explanation h3,
      .fm-explanation h4, .fm-explanation h5, .fm-explanation h6 {margin:1rem 0 .4rem; line-height:1.3;}
      .fm-source-sentence {position:relative; display:inline; border-bottom:1px dotted #46867f; cursor:help;}
      .fm-source-sentence:focus {outline:2px solid #208174; outline-offset:2px;}
      .fm-source-popup {visibility:hidden; opacity:0; position:absolute; z-index:999; left:0; bottom:calc(100% + .55rem);
        width:min(34rem,80vw); padding:.75rem .85rem; border:1px solid #b9cbc7; border-radius:.45rem;
        background:#fff; color:#172321; box-shadow:0 7px 24px rgba(23,35,33,.18); font-size:.82rem;
        line-height:1.45; transition:opacity .12s ease; pointer-events:none; white-space:normal;}
      .fm-source-sentence:hover .fm-source-popup, .fm-source-sentence:focus .fm-source-popup,
      .fm-source-sentence:focus-within .fm-source-popup {visibility:visible; opacity:1;}
      .fm-source-popup strong {color:#14685f;}
      @media (prefers-color-scheme: dark) {.fm-source-popup {background:#172321; color:#f4f7f6; border-color:#4c6560;}}
    </style>
    """, unsafe_allow_html=True)


def fmt(value: float | int | None, suffix: str = "", digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}{suffix}"


def deferred_download(
    label: str,
    *,
    data_factory: Callable[[], str | bytes],
    file_name: str,
    mime: str,
    key: str,
    cache_token: str,
) -> None:
    """Prepare potentially large download data only after an explicit click."""
    state_key = f"faithfulmed_prepared_download_{key}"
    prepared = st.session_state.get(state_key)
    if not isinstance(prepared, dict) or prepared.get("cache_token") != cache_token:
        st.session_state.pop(state_key, None)
        prepared = None

    if st.button(f"Prepare {label}", key=f"{key}_prepare"):
        with st.spinner(f"Preparing {file_name}…"):
            prepared = {
                "cache_token": cache_token,
                "data": data_factory(),
            }
        st.session_state[state_key] = prepared

    if prepared is not None:
        st.download_button(
            label,
            data=prepared["data"],
            file_name=file_name,
            mime=mime,
            key=f"{key}_download",
            on_click="ignore",
        )
    else:
        st.caption("The file is generated only when requested.")


def metric_summary(records: list[dict[str, Any]]) -> None:
    summary = aggregate_metrics(records)
    st.subheader("Evaluation metrics")
    st.caption(
        "Scores use the displayed patient-friendly explanation. Factual fidelity is an automated "
        "report-and-UMLS evidence estimate—not a human agreement score."
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


def source_match_summary(records: list[dict[str, Any]], threshold: float) -> None:
    if not st.toggle(
        "Load embedding source-match summary",
        value=False,
        key="embedding_source_match_summary",
    ):
        return
    matching = [
        record["source_matching"] for record in records
        if isinstance(record.get("source_matching"), dict)
    ]
    with st.expander("Embedding source-match summary", expanded=False):
        if not matching:
            st.info("No source-similarity results are available for the current outputs.")
            return
        metrics = source_match_metrics(
            matching, threshold, ambiguity_margin=AMBIGUOUS_MATCH_MARGIN,
        )
        total = metrics["sentences"]
        first = st.columns(3)
        first[0].metric(
            "Above-threshold source match rate",
            fmt(metrics["above_threshold_rate"], "%"),
            help=(
                "Formula: explanation sentences whose closest source match is at or above the "
                "threshold ÷ all matched explanation sentences × 100. In plain language, this is "
                "the percentage of explanation sentences that have at least one reasonably similar "
                "sentence in the original report. It measures similarity—not factual support."
            ),
        )
        first[1].metric(
            "Below-threshold source match rate",
            fmt(metrics["below_threshold_rate"], "%"),
            help=(
                "The percentage of explanation sentences whose closest original-report sentence is "
                "below the selected threshold. The closest candidate is still shown for inspection. "
                "A low score does not automatically mean the explanation is false."
            ),
        )
        first[2].metric(
            "Multiple-match rate",
            fmt(metrics["multiple_match_rate"], "%"),
            help=(
                "The percentage of explanation sentences with at least two original-report sentences "
                "at or above the threshold. This can indicate that a sentence combines information, "
                "or that several source sentences discuss similar material."
            ),
        )
        second = st.columns(2)
        second[0].metric(
            "Ambiguous-match rate",
            fmt(metrics["ambiguous_match_rate"], "%"),
            help=(
                f"The percentage of explanation sentences with at least two above-threshold matches "
                f"whose top two cosine scores differ by no more than {AMBIGUOUS_MATCH_MARGIN:.2f}. "
                "In plain language, the embedding model cannot clearly distinguish which of those "
                "source sentences is the best match."
            ),
        )
        second[1].metric(
            "Minimum closest-match score",
            fmt(metrics["minimum_closest_match_score"], digits=3),
            help=(
                "The lowest best-match cosine similarity among the displayed explanation sentences. "
                "It identifies the sentence with the weakest source candidate, but it is not a "
                "factuality score or support probability."
            ),
        )
        st.caption(
            f"Threshold {threshold:.2f} · {total} sentence(s) assessed · "
            f"Above: {metrics['above_threshold']} · Below: {metrics['below_threshold']} · "
            f"Multiple matches: {metrics['multiple_match_sentences']} · "
            f"Ambiguous: {metrics['ambiguous_match_sentences']}. "
            "All values are sentence-level embedding similarities, not claim-level factual judgments."
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


def readability_distributions(records: list[dict[str, Any]]) -> None:
    """Preserve the baseline Workbench distribution views in the combined app."""
    if not st.toggle("Load readability distributions", value=False, key="baseline_readability_distributions"):
        return
    rows = [
        record["readability_metrics"]
        for record in records
        if isinstance(record.get("readability_metrics"), dict)
    ]
    if not rows:
        st.info("No cached readability metrics are available.")
        return
    frame = pd.DataFrame(rows)
    definitions = (
        ("flesch_kincaid_grade", "Flesch–Kincaid grade"),
        ("smog_index", "SMOG index"),
        ("jargon_density", "Jargon-density proxy"),
        ("word_count", "Word count"),
    )
    for metric, label in definitions:
        if metric not in frame:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if values.empty:
            continue
        st.markdown(f"**{label}**")
        if values.nunique() == 1:
            st.caption(f"All {len(values)} outputs have a value of {values.iloc[0]:.2f}.")
            continue
        bins = pd.cut(values, bins=min(10, values.nunique()), duplicates="drop")
        counts = bins.value_counts(sort=False).sort_index()
        chart = pd.DataFrame({
            "Range": [f"{interval.left:.2f}–{interval.right:.2f}" for interval in counts.index],
            "Outputs": counts.values,
        })
        st.bar_chart(chart, x="Range", y="Outputs")


def _match_popup(item: dict[str, Any]) -> str:
    best = item.get("best_match")
    if not best:
        return "No source sentence match is available."
    pieces = [
        "<strong>Closest similarity match</strong><br>",
        html.escape(best["source_text"]),
        f"<br><strong>Cosine similarity:</strong> {best['score']:.3f}",
        f"<br><strong>Status:</strong> {html.escape(best['threshold_status'])}",
    ]
    others = item.get("other_possible_matches") or []
    if others:
        pieces.append("<br><br><strong>Other possible matches</strong>")
        for match in others:
            pieces.extend([
                "<br>• ", html.escape(match["source_text"]),
                f" <em>({match['score']:.3f})</em>",
            ])
    return "".join(pieces)


def _hover_sentence_html(item: dict[str, Any], display_text: str) -> str:
    """Render safe inline Markdown inside one hoverable sentence wrapper."""
    sentence = INLINE_MARKDOWN.renderInline(display_text)
    popup = _match_popup(item)
    label = html.escape(
        f"Source similarity for {item['explanation_sentence_id']}. "
        "Hover or focus for the closest source sentence.",
        quote=True,
    )
    return (
        f'<span class="fm-source-sentence" tabindex="0" aria-label="{label}">'
        f'{sentence}<span class="fm-source-popup" role="tooltip">{popup}</span></span>'
    )


def hoverable_explanation_html(
    final_explanation: str,
    rendered_matching: dict[str, Any],
) -> str:
    """Preserve Markdown blocks while adding sentence-level hover targets."""
    items = list(rendered_matching.get("explanation_sentences") or [])
    item_index = 0
    blocks: list[str] = []
    open_list: str | None = None

    def close_list() -> None:
        nonlocal open_list
        if open_list:
            blocks.append(f"</{open_list}>")
            open_list = None

    for raw_line in final_explanation.splitlines():
        line = raw_line.strip()
        if not line:
            close_list()
            continue
        sentence_count = len(split_sentences(line, "T"))
        line_items = items[item_index:item_index + sentence_count]
        item_index += sentence_count
        if not line_items:
            close_list()
            blocks.append(f"<p>{INLINE_MARKDOWN.renderInline(line)}</p>")
            continue

        unordered = re.match(r"^([-*+])\s+", line)
        ordered = re.match(r"^(\d+)[.)]\s+", line)
        heading = re.match(r"^(#{1,6})\s+", line)
        prefix_length = 0
        block_tag = "p"
        list_tag: str | None = None
        if unordered:
            prefix_length = unordered.end()
            list_tag = "ul"
        elif ordered:
            prefix_length = ordered.end()
            list_tag = "ol"
        elif heading:
            prefix_length = heading.end()
            block_tag = f"h{len(heading.group(1))}"

        sentence_html = []
        for position, item in enumerate(line_items):
            display = item["explanation_text"]
            if position == 0 and prefix_length:
                display = display[prefix_length:]
            sentence_html.append(_hover_sentence_html(item, display))
        content = " ".join(sentence_html)

        if list_tag:
            if open_list != list_tag:
                close_list()
                blocks.append(f"<{list_tag}>")
                open_list = list_tag
            blocks.append(f"<li>{content}</li>")
        else:
            close_list()
            blocks.append(f"<{block_tag}>{content}</{block_tag}>")

    close_list()
    for item in items[item_index:]:
        blocks.append(f"<p>{_hover_sentence_html(item, item['explanation_text'])}</p>")
    return '<div class="fm-explanation">' + "".join(blocks) + "</div>"


def render_hoverable_explanation(record: dict[str, Any], threshold: float) -> None:
    """Render escaped sentence spans with hover/focus matches and a tap fallback."""
    matching = record.get("source_matching")
    if not isinstance(matching, dict):
        st.write(record.get("final_explanation") or "")
        if record.get("source_matching_error"):
            st.warning(f"Source matching unavailable: {record['source_matching_error']}")
        return
    rendered = apply_match_threshold(matching, threshold)
    st.markdown(
        hoverable_explanation_html(record.get("final_explanation") or "", rendered),
        unsafe_allow_html=True,
    )
    st.caption(
        "Dotted sentences have embedding-based source candidates. Similarity is not proof of factual support."
    )
    details_key = f"source_match_details_{record.get('run_id') or record.get('example_id', 'run')}"
    if st.toggle("Load click/tap source-match details", value=False, key=details_key):
        for item in rendered["explanation_sentences"]:
            best = item.get("best_match")
            st.markdown(f"**{item['explanation_sentence_id']} — {item['explanation_text']}**")
            if best:
                st.write(
                    f"Closest: {best['source_sentence_id']} · {best['score']:.3f} · "
                    f"{best['threshold_status']}"
                )
                st.code(best["source_text"], wrap_lines=True)
                for other in item.get("other_possible_matches") or []:
                    st.write(f"Other possible match: {other['source_sentence_id']} · {other['score']:.3f}")
                    st.code(other["source_text"], wrap_lines=True)


def render_source_matching(
    record: dict[str, Any],
    threshold: float,
    embedding_cache: EmbeddingCache,
    *,
    key_prefix: str = "system",
) -> None:
    matching = record.get("source_matching")
    st.caption(
        f"Provisional threshold: {threshold:.2f}. This only filters existing cosine scores and "
        "does not call the embedding API. Similarity matches are not factual-support judgments."
    )
    if not isinstance(matching, dict):
        if record.get("source_matching_error"):
            st.error(f"Source matching unavailable: {record['source_matching_error']}")
        else:
            st.info("No source-matching result is attached to this saved run. Run the example to compute it.")
    else:
        rendered = apply_match_threshold(matching, threshold)
        rows = []
        for item in rendered["explanation_sentences"]:
            best = item.get("best_match") or {}
            rows.append({
                "Explanation ID": item["explanation_sentence_id"],
                "Explanation sentence": item["explanation_text"],
                "Explanation vector": item["embedding_source"],
                "Closest source ID": best.get("source_sentence_id"),
                "Closest source sentence": best.get("source_text"),
                "Closest source vector": best.get("embedding_source"),
                "Cosine similarity": best.get("score"),
                "Threshold status": best.get("threshold_status"),
                "Other matches above threshold": len(item.get("other_possible_matches") or []),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        deferred_download(
            "Download selected source matches (JSON)",
            data_factory=lambda: json.dumps(rendered, ensure_ascii=False, indent=2),
            file_name=f"{record.get('example_id', 'run')}_source_matches.json",
            mime="application/json",
            key=f"{key_prefix}_source_matches_download",
            cache_token=(
                f"{record.get('run_id', record.get('example_id', 'run'))}:"
                f"{threshold:.4f}"
            ),
        )
        if st.toggle(
            "Load run vector-cache status",
            value=False,
            key=f"{key_prefix}_source_match_vector_status",
        ):
            status_rows = [
                {
                    "Role": "source",
                    "Sentence ID": item["source_sentence_id"],
                    "Sentence": item["source_text"],
                    "Vector source": item["embedding_source"],
                }
                for item in matching["source_sentences"]
            ] + [
                {
                    "Role": "explanation",
                    "Sentence ID": item["explanation_sentence_id"],
                    "Sentence": item["explanation_text"],
                    "Vector source": item["embedding_source"],
                }
                for item in matching["explanation_sentences"]
            ]
            st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)

    if st.toggle(
        "Load embedding-cache debug tools",
        value=False,
        key=f"{key_prefix}_embedding_cache_debug",
    ):
        stats = embedding_cache.stats()
        st.caption(
            f"SQLite: `{stats['path']}` · {stats['vectors']} unique vectors · "
            f"{stats['usages']} sentence usages"
        )
        entries = embedding_cache.inspect_entries(limit=100)
        if entries:
            st.dataframe(pd.DataFrame(entries), hide_index=True, use_container_width=True)
        else:
            st.caption("The embedding cache is empty.")
        deferred_download(
            "Export embedding cache (JSONL)",
            data_factory=embedding_cache.export_jsonl,
            file_name="faithfulmed_source_embedding_cache.jsonl",
            mime="application/x-ndjson",
            key=f"{key_prefix}_embedding_cache_export",
            cache_token=f"{stats['vectors']}:{stats['usages']}",
        )


def render_trace(record: dict[str, Any], *, key_prefix: str = "system") -> None:
    trace = record.get("pipeline_trace") or {}
    stages = trace.get("stages", [])
    if stages:
        st.dataframe(pd.DataFrame([
            {
                "Stage": stage.get("agent", "Stage"),
                "Iteration": stage.get("iteration"),
                "Status": stage.get("status", "unknown"),
                "Started": stage.get("started_at_utc"),
                "Completed": stage.get("completed_at_utc"),
            }
            for stage in stages
        ]), hide_index=True, use_container_width=True)
    details_key = (
        f"{key_prefix}_trace_details_"
        f"{record.get('run_id') or record.get('example_id', 'run')}"
    )
    if not st.toggle("Load full trace inputs and outputs", value=False, key=details_key):
        st.caption("Full stage payloads remain saved in the run cache and download.")
        return
    for stage in stages:
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


def record_with_reviews(
    record: dict[str, Any], store: ReviewDecisionStore, threshold: float | None = None,
) -> dict[str, Any]:
    complete = record_for_download(record)
    run_decisions = store.for_run(record.get("run_id", ""))
    complete["manual_claim_reviews"] = list(run_decisions.values())
    complete["human_review_metrics"] = review_metrics([record], store.latest())
    if threshold is not None and isinstance(record.get("source_matching"), dict):
        complete["source_matching"] = apply_match_threshold(record["source_matching"], threshold)
    return complete


def render_run_history(cache: AppendOnlyRunCache, store: ReviewDecisionStore) -> None:
    if not st.toggle("Browse saved run history", value=False, key="show_saved_run_history"):
        st.caption("Run history is loaded only when requested.")
        return
    history = list(reversed(cache.records()))
    st.caption(f"Saved run history: {len(history)} run(s)")
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


@st.fragment
def render_human_claim_review(
    records: list[dict[str, Any]],
    store: ReviewDecisionStore,
    selected: int,
    *,
    key_prefix: str = "system",
) -> None:
    review_summary(records, store.latest())
    record = records[selected]
    evaluator = record.get("factual_evaluator") or {}
    st.caption(
        f"Run ID: `{record.get('run_id', 'unavailable')}` · Candidates come directly from the "
        "existing factual evaluator and are not human gold labels."
    )
    run_key = record.get("run_id") or record.get("example_id", "run")
    if st.toggle(
        "Load original source used by the pipeline and factual evaluator",
        value=False,
        key=f"{key_prefix}_review_source_{run_key}",
    ):
        st.code(record.get("source_text") or "No source text", wrap_lines=True)
    adaptations = (record.get("post_generation_references") or {}).get("plain_language_adaptations")
    if adaptations and st.toggle(
        "Load PLABA human adaptations (reference only)",
        value=False,
        key=f"{key_prefix}_review_adaptations_{run_key}",
    ):
        st.caption("These were kept out of generation and are not required wording or source evidence.")
        for adaptation in adaptations:
            st.write(adaptation.get("text", ""))
    if evaluator.get("error"):
        st.error(f"Factual evaluation unavailable: {evaluator['error']}")
        if st.toggle(
            "Load raw evaluator response",
            value=False,
            key=f"{key_prefix}_review_raw_evaluator_{run_key}",
        ):
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
            definitions = claim.get("reference_definitions") or []
            if definitions:
                st.markdown("**Related UMLS reference definitions**")
                for definition in definitions:
                    st.markdown(
                        f"**{definition.get('source', 'UMLS')} "
                        f"{definition.get('source_id', '')} · {definition.get('term', '')}**"
                    )
                    st.code(definition.get("definition") or "", wrap_lines=True)
                st.caption("These definitions support general term explanations only, not patient-specific facts.")
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
                    st.rerun(scope="fragment")


def render_results(
    records: list[dict[str, Any]],
    review_store: ReviewDecisionStore,
    source_match_threshold: float,
    embedding_cache: EmbeddingCache,
    *,
    key_prefix: str = "system",
) -> None:
    if not records:
        return
    for record in records:
        refresh_cached_factual_evaluation(record)
    trace_label = "Generation provenance" if key_prefix == "baseline" else "Pipeline traces"
    labels = [f"{record['dataset']} · {record['split']} · {record['example_id']}" for record in records]

    per_example_table(records)
    selected = st.selectbox(
        "Inspect output",
        range(len(records)),
        format_func=lambda index: labels[index],
        key=f"{key_prefix}_selected_record",
    )
    record = records[selected]

    views = [
        "Overview", "Source similarities", "Human claim review",
        "Automatic factual evaluation", trace_label,
    ]
    view = st.segmented_control(
        "Result view",
        views,
        default="Overview",
        selection_mode="single",
        key=f"{key_prefix}_result_view",
    ) or "Overview"

    if view == "Overview":
        review_summary(records, review_store.latest())
        metric_summary(records)
        if record.get("result_source") == "baseline_fresh_generation":
            source_label = "New explanation generated by Force new run"
        elif record.get("cached_record_source") == "baseline_fresh_generation":
            source_label = "Loaded a saved Force new run from the baseline cache"
        elif str(record.get("result_source", "")).startswith("baseline_"):
            source_label = "Loaded from runs/baseline.jsonl; the baseline generation was not rerun"
        else:
            source_label = "Loaded from matching cache" if record.get("result_source") == "cache" else "Generated in this run"
        st.caption(source_label)
        if record.get("final_explanation"):
            st.markdown("#### Final patient-friendly explanation")
            render_hoverable_explanation(record, source_match_threshold)
        elif record.get("pipeline_trace", {}).get("status") == "not_accepted":
            st.warning("Latest draft — not accepted by the production Verifier")
            st.write(record["pipeline_trace"].get("latest_draft", ""))
        else:
            st.error(record.get("pipeline_trace", {}).get("error") or "This run has no final explanation.")
        if record.get("readability_error"):
            st.warning(f"Readability metrics unavailable: {record['readability_error']}")

    elif view == "Source similarities":
        source_match_summary(records, source_match_threshold)
        render_source_matching(
            record, source_match_threshold, embedding_cache,
            key_prefix=key_prefix,
        )

    elif view == "Human claim review":
        render_human_claim_review(
            records, review_store, selected, key_prefix=key_prefix,
        )

    elif view == "Automatic factual evaluation":
        evaluator = record.get("factual_evaluator") or {}
        parsed = evaluator.get("parsed_output")
        st.caption("Automated provisional labels using report evidence and allowed UMLS term definitions.")
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
                if st.toggle(
                    "Load skipped-claim details",
                    value=False,
                    key=f"{key_prefix}_skipped_claims_{record.get('run_id', selected)}",
                ):
                    st.json(parsed["skipped_claims"])
        elif evaluator.get("error"):
            st.error(f"Factuality unavailable: {evaluator['error']}")
            if st.toggle(
                "Load raw evaluator output",
                value=False,
                key=f"{key_prefix}_claim_raw_output_{record.get('run_id', selected)}",
            ):
                st.code(evaluator.get("raw_output") or "No output", language="json", wrap_lines=True)
        else:
            st.info((parsed or {}).get("status_reason") or "No factuality assessment is available.")

    else:
        render_trace(record, key_prefix=key_prefix)
        review_version = (
            review_store.path.stat().st_mtime_ns if review_store.path.exists() else 0
        )
        deferred_download(
            "Download selected run trace",
            data_factory=lambda: json.dumps(
                record_with_reviews(record, review_store, source_match_threshold),
                ensure_ascii=False,
                indent=2,
            ),
            file_name=f"{record['example_id']}_faithfulmed_run.json",
            mime="application/json",
            key=f"{key_prefix}_trace_download",
            cache_token=(
                f"{record.get('run_id', record.get('example_id'))}:"
                f"{source_match_threshold:.4f}:{review_version}"
            ),
        )


def render_live(
    container: Any,
    label: str,
    stages: dict[str, dict[str, Any]],
    evaluator_running: bool = False,
) -> None:
    with container.container():
        st.markdown(f"#### Running {label}")
        display_stages = [
            "Extractor", "UMLSLookup", "Simplifier", "Readability", "Verifier / Refiner",
            "Factual evaluator",
        ]
        states = {name: "waiting" for name in display_stages}
        for record in stages.values():
            name = "Verifier / Refiner" if record.get("agent") in {"Verifier", "Refiner"} else record.get("agent")
            if name in states:
                states[name] = record.get("status", "running")
        if evaluator_running:
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
                "UMLSLookup": "Looking up Extractor-selected terms in UMLS.",
                "Simplifier": "Simplifier is drafting the patient-friendly explanation.",
                "Verifier": "Verifier is checking the current draft.",
                "Refiner": "Refiner is applying targeted corrections.",
                "Readability": "Readability is improving the draft before its final verification.",
            }
            st.caption(messages.get(latest.get("agent"), f"{latest.get('agent')} is running."))
            st.markdown(
                f'<div class="fm-status"><strong>{latest.get("agent")}{suffix}</strong><br><small>{latest.get("status", "running").title()}</small></div>',
                unsafe_allow_html=True,
            )
        if evaluator_running:
            st.caption("The independent factual evaluator is checking the final output against report and allowed UMLS evidence.")
            st.markdown('<div class="fm-status"><strong>Independent factual evaluator</strong><br><small>Running</small></div>', unsafe_allow_html=True)


async def execute_examples(
    examples: list[SystemExample], pipeline_model: str, evaluator_model: str,
    force: bool, persist: bool, live_placeholder: Any,
    embedding_cache: EmbeddingCache, embedding_settings: EmbeddingSettings,
    embed_batch: Any, source_match_threshold: float,
) -> list[dict[str, Any]]:
    cache = AppendOnlyRunCache() if persist else None
    records: list[dict[str, Any]] = []
    progress = st.progress(0, text="Preparing selected examples")
    for position, example in enumerate(examples, start=1):
        stages: dict[str, dict[str, Any]] = {}
        evaluator_running = False
        label = f"{example.dataset} · {example.example_id} ({position}/{len(examples)})"

        def on_event(event: dict[str, Any]) -> None:
            if event.get("type") == "stage_chunk":
                return
            record = event.get("record")
            if record:
                stages[record["stage_id"]] = record
            render_live(live_placeholder, label, stages, evaluator_running)

        def on_evaluator_start() -> None:
            nonlocal evaluator_running
            evaluator_running = True
            render_live(live_placeholder, label, stages, evaluator_running)

        record = await run_system_example(
            example, pipeline_model, evaluator_model, cache=cache, force=force, persist=persist,
            on_pipeline_event=on_event, on_evaluator_start=on_evaluator_start,
        )
        final = record.get("final_explanation")
        if final:
            try:
                matching = build_source_matches(
                    record.get("source_text") or "", final,
                    report_id=(
                        f"{record.get('dataset')}:{record.get('split')}:"
                        f"{record.get('example_id')}:{record.get('run_id')}"
                    ),
                    cache=embedding_cache, embed_batch=embed_batch,
                    settings=embedding_settings,
                )
                matching["threshold_used_at_run"] = source_match_threshold
                record["source_matching"] = matching
                record.pop("source_matching_error", None)
            except Exception as exc:
                record["source_matching_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        progress.progress(position / len(examples), text=f"Completed {position} of {len(examples)}")
    live_placeholder.empty()
    progress.empty()
    return records


async def execute_forced_baseline_runs(
    records: list[dict[str, Any]],
    evaluator_model: str,
    *,
    cache: BaselineEnrichmentCache,
    embedding_cache: EmbeddingCache,
    embedding_settings: EmbeddingSettings,
    embed_batch: Any,
    live_placeholder: Any,
) -> list[dict[str, Any]]:
    """Regenerate and fully analyze selected baseline samples."""
    completed: list[dict[str, Any]] = []
    progress = st.progress(0, text="Preparing new baseline runs")
    for position, record in enumerate(records, start=1):
        label = f"{record.get('split')} · {record.get('example_id')}"
        live_placeholder.caption(f"Generating and evaluating baseline · {label}")

        enriched = await run_forced_baseline(
            record,
            evaluator_model,
            cache=cache,
            embedding_cache=embedding_cache,
            embedding_settings=embedding_settings,
            embed_batch=embed_batch,
        )
        completed.append(enriched)
        progress.progress(
            position / len(records),
            text=f"Completed {position} of {len(records)} new baseline runs",
        )
    live_placeholder.empty()
    progress.empty()
    return completed


async def execute_cached_baseline_evaluations(
    records: list[dict[str, Any]],
    evaluator_model: str,
    *,
    cache: BaselineEnrichmentCache,
    live_placeholder: Any,
) -> list[dict[str, Any]]:
    """Evaluate saved baseline text without regenerating or embedding it."""
    completed: list[dict[str, Any]] = []
    progress = st.progress(0, text="Preparing cached baseline evaluations")
    for position, record in enumerate(records, start=1):
        label = f"{record.get('split')} · {record.get('example_id')}"
        if has_factual_evaluation(record):
            live_placeholder.caption(f"Using cached factual evaluation · {label}")
        else:
            live_placeholder.caption(f"Evaluating saved baseline output · {label}")
        enriched = await enrich_baseline_record(
            record,
            evaluator_model,
            cache=cache,
            embedding_cache=None,
            embedding_settings=None,
            embed_batch=None,
            include_source_matching=False,
        )
        completed.append(enriched)
        progress.progress(
            position / len(records),
            text=f"Completed {position} of {len(records)} cached baseline evaluations",
        )
    live_placeholder.empty()
    progress.empty()
    return completed


def render_baseline_workspace(
    evaluator_model: str,
    source_match_threshold: float,
    embedding_cache: EmbeddingCache,
    embedding_settings: EmbeddingSettings,
) -> None:
    """Render cached baseline generations in the system evaluation interface."""
    enrichment_cache = BaselineEnrichmentCache()
    review_store = ReviewDecisionStore(DEFAULT_BASELINE_REVIEW_PATH)
    all_records = load_baseline_records(evaluator_model, cache=enrichment_cache)
    if not all_records:
        st.error(
            f"No successful baseline generations were found at `{DEFAULT_BASELINE_RESULTS_PATH}`."
        )
        return

    factual_evaluated_count = sum(has_factual_evaluation(record) for record in all_records)
    factual_scored_count = sum(
        (record.get("factual_metrics") or {}).get("claims_assessed", 0) > 0
        for record in all_records
    )
    matching_count = sum(
        has_compatible_source_matching(record, embedding_settings) for record in all_records
    )
    status = st.columns(4)
    status[0].metric("Cached generations", len(all_records))
    status[1].metric("Readability scored", sum(bool(record.get("readability_metrics")) for record in all_records))
    status[2].metric("Factuality scored", factual_scored_count)
    status[3].metric("Embedding matches", matching_count)
    st.caption(
        "Baseline text and readability scores come directly from `runs/baseline.jsonl`. "
        "The normal run is cache-only; only Force new run calls the baseline model. "
        f"Factual evaluator results are cached for {factual_evaluated_count}/{len(all_records)} outputs."
    )
    baseline_summary = aggregate_metrics(all_records)
    if baseline_summary.get("hallucination_rate") == 0:
        st.info(
            "Saved automatic result: 0/41 source-assessed outputs were flagged for a clinically "
            "meaningful hallucination. Gemini labeled all 782 assessed claims supported; one refusal "
            "is excluded. This is provisional—not human gold—and the human queue cannot correct a "
            "claim Gemini mistakenly labeled supported. See `docs/BASELINE_FACTUAL_AUDIT.md`."
        )

    available_splits = list(dict.fromkeys(str(record.get("split")) for record in all_records))
    selected_splits = st.multiselect(
        "Splits shown", available_splits, default=available_splits,
        key="baseline_splits_shown",
    )
    visible = [record for record in all_records if str(record.get("split")) in selected_splits]
    if not visible:
        st.info("Choose at least one split to view baseline results.")
        return

    with st.expander("Run baseline samples", expanded=True):
        st.write(
            "Open cached outputs makes no API calls. Evaluate cached baseline outputs keeps the "
            "saved baseline answer and calls only the independent factual evaluator when a "
            "compatible evaluation is missing. Force new run also asks Gemini for a new "
            "simplification and recalculates the other analyses."
        )
        run_all = st.checkbox(
            f"Select all {len(visible)} visible outputs", value=False,
            key="baseline_select_all_visible",
        )
        by_label = {
            f"{record.get('split')} · {record.get('example_id')} · {record.get('baseline_model')}": record
            for record in visible
        }
        if run_all:
            selected = list(visible)
            st.caption(f"Selected: all {len(selected)} visible cached outputs.")
        else:
            default_labels = list(by_label)[:min(5, len(by_label))]
            chosen = st.multiselect(
                "Cached outputs to analyze", list(by_label), default=default_labels,
                key="baseline_outputs_to_analyze",
            )
            selected = [by_label[label] for label in chosen]
        cached_factual = sum(has_factual_evaluation(record) for record in selected)
        cached_matching = sum(
            has_compatible_source_matching(record, embedding_settings) for record in selected
        )
        st.caption(
            f"Selected {len(selected)} · cached factual evaluations: {cached_factual} · "
            f"cached source-match records: {cached_matching}."
        )
        normal_column, evaluate_column, force_column = st.columns(3)
        normal = normal_column.button(
            "Open cached outputs", type="primary", disabled=not selected,
            use_container_width=True, key="baseline_run_selected_samples",
        )
        evaluate_cached = evaluate_column.button(
            "Evaluate cached baseline outputs", disabled=not selected,
            use_container_width=True, key="baseline_evaluate_cached_outputs",
            help=(
                "Uses the saved baseline explanations and runs only missing factual evaluations. "
                "It does not regenerate explanations or calculate source embeddings."
            ),
        )
        force = force_column.button(
            "Force new run", disabled=not selected, use_container_width=True,
            key="baseline_force_new_run",
            help=(
                "Generates a new baseline explanation, recalculates readability, embeds the new "
                "answer, and reruns the factual evaluator. Compatible source vectors are reused."
            ),
        )

    if normal:
        st.session_state["faithfulmed_baseline_records"] = selected
        st.session_state["faithfulmed_baseline_evaluator_model"] = evaluator_model

    if evaluate_cached:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is missing. Add it to `.env`, then rerun this page.")
        else:
            live_placeholder = st.empty()
            evaluated_records = asyncio.run(execute_cached_baseline_evaluations(
                selected,
                evaluator_model,
                cache=enrichment_cache,
                live_placeholder=live_placeholder,
            ))
            st.session_state["faithfulmed_baseline_records"] = evaluated_records
            st.session_state["faithfulmed_baseline_evaluator_model"] = evaluator_model

    if force:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is missing. Add it to `.env`, then rerun this page.")
        else:
            from google import genai

            client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            live_placeholder = st.empty()
            fresh_records = asyncio.run(execute_forced_baseline_runs(
                selected,
                evaluator_model,
                cache=enrichment_cache,
                embedding_cache=embedding_cache,
                embedding_settings=embedding_settings,
                embed_batch=google_batch_embedder(client),
                live_placeholder=live_placeholder,
            ))
            st.session_state["faithfulmed_baseline_records"] = fresh_records
            st.session_state["faithfulmed_baseline_evaluator_model"] = evaluator_model

    current = st.session_state.get("faithfulmed_baseline_records", [])
    if (
        st.session_state.get("faithfulmed_baseline_evaluator_model") != evaluator_model
        and current
    ):
        current = []
        st.info("Choose samples and run them to load results for the selected evaluator model.")
    if not current:
        st.info("Choose samples, then run cached results or force a new baseline generation.")
        return

    st.markdown('<div class="fm-rule"></div>', unsafe_allow_html=True)
    render_results(
        current,
        review_store,
        source_match_threshold,
        embedding_cache,
        key_prefix="baseline",
    )
    readability_distributions(current)
    review_version = review_store.path.stat().st_mtime_ns if review_store.path.exists() else 0
    baseline_ids = ":".join(str(record.get("run_id", "")) for record in current)
    deferred_download(
        "Download visible baseline results (JSON)",
        data_factory=lambda: json.dumps(
            [
                record_with_reviews(record, review_store, source_match_threshold)
                for record in current
            ],
            ensure_ascii=False,
            indent=2,
        ),
        file_name="faithfulmed_baseline_results.json",
        mime="application/json",
        key="baseline_results_download",
        cache_token=(
            f"{baseline_ids}:{source_match_threshold:.4f}:{review_version}"
        ),
    )


def main() -> None:
    load_env()
    st.set_page_config(page_title="FaithfulMed System Demo", page_icon="🩺", layout="wide")
    inject_styles()
    st.markdown('<div class="fm-kicker">FaithfulMed · system evaluation</div>', unsafe_allow_html=True)
    workspace = st.segmented_control(
        "Evaluation workspace",
        ["Full agentic system", "Single-agent baseline"],
        default="Full agentic system",
        selection_mode="single",
        label_visibility="collapsed",
        key="evaluation_workspace",
    )

    matching_config = load_source_matching_config()
    embedding_settings = settings_from_config(matching_config)
    embedding_cache = EmbeddingCache()

    if workspace == "Single-agent baseline":
        st.title("Single-agent baseline evaluation")
        st.markdown(
            '<div class="fm-lead">Inspect the 42 saved Gemini baseline explanations, add the same independent factual evaluation and source-similarity analysis used by the full system, and review flagged claims by hand.</div>',
            unsafe_allow_html=True,
        )
        with st.sidebar:
            st.subheader("Baseline analysis")
            evaluator_model = st.text_input(
                "Factual evaluator model", VERIFIER_DEFAULT_MODEL,
                key="baseline_evaluator_model",
            )
            source_match_threshold = st.slider(
                "Source match threshold", min_value=0.0, max_value=1.0,
                value=float(matching_config["embedding_match_threshold"]), step=0.01,
                key="baseline_source_match_threshold",
                help=(
                    "Changing this cutoff only relabels cached cosine scores; it does not call Gemini."
                ),
            )
            st.caption(
                f"Embeddings: `{embedding_settings.model}` · `{embedding_settings.task_type}` · "
                f"{embedding_settings.output_dimensionality} dimensions."
            )
            st.info("Baseline generation is cache-only on this page.")
        render_baseline_workspace(
            evaluator_model,
            source_match_threshold,
            embedding_cache,
            embedding_settings,
        )
        return

    st.title("End-to-end patient-friendly explanation pipeline")
    st.markdown(
        '<div class="fm-lead">Run a pasted medical report or a de-identified dataset split through Extractor, Simplifier, Readability, the bounded Verifier–Refiner loop, and an independent factual evaluation.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Models")
        pipeline_model = st.text_input("Pipeline model", VERIFIER_DEFAULT_MODEL)
        evaluator_model = st.text_input("Factual evaluator model", VERIFIER_DEFAULT_MODEL)
        st.caption(f"Maximum: {MAX_REVISION_CYCLES} Verifier-to-Refiner revision cycles.")
        source_match_threshold = st.slider(
            "Source match threshold", min_value=0.0, max_value=1.0,
            value=float(matching_config["embedding_match_threshold"]), step=0.01,
            help=(
                "Provisional, not calibrated. Changing this cutoff only relabels cached cosine "
                "scores; it does not call Gemini."
            ),
        )
        st.caption(
            f"Embeddings: `{embedding_settings.model}` · `{embedding_settings.task_type}` · "
            f"{embedding_settings.output_dimensionality} dimensions."
        )
        if st.button("Clear embedding cache", use_container_width=True):
            removed = embedding_cache.clear()
            st.success(f"Cleared {removed} cached embedding vector(s).")
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
            from google import genai

            embedding_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            records = asyncio.run(execute_examples(
                examples, pipeline_model, evaluator_model, force, persist, live_placeholder,
                embedding_cache, embedding_settings,
                google_batch_embedder(embedding_client), source_match_threshold,
            ))
            st.session_state["faithfulmed_system_records"] = records

    records = st.session_state.get("faithfulmed_system_records", [])
    if records:
        st.markdown('<div class="fm-rule"></div>', unsafe_allow_html=True)
        render_results(
            records, review_store, source_match_threshold, embedding_cache,
            key_prefix="system",
        )
        review_version = review_store.path.stat().st_mtime_ns if review_store.path.exists() else 0
        record_ids = ":".join(str(record.get("run_id", "")) for record in records)
        deferred_download(
            "Download current batch (JSON)",
            data_factory=lambda: json.dumps(
                [
                    record_with_reviews(record, review_store, source_match_threshold)
                    for record in records
                ],
                ensure_ascii=False,
                indent=2,
            ),
            file_name="faithfulmed_system_batch.json",
            mime="application/json",
            key="system_batch_download",
            cache_token=(
                f"{record_ids}:{source_match_threshold:.4f}:{review_version}"
            ),
        )


if __name__ == "__main__":
    main()
