import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJ_ROOT / "runs" / "baseline.jsonl"


def load_results():
    if not RESULTS_FILE.exists():
        return pd.DataFrame()

    records = []

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return pd.DataFrame(records)


def baseline_page():
    """Display saved Baseline evaluation metrics and graphs."""

    st.title("FaithfulMed Baseline Evaluation")
    st.write(
        "Results from the Single Gemini Baseline evaluated on "
        "the dev, scratch, and test datasets."
    )

    # Load previously saved evaluation results.
    results_file = (
        Path(__file__).resolve().parent
        / "runs"
        / "baseline.jsonl"
    )
    
    # Load previously saved evaluation results.
    if not RESULTS_FILE.exists():
        st.warning(
            f"No baseline results found at: {RESULTS_FILE}. "
            "Run the baseline evaluation from the backend first."
        )
        return

    records = []

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        st.info("The baseline results file is empty.")
        return

    st.write("Records loaded:", len(records))

    results_df = pd.DataFrame(records)

    # Check that the expected columns exist.
    if "status" not in results_df.columns:
        st.error("The results file does not contain a status column.")
        return

    # Overall evaluation status.
    successful = results_df[
        results_df["status"] == "success"
    ].copy()

    failed_count = int(
        (results_df["status"] == "failed").sum()
    )
    success_count = len(successful)
    total_count = len(results_df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Evaluations", total_count)
    col2.metric("Successful", success_count)
    col3.metric("Failed", failed_count)

    st.divider()

    if successful.empty:
        st.warning("There are no successful evaluations to graph.")
        st.dataframe(results_df, use_container_width=True)
        return

    # Expand the nested metrics dictionary into DataFrame columns.
    metrics_df = pd.json_normalize(
        successful["metrics"].apply(
            lambda value: value if isinstance(value, dict) else {}
        )
    )

    metrics_df.index = successful.index

    results_df = successful[
        ["split", "example_id", "model"]
    ].join(metrics_df)

    # Convert numeric metrics to numbers.
    numeric_metrics = [
        "flesch_kincaid_grade",
        "smog_index",
        "jargon_density",
        "word_count",
    ]

    for metric in numeric_metrics:
        if metric in results_df.columns:
            results_df[metric] = pd.to_numeric(
                results_df[metric], errors="coerce"
            )

    st.header("Readability Metrics")

    # Overall averages.
    metric_columns = [
        metric
        for metric in numeric_metrics
        if metric in results_df.columns
        and results_df[metric].notna().any()
    ]

    if metric_columns:
        summary = results_df[metric_columns].mean()

        metric_cols = st.columns(len(metric_columns))

        metric_labels = {
            "flesch_kincaid_grade": "Avg. FK Grade",
            "smog_index": "Avg. SMOG Index",
            "jargon_density": "Avg. Jargon Density",
            "word_count": "Avg. Word Count",
        }

        for column, metric in zip(metric_cols, metric_columns):
            value = summary[metric]
            column.metric(
                metric_labels.get(metric, metric),
                f"{value:.2f}",
            )

    
    # Graph each readability metric across all successful evaluations.
    st.header("Metric Distributions")
    st.caption(
        "Each graph includes all successful baseline evaluations "
        "across dev, scratch, and test."
    )

    metric_graphs = [
        (
            "flesch_kincaid_grade",
            "Flesch-Kincaid Grade Level",
            "Reading grade level of the explanation.",
        ),
        (
            "smog_index",
            "SMOG Index",
            "Estimated reading grade level based on sentence complexity.",
        ),
        (
            "jargon_density",
            "Jargon Density",
            "Proportion of words classified as medical jargon.",
        ),
        (
            "word_count",
            "Word Count",
            "Number of words in each explanation.",
        ),
    ]

    for metric, title, description in metric_graphs:
        if metric not in results_df.columns:
            continue

        values = results_df[metric].dropna()

        if values.empty:
            continue

        st.subheader(title)
        st.caption(description)

        # Show the distribution of individual evaluation results.
        if values.nunique() > 1:
            # Divide values into 10 ranges and count the values in each range.
            bins = pd.cut(values, bins=10)
            counts = bins.value_counts(sort=False).sort_index()

            # Create labels in ascending numeric order.
            labels = [
                f"{interval.left:.2f}–{interval.right:.2f}"
                for interval in counts.index
            ]

            val = "Value Range"
            col = "#b0def7"
            
            if metric == "flesch_kincaid_grade":
                val = "Grade Level"
                col = "#fa5050"
            elif metric == "smog_index":
                val = "SMOG Index"
                col = "#f7eaa8"
            elif metric == "jargon_density":
                val = "Jargon Density"
                col = "#8fb2eb"
            elif metric == "word_count":
                val = "Word Count"
                col = "#c8b0eb"

            # Explicitly preserve the order of the ranges.
            chart_data = pd.DataFrame({
                val: pd.Categorical(
                    labels,
                    categories=labels,
                    ordered=True,
                ),
                "Number of Explanations": counts.values,
            })

            st.bar_chart(
                chart_data,
                x=val,
                y="Number of Explanations",
                color = col,
            )
        else:
            st.write(
                f"All evaluated explanations have a value of "
                f"{values.iloc[0]:.2f}."
            )    

    # Allow inspection of the saved evaluation records.
    st.divider()
    st.subheader("Evaluation Results")
    st.caption(
        "View the evaluation results for each sample."
    )

    display_columns = [
        column
        for column in [
            "split",
            "example_id",
            "model",
            "flesch_kincaid_grade",
            "smog_index",
            "jargon_density",
            "word_count",
            "meets_target",
            "is_refusal",
        ]
        if column in results_df.columns
    ]

    st.dataframe(
        results_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # Show failed examples separately for debugging.
    failed = pd.DataFrame(records)
    failed = failed[failed["status"] == "failed"]

    if not failed.empty:
        with st.expander("View failed evaluations"):
            failure_columns = [
                column
                for column in [
                    "split",
                    "example_id",
                    "source_file",
                    "error",
                ]
                if column in failed.columns
            ]

            st.dataframe(
                failed[failure_columns],
                use_container_width=True,
                hide_index=True,
            )