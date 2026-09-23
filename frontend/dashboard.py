"""Interactive dashboard for the autonomous synthetic-data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.agents.analyzer import DatasetAnalyzer
from app.pipeline.orchestrator import PipelineConfig, run_pipeline
from app.utils.data_utils import (
    create_sample_churn_dataset,
    create_unlabeled_customer_dataset,
    validate_dataframe,
)
from app.utils.logging_config import setup_logging

setup_logging()
st.set_page_config(page_title="Synthetic-AI Platform", layout="wide")
st.title("Autonomous Multi-Agent Synthetic Data Platform")
st.caption(
    "Tabular CSV prototype: analyze, generate, validate, benchmark, and retry."
)

SAMPLE_PATH = ROOT / "datasets" / "sample" / "labeled_customer.csv"
UNLABELED_SAMPLE_PATH = ROOT / "datasets" / "sample" / "unlabeled_customer.csv"


def issue_card(label: str, present: bool, ok_text: str) -> None:
    if present:
        st.error(f"Warning: {label}")
    else:
        st.success(f"OK: {ok_text}")


with st.sidebar:
    st.header("Run settings")
    sensitive_column = st.text_input("Sensitive/group column (optional)", value="")
    max_iterations = st.slider("Max optimization iterations", 1, 3, 3)
    preferred = st.selectbox(
        "Preferred generator",
        ["auto", "gaussian_copula", "ctgan", "tvae"],
    )
    enable_llm = st.checkbox("Gemini explanations (needs GEMINI_API_KEY)", value=False)
    st.markdown(
        "Deep generators (CTGAN/TVAE) are slower. "
        "Use GaussianCopula for a quick demo."
    )

uploaded = st.file_uploader("1. Dataset upload", type=["csv"])
sample_col_a, sample_col_b = st.columns(2)
use_sample = sample_col_a.button("Load labeled sample")
use_unlabeled_sample = sample_col_b.button("Load unlabeled sample")

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.filename = None

if use_sample:
    if not SAMPLE_PATH.exists():
        create_sample_churn_dataset(SAMPLE_PATH, n_rows=500, seed=42)
    st.session_state.df = pd.read_csv(SAMPLE_PATH)
    st.session_state.filename = SAMPLE_PATH.name

if use_unlabeled_sample:
    if not UNLABELED_SAMPLE_PATH.exists():
        create_unlabeled_customer_dataset(UNLABELED_SAMPLE_PATH, n_rows=500, seed=42)
    st.session_state.df = pd.read_csv(UNLABELED_SAMPLE_PATH)
    st.session_state.filename = UNLABELED_SAMPLE_PATH.name

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded)
    st.session_state.filename = uploaded.name

df = st.session_state.df
if df is None:
    st.info("Upload a CSV or load the sample dataset to begin.")
    st.stop()

try:
    df = validate_dataframe(df, min_rows=20)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(10))

target_options = ["None", "Auto Detect", *list(df.columns)]
default_target_index = target_options.index("target") if "target" in df.columns else 0
target_selection = st.selectbox(
    "Target Column (Optional)",
    target_options,
    index=default_target_index,
)
if target_selection == "None":
    target = None
    auto_detect_target = False
elif target_selection == "Auto Detect":
    target = None
    auto_detect_target = True
else:
    target = target_selection
    auto_detect_target = False

st.subheader("2. Dataset overview")
analyzer = DatasetAnalyzer()
try:
    analysis = analyzer.analyze(
        df,
        target_column=target,
        auto_detect_target=auto_detect_target,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", analysis.rows)
c2.metric("Columns", analysis.columns)
c3.metric("Missing %", f"{analysis.missing_percentage:.2f}")
c4.metric("Duplicate %", f"{analysis.duplicate_percentage:.2f}")
st.write("Numerical columns:", ", ".join(analysis.numerical_columns) or "-")
st.write("Categorical columns:", ", ".join(analysis.categorical_columns) or "-")
mode_label = "SUPERVISED / LABELED" if analysis.target_column else "UNSUPERVISED / UNLABELED"
st.write("Evaluation Mode:", mode_label)
st.write("Target:", analysis.target_column or "None")
if analysis.target_detection_reason:
    st.caption(analysis.target_detection_reason)
if analysis.class_distribution:
    dist_df = pd.DataFrame(
        {
            "class": list(analysis.class_distribution.keys()),
            "share": list(analysis.class_distribution.values()),
        }
    )
    st.plotly_chart(
        px.bar(dist_df, x="class", y="share", title="Target class distribution"),
        width="stretch",
    )

st.subheader("3. Detected problems")
cols = st.columns(3)
with cols[0]:
    issue_card(
        "Class imbalance",
        "severe_class_imbalance" in analysis.issues or "class_imbalance" in analysis.issues,
        "Class balance looks reasonable",
    )
with cols[1]:
    issue_card(
        "Missing values",
        "missing_values" in analysis.issues or "high_missing_values" in analysis.issues,
        "No major missing values",
    )
with cols[2]:
    issue_card(
        "Duplicate rows",
        "duplicate_rows" in analysis.issues,
        "No major duplicates",
    )
if analysis.issue_details:
    st.json(analysis.issue_details)

run_clicked = st.button("Run autonomous pipeline", type="primary")
if run_clicked:
    with st.spinner("Agents are analyzing, generating, validating, and benchmarking..."):
        output = run_pipeline(
            df,
            target_column=analysis.target_column,
            config=PipelineConfig(
                max_iterations=max_iterations,
                enable_llm=enable_llm,
                auto_detect_target=auto_detect_target,
                preferred_generator=None if preferred == "auto" else preferred,
                sensitive_column=sensitive_column.strip() or None,
                dataset_filename=st.session_state.filename,
            ),
        )
    st.session_state.output = output

output = st.session_state.get("output")
if output is None:
    st.info("Run the pipeline to see agent activity, synthetic data, and benchmarks.")
    st.stop()

result = output.result
synthetic = output.synthetic_df

st.subheader("4. Agent activity")
for log in result.agent_logs:
    st.markdown(f"**{log.agent}** - {log.message}")
    st.markdown("then")
st.markdown("**Pipeline** - finished")

if result.llm_summaries:
    with st.expander("Gemini / fallback explanations"):
        for key, text in result.llm_summaries.items():
            st.markdown(f"**{key}**")
            st.write(text)

st.subheader("5. Planner decision")
if result.iterations:
    first_plan = result.iterations[0].plan
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Original rows", first_plan.original_rows or result.dataset_analysis.rows)
    p2.metric("Generator", first_plan.generator)
    p3.metric("Generation mode", first_plan.generation_mode)
    p4.metric("Synthetic rows planned", first_plan.num_samples)
    if first_plan.target_column:
        t1, t2, t3 = st.columns(3)
        t1.metric("Target", first_plan.target_column)
        t2.metric("Current minority", first_plan.current_target_count or 0)
        t3.metric("Desired minority", first_plan.desired_target_count or 0)
        st.write("Target class:", first_plan.target_class)
    st.caption(first_plan.sample_count_reason or first_plan.reason)
    with st.expander("Sample-count details"):
        st.json(first_plan.sample_count_details)
else:
    st.info(result.final_evaluation.get("reason", "Synthetic generation was not required."))

st.subheader("6. Synthetic data")
if synthetic is None:
    st.warning("No synthetic dataset was produced.")
else:
    st.metric("Generated rows (best iteration)", len(synthetic))
    st.dataframe(synthetic.head(20))
    st.download_button(
        "Download synthetic CSV",
        data=synthetic.to_csv(index=False),
        file_name=f"{result.run_id}_synthetic.csv",
        mime="text/csv",
    )

st.subheader("7. Validation")
if result.iterations:
    best = next(
        (it for it in result.iterations if it.iteration == result.best_iteration),
        result.iterations[-1],
    )
    v = best.validation
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Fidelity", f"{v.fidelity_score:.3f}")
    m2.metric("Distribution", f"{v.distribution_score:.3f}")
    m3.metric("Correlation", f"{v.correlation_score:.3f}")
    m4.metric("Diversity", f"{v.diversity_score:.3f}")
    m5.metric("Privacy (basic)", f"{v.privacy_score:.3f}")
    st.caption("Privacy values are basic risk indicators, not a privacy guarantee.")
    with st.expander("How metrics are calculated"):
        st.json(v.metric_notes)

    if synthetic is not None:
        numeric_cols = analysis.numerical_columns[:3]
        for col in numeric_cols:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[col], name="original", opacity=0.6))
            fig.add_trace(go.Histogram(x=synthetic[col], name="synthetic", opacity=0.6))
            fig.update_layout(barmode="overlay", title=f"{col}: original vs synthetic")
            st.plotly_chart(fig, width="stretch")
        if analysis.target_column and analysis.target_column in synthetic.columns:
            cmp = pd.concat(
                [
                    df[analysis.target_column].astype(str).value_counts(normalize=True).rename("original"),
                    synthetic[analysis.target_column]
                    .astype(str)
                    .value_counts(normalize=True)
                    .rename("synthetic"),
                ],
                axis=1,
            ).fillna(0)
            cmp_df = cmp.rename_axis("class").reset_index()
            st.plotly_chart(
                px.bar(
                    cmp_df,
                    x="class",
                    y=["original", "synthetic"],
                    barmode="group",
                    title="Class distribution",
                ),
                width="stretch",
            )
        num_cols = [c for c in analysis.numerical_columns if c in df.columns]
        if len(num_cols) >= 2:
            st.plotly_chart(
                px.imshow(df[num_cols].corr(numeric_only=True), title="Original correlations"),
                width="stretch",
            )
            st.plotly_chart(
                px.imshow(synthetic[num_cols].corr(numeric_only=True), title="Synthetic correlations"),
                width="stretch",
            )

st.subheader("8. Benchmark")
if result.baseline:
    last_bench = None
    if result.best_iteration:
        rec = next(it for it in result.iterations if it.iteration == result.best_iteration)
        last_bench = rec.benchmark
    if last_bench:
        b, a = last_bench.baseline, last_bench.augmented
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"Baseline {b.primary_metric}", f"{b.primary_value:.4f}")
        col_b.metric(f"Augmented {a.primary_metric}", f"{a.primary_value:.4f}")
        delta = last_bench.improvement.get("primary_value")
        if delta is not None and b.primary_value:
            pct = 100.0 * delta / abs(b.primary_value)
            col_c.metric("Improvement", f"{pct:+.2f}%")
        st.json(
            {
                "baseline": b.model_dump(),
                "augmented": a.model_dump(),
                "improved": last_bench.improved,
            }
        )
    else:
        st.json(result.baseline.model_dump())
else:
    st.write("N/A - No target column selected.")
    best_utility = None
    if result.best_iteration:
        rec = next(it for it in result.iterations if it.iteration == result.best_iteration)
        best_utility = rec.unsupervised_utility
    if best_utility:
        st.subheader("Unsupervised / Statistical Utility")
        u1, u2, u3, u4, u5 = st.columns(5)
        u1.metric("Overall", f"{best_utility.overall_score:.3f}")
        u2.metric("Distribution", f"{best_utility.distribution_score:.3f}")
        u3.metric("Correlation", f"{best_utility.correlation_score:.3f}")
        u4.metric("Diversity", f"{best_utility.diversity_score:.3f}")
        u5.metric("Structure", f"{best_utility.structural_score:.3f}")
        with st.expander("Unsupervised utility notes"):
            st.json(best_utility.metric_notes)

st.subheader("9. Optimization history")
for rec in result.iterations:
    bench = rec.benchmark
    utility = rec.unsupervised_utility
    if bench:
        primary = bench.augmented.primary_value
        status = "ACCEPTED" if bench.improved else "NO ML IMPROVEMENT"
    elif utility:
        primary = utility.overall_score
        status = "ACCEPTED" if utility.passed and rec.validation.passed else "QUALITY BELOW THRESHOLD"
    else:
        primary = None
        status = "STOPPED"
    st.markdown(
        f"**Iteration {rec.iteration}** - `{rec.plan.generator}` - "
        f"requested={rec.samples_requested} - generated={rec.samples_generated} - "
        f"cumulative={rec.cumulative_synthetic_rows} - "
        f"primary={primary if primary is None else round(primary, 4)} - **{status}**"
    )
    st.caption(f"{rec.decision}: {rec.plan.reason}")

st.subheader("Final decision")
if result.final_decision == "completed_improved":
    st.success("Optimization completed. A synthetic-data attempt improved the selected utility objective and was accepted.")
elif result.final_decision == "completed_no_improvement" and result.evaluation_mode == "labeled":
    st.warning(
        "Optimization completed. Synthetic data was generated and validated, but downstream ML utility "
        "did not improve within the configured iteration limit."
    )
elif result.final_decision == "completed_no_improvement":
    st.warning(
        "Optimization completed. Synthetic data was generated, but the quality/utility threshold was not "
        "improved within the configured iteration limit."
    )
elif result.final_decision == "skipped_generation_not_needed":
    st.info("Generation was skipped because the planner did not find a strong reason for augmentation.")
else:
    st.error(result.final_decision)

if result.iterations:
    best_rec = next(
        (it for it in result.iterations if it.iteration == result.best_iteration),
        result.iterations[-1],
    )
    st.subheader("Best attempt")
    if best_rec.benchmark:
        b = best_rec.benchmark.baseline
        a = best_rec.benchmark.augmented
        delta = best_rec.benchmark.improvement.get("primary_value")
        ba, bb, bc, bd = st.columns(4)
        ba.metric("Generator", best_rec.plan.generator)
        bb.metric(f"Baseline {b.primary_metric}", f"{b.primary_value:.4f}")
        bc.metric(f"Best {a.primary_metric}", f"{a.primary_value:.4f}")
        bd.metric("Change", "N/A" if delta is None else f"{delta:+.4f}")
        if delta is not None and abs(delta) < 1e-9:
            st.caption("Best attempt matched the baseline: no degradation, but no measurable ML gain.")
        elif delta is not None and delta < 0:
            st.caption("Best attempt underperformed the baseline, so it was not accepted as an ML improvement.")
        elif delta is not None:
            st.caption("Best attempt improved the supervised benchmark.")
    elif best_rec.unsupervised_utility:
        u = best_rec.unsupervised_utility
        ba, bb, bc, bd = st.columns(4)
        ba.metric("Generator", best_rec.plan.generator)
        bb.metric("Utility", f"{u.overall_score:.4f}")
        bc.metric("Validation", f"{best_rec.validation.overall_score:.4f}")
        bd.metric("Generated rows", best_rec.samples_generated)
        st.caption("Best attempt is selected by the highest statistical/unsupervised utility score.")

if result.provenance:
    with st.expander("Prototype provenance fingerprint"):
        st.json(result.provenance.model_dump())
if result.fairness:
    with st.expander("Simple group comparison (not a full fairness audit)"):
        st.json(result.fairness.model_dump())
