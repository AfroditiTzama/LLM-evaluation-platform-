from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, inspect, text


st.set_page_config(
    page_title="DB LLM Evaluation Platform",
    page_icon="🧠",
    layout="wide",
)


DEFAULT_DB_URL = "sqlite:///llm_eval.db"

COLORS = [
    "#38bdf8",
    "#f59e0b",
    "#fb7185",
    "#22c55e",
    "#a78bfa",
    "#f97316",
]

CHART_LABELS = {
    "model_label": "Model",
    "avg_latency": "Average Latency (seconds)",
    "estimated_cost_per_1000_prompts_usd": "Estimated Cost / 1,000 Prompts (USD)",
    "final_score": "Final Score",
    "preliminary_score": "Preliminary Score",
    "avg_overall_quality": "Average Quality Score",
    "avg_hallucination_safety": "Hallucination Safety Score",
    "json_valid_rate_display": "JSON Validity Rate",
    "failure_rate": "Failure Rate",
}


def get_database_url():
    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    return db_url


@st.cache_resource
def get_engine():
    return create_engine(get_database_url())


def table_exists(table_name):
    engine = get_engine()
    inspector = inspect(engine)
    return inspector.has_table(table_name)


def read_table(table_name, run_id=None):
    engine = get_engine()

    if not table_exists(table_name):
        return pd.DataFrame()

    if run_id is None:
        query = text(f"SELECT * FROM {table_name}")
        return pd.read_sql_query(query, engine)

    query = text(f"SELECT * FROM {table_name} WHERE run_id = :run_id")
    return pd.read_sql_query(query, engine, params={"run_id": run_id})


def short_model_name(model):
    mapping = {
        "minimax/minimax-m2.5": "MiniMax M2.5",
        "minimax/minimax-m2.7": "MiniMax M2.7",
        "minimax/minimax-m3": "MiniMax M3",
        "qwen/qwen3.6-plus": "Qwen 3.6 Plus",
        "qwen/qwen3.7-plus": "Qwen 3.7 Plus",
        "qwen/qwen3.7-max": "Qwen 3.7 Max",
    }
    return mapping.get(model, model)


def add_model_label(df):
    df = df.copy()

    if "model" in df.columns:
        df["model_label"] = df["model"].apply(short_model_name)

    return df


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#f8fafc", family="Inter, Arial"),
        title=dict(font=dict(size=20, color="#f8fafc"), x=0.02),
        margin=dict(l=25, r=25, t=55, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        tickfont=dict(color="#e5e7eb"),
        title_font=dict(color="#e5e7eb"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        tickfont=dict(color="#e5e7eb"),
        title_font=dict(color="#e5e7eb"),
    )

    return fig


def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_new_evaluation_form():
    st.markdown('<div class="section-title">New Evaluation</div>', unsafe_allow_html=True)

    with st.expander("Run a new prompt or CSV evaluation", expanded=False):
        mode = st.radio(
            "Input mode",
            ["Single prompt", "CSV file"],
            horizontal=True,
        )

        with st.form("new_evaluation_form_db"):
            if mode == "Single prompt":
                prompt = st.text_area(
                    "Prompt",
                    height=130,
                    placeholder="Write the prompt you want to evaluate across all models...",
                )

                use_case = st.selectbox(
                    "Use case",
                    [
                        "custom",
                        "structured_json",
                        "customer_support",
                        "document_understanding",
                        "website_generation",
                        "reasoning",
                        "code_generation",
                    ],
                    index=0,
                )

                difficulty = st.selectbox(
                    "Difficulty",
                    ["custom", "easy", "medium", "hard"],
                    index=0,
                )

                expected_format = st.selectbox(
                    "Expected format",
                    ["text", "json"],
                    index=0,
                )

                reference_text = st.text_area(
                    "Reference text / document context",
                    height=100,
                    placeholder="Optional. Add menu, policy, document text, or reference material here.",
                )

                uploaded_file = None

            else:
                uploaded_file = st.file_uploader(
                    "Upload prompts CSV",
                    type=["csv"],
                )

                st.caption(
                    "CSV columns: question or prompt. Optional columns: prompt_id, use_case, difficulty, expected_format, reference_text, max_tokens."
                )

                prompt = ""
                use_case = "custom"
                difficulty = "custom"
                expected_format = "text"
                reference_text = ""

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
            )

            skip_judge = st.checkbox(
                "Skip quality judge",
                value=False,
            )

            submitted = st.form_submit_button("Run Evaluation")

        if submitted:
            command = [
                sys.executable,
                "run_weekly_pipeline.py",
                "--temperature",
                str(temperature),
            ]

            if skip_judge:
                command.append("--skip-judge")

            if mode == "Single prompt":
                if not prompt.strip():
                    st.error("Please enter a prompt first.")
                    return

                command.extend(["--prompt", prompt])
                command.extend(["--use-case", use_case])
                command.extend(["--difficulty", difficulty])
                command.extend(["--expected-format", expected_format])

                if reference_text.strip():
                    command.extend(["--reference-text", reference_text])

            else:
                if uploaded_file is None:
                    st.error("Please upload a CSV file first.")
                    return

                upload_dir = Path("uploaded_prompts")
                upload_dir.mkdir(exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                upload_path = upload_dir / f"uploaded_prompts_{timestamp}.csv"
                upload_path.write_bytes(uploaded_file.getbuffer())

                command.extend(["--input", str(upload_path)])

            st.info("Running evaluation. This may take a few minutes.")

            with st.spinner("Running all models, judge and database sync..."):
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                )

            if result.returncode == 0:
                st.success("Evaluation completed and saved to database.")
                if result.stdout:
                    st.code(result.stdout[-4000:], language="text")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Evaluation failed.")
                if result.stdout:
                    st.subheader("Output")
                    st.code(result.stdout[-4000:], language="text")
                if result.stderr:
                    st.subheader("Error")
                    st.code(result.stderr[-4000:], language="text")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', Arial, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 32%),
            radial-gradient(circle at top right, rgba(245, 158, 11, 0.13), transparent 30%),
            linear-gradient(135deg, #030712 0%, #0f172a 48%, #111827 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero-title {
        font-size: 56px;
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: -2.2px;
        color: #ffffff;
        margin-bottom: 0.4rem;
        text-shadow: 0 0 38px rgba(56,189,248,0.20);
    }

    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
        max-width: 900px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 900;
        color: #ffffff;
        margin-top: 1.4rem;
        margin-bottom: 0.75rem;
    }

    .kpi-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(30,41,59,0.78));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 24px;
        padding: 22px 24px;
        min-height: 138px;
        box-shadow: 0 20px 65px rgba(0,0,0,0.35);
    }

    .kpi-title {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    .kpi-value {
        font-size: 34px;
        color: #ffffff;
        font-weight: 900;
        margin-top: 11px;
        margin-bottom: 5px;
        letter-spacing: -1.1px;
    }

    .kpi-subtitle {
        font-size: 14px;
        color: #cbd5e1;
    }

    .chart-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.76));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 18px 18px 8px 18px;
        box-shadow: 0 20px 65px rgba(0,0,0,0.28);
        margin-bottom: 1rem;
    }

    .note-box {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px 20px;
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown(
        """
        <div class="hero-title">
            Database-backed<br>
            LLM Evaluation Platform
        </div>
        <div class="hero-subtitle">
            Persistent dashboard for custom prompt evaluations, weekly runs, JSON validity,
            quality judging, hallucination safety, cost, latency and final model recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_new_evaluation_form()

    st.sidebar.write("Database:")
    st.sidebar.code(get_database_url())

    runs = read_table("runs")

    if runs.empty:
        st.warning("No runs found in database yet. Run a new evaluation first.")
        return

    runs = runs.sort_values("run_id", ascending=False)

    run_labels = runs["run_id"].astype(str).tolist()

    selected_run_id = st.sidebar.selectbox(
        "Select run",
        run_labels,
        index=0,
    )

    selected_run_meta = runs[runs["run_id"] == selected_run_id].iloc[0]

    logs = add_model_label(read_table("model_outputs", selected_run_id))
    summary = add_model_label(read_table("model_summary", selected_run_id))
    prelim = add_model_label(read_table("preliminary_recommendations", selected_run_id))
    judge = add_model_label(read_table("judge_summary_model", selected_run_id))
    judge_scores = add_model_label(read_table("judge_scores", selected_run_id))
    final = add_model_label(read_table("final_recommendations", selected_run_id))
    prompts = read_table("eval_prompts", selected_run_id)

    if logs.empty:
        st.error("Selected run has no model outputs.")
        return

    if not final.empty:
        best = final.sort_values("final_rank").iloc[0]
        best_model = best["model_label"]
        best_score = f"{best['final_score']:.4f}"
    elif not prelim.empty:
        best = prelim.sort_values("preliminary_rank").iloc[0]
        best_model = best["model_label"]
        best_score = f"{best['preliminary_score']:.4f}"
    else:
        best_model = "N/A"
        best_score = "N/A"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Run ID", selected_run_id, "Stored in database")
    with c2:
        kpi_card("Prompts", int(selected_run_meta["prompt_count"]), "Unique prompts")
    with c3:
        kpi_card("Models", int(selected_run_meta["model_count"]), "Compared backends")
    with c4:
        kpi_card("Best Model", best_model, f"Score: {best_score}")

    st.markdown('<div class="section-title">Run Results</div>', unsafe_allow_html=True)

    if not final.empty:
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            fig = px.bar(
                final.sort_values("final_score", ascending=True),
                x="final_score",
                y="model_label",
                orientation="h",
                color="model_label",
                title="Final Score by Model",
                color_discrete_sequence=COLORS,
                labels=CHART_LABELS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            fig = px.scatter(
                final,
                x="estimated_cost_per_1000_prompts_usd",
                y="avg_overall_quality",
                size="final_score",
                color="model_label",
                hover_name="model_label",
                title="Quality vs Cost",
                color_discrete_sequence=COLORS,
                labels=CHART_LABELS,
            )
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    elif not prelim.empty:
        st.info("Final judge results not found. Showing preliminary recommendations.")

        fig = px.bar(
            prelim.sort_values("preliminary_score", ascending=True),
            x="preliminary_score",
            y="model_label",
            orientation="h",
            color="model_label",
            title="Preliminary Score by Model",
            color_discrete_sequence=COLORS,
            labels=CHART_LABELS,
        )
        fig.update_layout(showlegend=False)
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Operational Metrics</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)

    with m1:
        if not summary.empty:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            fig = px.bar(
                summary.sort_values("avg_latency"),
                x="model_label",
                y="avg_latency",
                color="model_label",
                title="Average Latency",
                color_discrete_sequence=COLORS,
                labels=CHART_LABELS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        if not summary.empty:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            fig = px.bar(
                summary.sort_values("estimated_cost_per_1000_prompts_usd"),
                x="model_label",
                y="estimated_cost_per_1000_prompts_usd",
                color="model_label",
                title="Estimated Cost per 1,000 Prompts",
                color_discrete_sequence=COLORS,
                labels=CHART_LABELS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    if not summary.empty and "json_valid_rate" in summary.columns:
        st.markdown('<div class="section-title">JSON Validity</div>', unsafe_allow_html=True)

        json_df = summary.copy()
        json_df["json_valid_rate_display"] = json_df["json_valid_rate"].fillna(1.0)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        fig = px.bar(
            json_df,
            x="model_label",
            y="json_valid_rate_display",
            color="model_label",
            title="JSON Validity Rate",
            color_discrete_sequence=COLORS,
            labels=CHART_LABELS,
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1.05])
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if not judge.empty:
        st.markdown('<div class="section-title">Quality Judge</div>', unsafe_allow_html=True)

        q1, q2 = st.columns(2)

        with q1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            fig = px.bar(
                judge.sort_values("avg_overall_quality"),
                x="avg_overall_quality",
                y="model_label",
                orientation="h",
                color="model_label",
                title="Judge Overall Quality",
                color_discrete_sequence=COLORS,
                labels=CHART_LABELS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with q2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            fig = px.bar(
                judge.sort_values("avg_hallucination_safety"),
                x="avg_hallucination_safety",
                y="model_label",
                orientation="h",
                color="model_label",
                title="Hallucination Safety",
                color_discrete_sequence=COLORS,
                labels=CHART_LABELS,
            )
            fig.update_layout(showlegend=False, xaxis_range=[0, 5.1])
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Final Recommendations",
            "Prompts",
            "Model Answers",
            "Judge Scores",
            "Database Tables",
        ]
    )

    with tab1:
        if not final.empty:
            st.dataframe(final, use_container_width=True)
        elif not prelim.empty:
            st.dataframe(prelim, use_container_width=True)
        else:
            st.warning("No recommendation data found.")

    with tab2:
        st.dataframe(prompts, use_container_width=True)

    with tab3:
        st.dataframe(logs, use_container_width=True)

    with tab4:
        if not judge_scores.empty:
            st.dataframe(judge_scores, use_container_width=True)
        else:
            st.info("No judge scores found for this run.")

    with tab5:
        st.write("Tables loaded from database:")
        for table in [
            "runs",
            "eval_prompts",
            "model_outputs",
            "model_summary",
            "preliminary_recommendations",
            "judge_scores",
            "judge_summary_model",
            "final_recommendations",
        ]:
            if table_exists(table):
                df = read_table(table, selected_run_id) if table != "runs" else read_table(table)
                st.write(f"- `{table}`: {len(df)} rows")
            else:
                st.write(f"- `{table}`: missing")


if __name__ == "__main__":
    main()
