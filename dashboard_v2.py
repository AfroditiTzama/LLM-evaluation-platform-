from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🧠",
    layout="wide",
)


DATA_DIR = Path(".")

FILES = {
    "logs": "all_model_logs_merged_fixed.csv",
    "metrics_model": "metrics_by_model.csv",
    "metrics_usecase": "metrics_by_model_usecase.csv",
    "failures_model": "failure_summary_by_model.csv",
    "failures_usecase": "failure_summary_by_model_usecase.csv",
    "cost_model": "cost_analysis_by_model.csv",
    "cost_usecase": "cost_analysis_by_model_usecase.csv",
    "judge_model": "judge_summary_by_model.csv",
    "judge_usecase": "judge_summary_by_model_usecase.csv",
    "final_overall": "final_recommendations_overall.csv",
    "final_usecase": "final_recommendations_by_usecase.csv",
}

COLORS = [
    "#38bdf8",
    "#f59e0b",
    "#fb7185",
    "#22c55e",
    "#a78bfa",
    "#f97316",
]


def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Missing file: {filename}")
        st.stop()
    return pd.read_csv(path)


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


def recommendation_card(title, model, score, extra):
    st.markdown(
        f"""
        <div class="recommend-card">
            <div class="recommend-title">{title}</div>
            <div class="recommend-model">{model}</div>
            <div class="recommend-score">Final score: {score}</div>
            <div class="recommend-extra">{extra}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def chart_card_start():
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)


def chart_card_end():
    st.markdown('</div>', unsafe_allow_html=True)


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
            radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.10), transparent 28%),
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
        font-size: 60px;
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: -2.4px;
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
        letter-spacing: -0.5px;
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
        font-size: 36px;
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

    .recommend-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.97), rgba(17,24,39,0.88));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 24px 25px;
        min-height: 176px;
        box-shadow: 0 20px 65px rgba(0,0,0,0.36);
    }

    .recommend-title {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 900;
    }

    .recommend-model {
        font-size: 28px;
        color: #ffffff;
        font-weight: 900;
        margin-top: 12px;
        letter-spacing: -0.8px;
    }

    .recommend-score {
        color: #38bdf8;
        font-weight: 900;
        margin-top: 8px;
    }

    .recommend-extra {
        color: #cbd5e1;
        margin-top: 8px;
        font-size: 14px;
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

    div[data-testid="stDataFrame"] {
        background: rgba(15,23,42,0.85);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15,23,42,0.76);
        border-radius: 14px;
        padding: 10px 18px;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56,189,248,0.22);
        color: white;
        border: 1px solid rgba(56,189,248,0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    logs = add_model_label(load_csv(FILES["logs"]))
    metrics_model = add_model_label(load_csv(FILES["metrics_model"]))
    metrics_usecase = add_model_label(load_csv(FILES["metrics_usecase"]))
    failures_model = add_model_label(load_csv(FILES["failures_model"]))
    failures_usecase = add_model_label(load_csv(FILES["failures_usecase"]))
    cost_model = add_model_label(load_csv(FILES["cost_model"]))
    cost_usecase = add_model_label(load_csv(FILES["cost_usecase"]))
    judge_model = add_model_label(load_csv(FILES["judge_model"]))
    judge_usecase = add_model_label(load_csv(FILES["judge_usecase"]))
    final_overall = add_model_label(load_csv(FILES["final_overall"]))
    final_usecase = add_model_label(load_csv(FILES["final_usecase"]))

    st.sidebar.title("Dashboard Filters")

    all_models = sorted(logs["model_label"].unique())
    all_usecases = sorted(logs["use_case"].unique())

    selected_models = st.sidebar.multiselect(
        "Models",
        all_models,
        default=all_models,
    )

    selected_usecases = st.sidebar.multiselect(
        "Use cases",
        all_usecases,
        default=all_usecases,
    )

    filtered_logs = logs[
        logs["model_label"].isin(selected_models)
        & logs["use_case"].isin(selected_usecases)
    ].copy()

    final_overall_view = final_overall[
        final_overall["model_label"].isin(selected_models)
    ].copy()

    final_usecase_view = final_usecase[
        final_usecase["model_label"].isin(selected_models)
        & final_usecase["use_case"].isin(selected_usecases)
    ].copy()

    metrics_model_view = metrics_model[
        metrics_model["model_label"].isin(selected_models)
    ].copy()

    metrics_usecase_view = metrics_usecase[
        metrics_usecase["model_label"].isin(selected_models)
        & metrics_usecase["use_case"].isin(selected_usecases)
    ].copy()

    cost_model_view = cost_model[
        cost_model["model_label"].isin(selected_models)
    ].copy()

    failures_model_view = failures_model[
        failures_model["model_label"].isin(selected_models)
    ].copy()

    failures_usecase_view = failures_usecase[
        failures_usecase["model_label"].isin(selected_models)
        & failures_usecase["use_case"].isin(selected_usecases)
    ].copy()

    judge_model_view = judge_model[
        judge_model["model_label"].isin(selected_models)
    ].copy()

    st.markdown(
        """
        <div class="hero-title">
            LLM Evaluation<br>
            Benchmark Dashboard
        </div>
        <div class="hero-subtitle">
            Final model comparison dashboard combining LLM-as-a-judge quality, latency,
            estimated cost, heuristic failures and possible truncation signals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_overall = final_overall.sort_values("final_rank").iloc[0]
    best_quality = final_overall.sort_values("avg_overall_quality", ascending=False).iloc[0]
    cheapest = final_overall.sort_values("estimated_cost_per_1000_prompts_usd").iloc[0]
    fastest = final_overall.sort_values("avg_latency").iloc[0]

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        kpi_card("Total Runs", f"{len(filtered_logs):,}", "All model responses")
    with k2:
        kpi_card("Models", f"{filtered_logs['model_label'].nunique()}", "Compared backends")
    with k3:
        kpi_card("Unique Prompts", f"{filtered_logs['prompt_id'].nunique()}", "Benchmark prompts")
    with k4:
        kpi_card("Best Balanced", best_overall["model_label"], f"Final score: {best_overall['final_score']:.4f}")

    section_title("Final Executive Results")

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        kpi_card(
            "Best Overall",
            best_overall["model_label"],
            f"Quality: {best_overall['avg_overall_quality']:.2f}/10",
        )
    with e2:
        kpi_card(
            "Best Quality",
            best_quality["model_label"],
            f"{best_quality['avg_overall_quality']:.2f}/10 judged score",
        )
    with e3:
        kpi_card(
            "Lowest Cost",
            cheapest["model_label"],
            f"${cheapest['estimated_cost_per_1000_prompts_usd']:.4f} / 1K prompts",
        )
    with e4:
        kpi_card(
            "Fastest",
            fastest["model_label"],
            f"{fastest['avg_latency']:.2f}s avg latency",
        )

    section_title("Recommended Model by Use Case")

    best_per_usecase = (
        final_usecase
        .sort_values("final_rank")
        .groupby("use_case")
        .head(1)
        .sort_values("use_case")
    )

    rec_cols = st.columns(3)

    for col, (_, row) in zip(rec_cols, best_per_usecase.iterrows()):
        with col:
            recommendation_card(
                row["use_case"].replace("_", " ").title(),
                row["model_label"],
                f"{row['final_score']:.4f}",
                f"Quality: {row['avg_overall_quality']:.2f}/10 | Cost: ${row['estimated_cost_per_1000_prompts_usd']:.4f}/1K",
            )

    st.markdown(
        """
        <div class="note-box">
        <b>Interpretation:</b> The final ranking is not based only on response quality.
        It combines LLM-as-a-judge quality with operational metrics. This is why the
        best quality model can differ from the best balanced backend model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_final, tab_quality, tab_cost_latency, tab_failures, tab_usecases, tab_raw = st.tabs(
        [
            "Overview",
            "Final Recommendations",
            "Quality Judge",
            "Cost & Latency",
            "Failures",
            "Use Cases",
            "Raw Tables",
        ]
    )

    with tab_overview:
        section_title("Overall Scoreboard")

        left, right = st.columns([1.15, 1])

        with left:
            chart_card_start()
            fig = px.bar(
                final_overall_view.sort_values("final_score", ascending=True),
                x="final_score",
                y="model_label",
                orientation="h",
                title="Final Balanced Score by Model",
                color="model_label",
                color_discrete_sequence=COLORS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        with right:
            chart_card_start()
            fig = px.pie(
                cost_model_view,
                names="model_label",
                values="total_cost_usd",
                hole=0.58,
                title="Cost Distribution Across Models",
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(
                textinfo="percent",
                textfont=dict(color="white", size=14),
                marker=dict(line=dict(color="rgba(255,255,255,0.12)", width=1)),
            )
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        c1, c2 = st.columns(2)

        with c1:
            chart_card_start()
            fig = px.scatter(
                final_overall_view,
                x="estimated_cost_per_1000_prompts_usd",
                y="avg_overall_quality",
                size="final_score",
                color="model_label",
                hover_name="model_label",
                title="Quality vs Cost",
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(marker=dict(opacity=0.9, line=dict(width=1, color="white")))
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        with c2:
            chart_card_start()
            fig = px.scatter(
                final_overall_view,
                x="avg_latency",
                y="failure_rate",
                size="final_score",
                color="model_label",
                hover_name="model_label",
                title="Latency vs Failure Rate",
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(marker=dict(opacity=0.9, line=dict(width=1, color="white")))
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

    with tab_final:
        section_title("Final Ranking Table")

        final_cols = [
            "final_rank",
            "model_label",
            "final_score",
            "avg_overall_quality",
            "avg_latency",
            "estimated_cost_per_1000_prompts_usd",
            "failure_rate",
            "possible_truncations",
        ]

        st.dataframe(
            final_overall_view[final_cols].sort_values("final_rank"),
            use_container_width=True,
        )

        section_title("Final Recommendation by Use Case")

        usecase_cols = [
            "use_case",
            "final_rank",
            "model_label",
            "final_score",
            "avg_overall_quality",
            "avg_latency",
            "estimated_cost_per_1000_prompts_usd",
            "failure_rate",
            "possible_truncations",
        ]

        st.dataframe(
            final_usecase_view[usecase_cols].sort_values(["use_case", "final_rank"]),
            use_container_width=True,
        )

        chart_card_start()
        fig = px.bar(
            final_usecase_view,
            x="use_case",
            y="final_score",
            color="model_label",
            barmode="group",
            title="Final Score by Use Case",
            color_discrete_sequence=COLORS,
        )
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        chart_card_end()

    with tab_quality:
        section_title("LLM-as-a-Judge Quality Results")

        q1, q2 = st.columns([1, 1])

        with q1:
            chart_card_start()
            fig = px.bar(
                judge_model_view.sort_values("avg_overall_quality", ascending=True),
                x="avg_overall_quality",
                y="model_label",
                orientation="h",
                title="Average Judge Quality Score",
                color="model_label",
                color_discrete_sequence=COLORS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        with q2:
            chart_card_start()
            radar_df = final_overall_view.sort_values("final_rank").copy()

            categories = [
                "quality_score",
                "latency_score_final",
                "cost_score_final",
                "failure_score_final",
                "truncation_score_final",
            ]

            fig = go.Figure()

            for _, row in radar_df.iterrows():
                values = [row[c] for c in categories]
                values.append(values[0])
                labels = [
                    "Quality",
                    "Latency",
                    "Cost",
                    "Failures",
                    "Truncation",
                ]
                labels.append(labels[0])

                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=labels,
                        fill="toself",
                        name=row["model_label"],
                    )
                )

            fig.update_layout(
                title="Normalized Final Score Components",
                polar=dict(
                    bgcolor="rgba(15,23,42,0.35)",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor="rgba(255,255,255,0.12)",
                        tickfont=dict(color="#e5e7eb"),
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255,255,255,0.12)",
                        tickfont=dict(color="#e5e7eb"),
                    ),
                ),
            )
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        st.dataframe(
            judge_model_view.sort_values("avg_overall_quality", ascending=False),
            use_container_width=True,
        )

    with tab_cost_latency:
        section_title("Cost and Latency Analysis")

        a, b = st.columns(2)

        with a:
            chart_card_start()
            fig = px.bar(
                final_overall_view.sort_values("estimated_cost_per_1000_prompts_usd"),
                x="estimated_cost_per_1000_prompts_usd",
                y="model_label",
                orientation="h",
                title="Estimated Cost per 1,000 Prompts",
                color="model_label",
                color_discrete_sequence=COLORS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        with b:
            chart_card_start()
            fig = px.line(
                metrics_usecase_view,
                x="use_case",
                y="avg_latency",
                color="model_label",
                markers=True,
                title="Average Latency by Use Case",
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=9))
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        st.dataframe(
            final_overall_view[
                [
                    "model_label",
                    "avg_latency",
                    "estimated_cost_per_1000_prompts_usd",
                    "avg_overall_quality",
                    "final_score",
                ]
            ].sort_values("estimated_cost_per_1000_prompts_usd"),
            use_container_width=True,
        )

    with tab_failures:
        section_title("Heuristic Failure Analysis")

        f1, f2 = st.columns(2)

        with f1:
            chart_card_start()
            fig = px.bar(
                final_overall_view.sort_values("failure_rate"),
                x="model_label",
                y="failure_rate",
                title="Failure Rate by Model",
                color="model_label",
                color_discrete_sequence=COLORS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        with f2:
            heatmap_data = failures_usecase_view.copy()
            heatmap_data["use_case_label"] = heatmap_data["use_case"].str.replace("_", " ").str.title()

            pivot = heatmap_data.pivot(
                index="model_label",
                columns="use_case_label",
                values="failure_rate",
            )

            chart_card_start()
            fig = px.imshow(
                pivot,
                text_auto=".2f",
                title="Failure Rate Heatmap",
                color_continuous_scale="YlOrRd",
            )
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        st.dataframe(
            failures_model_view.sort_values("failure_rate"),
            use_container_width=True,
        )

    with tab_usecases:
        section_title("Use Case Deep Dive")

        selected_usecase = st.selectbox(
            "Select use case",
            sorted(final_usecase_view["use_case"].unique()),
        )

        subset = final_usecase_view[final_usecase_view["use_case"] == selected_usecase].copy()

        u1, u2 = st.columns(2)

        with u1:
            chart_card_start()
            fig = px.bar(
                subset.sort_values("final_score"),
                x="final_score",
                y="model_label",
                orientation="h",
                title=f"Final Score — {selected_usecase}",
                color="model_label",
                color_discrete_sequence=COLORS,
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        with u2:
            chart_card_start()
            fig = px.scatter(
                subset,
                x="estimated_cost_per_1000_prompts_usd",
                y="avg_overall_quality",
                size="final_score",
                color="model_label",
                hover_name="model_label",
                title=f"Quality vs Cost — {selected_usecase}",
                color_discrete_sequence=COLORS,
            )
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

        st.dataframe(
            subset[
                [
                    "final_rank",
                    "model_label",
                    "final_score",
                    "avg_overall_quality",
                    "avg_latency",
                    "estimated_cost_per_1000_prompts_usd",
                    "failure_rate",
                ]
            ].sort_values("final_rank"),
            use_container_width=True,
        )

    with tab_raw:
        section_title("Raw Tables")

        st.subheader("Final overall recommendations")
        st.dataframe(final_overall, use_container_width=True)

        st.subheader("Final use-case recommendations")
        st.dataframe(final_usecase, use_container_width=True)

        st.subheader("Filtered logs")
        st.dataframe(filtered_logs, use_container_width=True)

        st.download_button(
            "Download final overall recommendations",
            data=final_overall.to_csv(index=False).encode("utf-8-sig"),
            file_name="final_recommendations_overall.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
