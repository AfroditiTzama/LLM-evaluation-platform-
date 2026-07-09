import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="LLM Evaluation Platform",
    layout="wide",
)


DATA_DIR = Path(".")


FILES = {
    "logs": "all_model_logs_merged_fixed.csv",
    "metrics_model": "metrics_by_model.csv",
    "metrics_usecase": "metrics_by_model_usecase.csv",
    "metrics_difficulty": "metrics_by_model_difficulty.csv",
    "failures_model": "failure_summary_by_model.csv",
    "failures_usecase": "failure_summary_by_model_usecase.csv",
    "cost_model": "cost_analysis_by_model.csv",
    "cost_usecase": "cost_analysis_by_model_usecase.csv",
    "recommendations_usecase": "recommendations_by_usecase.csv",
    "recommendations_overall": "recommendations_overall.csv",
}


def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        st.error(f"Missing file: {filename}")
        st.stop()

    return pd.read_csv(path)


def safe_round(value, digits=3):
    try:
        return round(float(value), digits)
    except Exception:
        return value


def normalize_lower_is_better(series):
    series = pd.to_numeric(series, errors="coerce")
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([0.0] * len(series), index=series.index)

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1 - ((series - min_val) / (max_val - min_val))


def simple_bar_chart(df, x_col, y_col, title, xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = df.sort_values(y_col)
    ax.bar(plot_df[x_col], plot_df[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)


def grouped_bar_chart(df, group_col, x_col, y_col, title, ylabel=""):
    pivot = df.pivot(index=x_col, columns=group_col, values=y_col)

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)


def scatter_chart(df, x_col, y_col, label_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df[x_col], df[y_col])

    for _, row in df.iterrows():
        ax.annotate(
            row[label_col],
            (row[x_col], row[y_col]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    st.pyplot(fig)


def heatmap_chart(df, index_col, columns_col, value_col, title):
    pivot = df.pivot(index=index_col, columns=columns_col, values=value_col)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)


def radar_chart(model_row, title):
    categories = [
        "Latency",
        "Cost",
        "Failure",
        "Token Efficiency",
        "Truncation",
    ]

    values = [
        model_row["latency_score"],
        model_row["cost_score"],
        model_row["failure_score"],
        model_row["token_score"],
        model_row["truncation_score"],
    ]

    values += values[:1]
    angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title(title)

    fig.tight_layout()
    st.pyplot(fig)


def prepare_radar_data(metrics_model, failures_model, cost_model):
    df = metrics_model.merge(
        failures_model[["model", "failure_rate", "possible_truncations"]],
        on="model",
        how="left",
    )

    df = df.merge(
        cost_model[["model", "avg_cost_per_prompt_usd", "estimated_cost_per_1000_prompts_usd"]],
        on="model",
        how="left",
    )

    df["latency_score"] = normalize_lower_is_better(df["avg_latency"])
    df["cost_score"] = normalize_lower_is_better(df["avg_cost_per_prompt_usd"])
    df["failure_score"] = normalize_lower_is_better(df["failure_rate"])
    df["token_score"] = normalize_lower_is_better(df["avg_total_tokens"])
    df["truncation_score"] = normalize_lower_is_better(df["possible_truncations"])

    return df


def main():
    st.title("LLM Evaluation Platform")
    st.caption("Benchmark dashboard for latency, tokens, cost, failures, and preliminary model recommendations.")

    logs = load_csv(FILES["logs"])
    metrics_model = load_csv(FILES["metrics_model"])
    metrics_usecase = load_csv(FILES["metrics_usecase"])
    metrics_difficulty = load_csv(FILES["metrics_difficulty"])
    failures_model = load_csv(FILES["failures_model"])
    failures_usecase = load_csv(FILES["failures_usecase"])

    # Avoid duplicate column names after merges.
    # The failure CSVs are the source of truth for possible_truncations.
    metrics_model = metrics_model.drop(columns=["possible_truncations"], errors="ignore")
    metrics_usecase = metrics_usecase.drop(columns=["possible_truncations"], errors="ignore")
    metrics_difficulty = metrics_difficulty.drop(columns=["possible_truncations"], errors="ignore")
    cost_model = load_csv(FILES["cost_model"])
    cost_usecase = load_csv(FILES["cost_usecase"])
    recommendations_usecase = load_csv(FILES["recommendations_usecase"])
    recommendations_overall = load_csv(FILES["recommendations_overall"])

    models = sorted(logs["model"].dropna().unique())
    use_cases = sorted(logs["use_case"].dropna().unique())

    st.sidebar.header("Filters")
    selected_models = st.sidebar.multiselect(
        "Models",
        models,
        default=models,
    )

    selected_use_cases = st.sidebar.multiselect(
        "Use cases",
        use_cases,
        default=use_cases,
    )

    filtered_logs = logs[
        logs["model"].isin(selected_models)
        & logs["use_case"].isin(selected_use_cases)
    ].copy()

    tab_overview, tab_recommendations, tab_latency, tab_tokens_cost, tab_failures, tab_usecases, tab_scatter, tab_radar, tab_raw = st.tabs(
        [
            "Overview",
            "Recommendations",
            "Latency",
            "Tokens & Cost",
            "Failures",
            "Use Cases",
            "Scatter / Pareto",
            "Radar",
            "Raw Data",
        ]
    )

    with tab_overview:
        st.header("Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", len(filtered_logs))
        col2.metric("Models", filtered_logs["model"].nunique())
        col3.metric("Unique prompts", filtered_logs["prompt_id"].nunique())
        col4.metric("Use cases", filtered_logs["use_case"].nunique())

        st.subheader("Model-level metrics")

        overview = metrics_model.merge(
            cost_model[["model", "total_cost_usd", "estimated_cost_per_1000_prompts_usd"]],
            on="model",
            how="left",
        ).merge(
            failures_model[["model", "failure_rate", "possible_truncations"]],
            on="model",
            how="left",
        )

        display_cols = [
            "model",
            "avg_latency",
            "median_latency",
            "p95_latency",
            "avg_total_tokens",
            "total_tokens",
            "total_cost_usd",
            "estimated_cost_per_1000_prompts_usd",
            "failure_rate",
            "possible_truncations",
        ]

        st.dataframe(
            overview[display_cols].sort_values("avg_latency"),
            use_container_width=True,
        )

        st.subheader("Quick interpretation")

        fastest = overview.sort_values("avg_latency").iloc[0]
        cheapest = overview.sort_values("estimated_cost_per_1000_prompts_usd").iloc[0]
        lowest_failure = overview.sort_values("failure_rate").iloc[0]

        st.write(f"**Fastest model:** {fastest['model']} — {safe_round(fastest['avg_latency'])}s average latency.")
        st.write(f"**Lowest cost model:** {cheapest['model']} — ${safe_round(cheapest['estimated_cost_per_1000_prompts_usd'], 4)} per 1,000 prompts.")
        st.write(f"**Lowest heuristic failure rate:** {lowest_failure['model']} — {safe_round(lowest_failure['failure_rate'])} failure rate.")

    with tab_recommendations:
        st.header("Preliminary Recommendations")

        st.warning(
            "These recommendations are based on latency, cost, heuristic failure rate, and truncation signals. "
            "They do not yet include LLM-as-a-judge or human evaluation."
        )

        st.subheader("Recommended model by use case")

        rec_display_cols = [
            "use_case",
            "rank",
            "model",
            "recommendation_score",
            "avg_latency",
            "failure_rate",
            "estimated_cost_per_1000_prompts_usd",
            "possible_truncations",
        ]

        available_rec_cols = [c for c in rec_display_cols if c in recommendations_usecase.columns]

        st.dataframe(
            recommendations_usecase[available_rec_cols].sort_values(["use_case", "rank"]),
            use_container_width=True,
        )

        st.subheader("Best model per use case")

        best_per_usecase = recommendations_usecase.sort_values("rank").groupby("use_case").head(1)

        for _, row in best_per_usecase.iterrows():
            st.markdown(
                f"**{row['use_case']}** → `{row['model']}` "
                f"(score: {safe_round(row['recommendation_score'], 4)}, "
                f"latency: {safe_round(row['avg_latency'])}s, "
                f"failure rate: {safe_round(row['failure_rate'])})"
            )

        st.subheader("Overall recommendations")

        overall_cols = [
            "overall_rank",
            "model",
            "overall_score",
            "avg_latency",
            "failure_rate",
            "estimated_cost_per_1000_prompts_usd",
            "possible_truncations",
        ]

        available_overall_cols = [c for c in overall_cols if c in recommendations_overall.columns]

        st.dataframe(
            recommendations_overall[available_overall_cols].sort_values("overall_rank"),
            use_container_width=True,
        )

    with tab_latency:
        st.header("Latency Analysis")

        simple_bar_chart(
            metrics_model,
            "model",
            "avg_latency",
            "Average latency by model",
            ylabel="Seconds",
        )

        simple_bar_chart(
            metrics_model,
            "model",
            "p95_latency",
            "P95 latency by model",
            ylabel="Seconds",
        )

        grouped_bar_chart(
            metrics_usecase,
            "use_case",
            "model",
            "avg_latency",
            "Average latency by model and use case",
            ylabel="Seconds",
        )

        st.subheader("Latency table")
        st.dataframe(
            metrics_model[
                [
                    "model",
                    "avg_latency",
                    "median_latency",
                    "p95_latency",
                    "min_latency",
                    "max_latency",
                ]
            ].sort_values("avg_latency"),
            use_container_width=True,
        )

    with tab_tokens_cost:
        st.header("Tokens & Cost")

        simple_bar_chart(
            metrics_model,
            "model",
            "total_tokens",
            "Total tokens by model",
            ylabel="Tokens",
        )

        simple_bar_chart(
            metrics_model,
            "model",
            "avg_output_tokens",
            "Average output tokens by model",
            ylabel="Output tokens",
        )

        simple_bar_chart(
            cost_model,
            "model",
            "estimated_cost_per_1000_prompts_usd",
            "Estimated cost per 1,000 prompts",
            ylabel="USD",
        )

        grouped_bar_chart(
            cost_usecase,
            "use_case",
            "model",
            "estimated_cost_per_1000_prompts_usd",
            "Estimated cost per 1,000 prompts by use case",
            ylabel="USD",
        )

        st.subheader("Cost table")
        st.dataframe(
            cost_model[
                [
                    "model",
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_tokens",
                    "total_cost_usd",
                    "avg_cost_per_prompt_usd",
                    "estimated_cost_per_1000_prompts_usd",
                ]
            ].sort_values("estimated_cost_per_1000_prompts_usd"),
            use_container_width=True,
        )

    with tab_failures:
        st.header("Failure Analysis")

        simple_bar_chart(
            failures_model,
            "model",
            "failure_rate",
            "Heuristic failure rate by model",
            ylabel="Failure rate",
        )

        simple_bar_chart(
            failures_model,
            "model",
            "possible_truncations",
            "Possible truncations by model",
            ylabel="Count",
        )

        heatmap_chart(
            failures_usecase,
            "model",
            "use_case",
            "failure_rate",
            "Failure rate heatmap by model and use case",
        )

        st.subheader("Failure summary by model")
        st.dataframe(
            failures_model.sort_values("failure_rate"),
            use_container_width=True,
        )

        st.subheader("Failure summary by model and use case")
        st.dataframe(
            failures_usecase.sort_values(["use_case", "failure_rate"]),
            use_container_width=True,
        )

    with tab_usecases:
        st.header("Use Case Analysis")

        selected_usecase = st.selectbox("Select use case", use_cases)

        usecase_metrics = metrics_usecase[metrics_usecase["use_case"] == selected_usecase]
        usecase_cost = cost_usecase[cost_usecase["use_case"] == selected_usecase]
        usecase_failures = failures_usecase[failures_usecase["use_case"] == selected_usecase]

        col1, col2 = st.columns(2)

        with col1:
            simple_bar_chart(
                usecase_metrics,
                "model",
                "avg_latency",
                f"Average latency — {selected_usecase}",
                ylabel="Seconds",
            )

        with col2:
            simple_bar_chart(
                usecase_cost,
                "model",
                "estimated_cost_per_1000_prompts_usd",
                f"Cost per 1,000 prompts — {selected_usecase}",
                ylabel="USD",
            )

        simple_bar_chart(
            usecase_failures,
            "model",
            "failure_rate",
            f"Failure rate — {selected_usecase}",
            ylabel="Failure rate",
        )

        st.subheader("Use case recommendation")
        rec_subset = recommendations_usecase[recommendations_usecase["use_case"] == selected_usecase]
        st.dataframe(
            rec_subset.sort_values("rank"),
            use_container_width=True,
        )

    with tab_scatter:
        st.header("Scatter / Pareto-style Comparison")

        combined = metrics_model.merge(
            cost_model[["model", "estimated_cost_per_1000_prompts_usd", "avg_cost_per_prompt_usd"]],
            on="model",
            how="left",
        ).merge(
            failures_model[["model", "failure_rate", "possible_truncations"]],
            on="model",
            how="left",
        )

        scatter_chart(
            combined,
            "estimated_cost_per_1000_prompts_usd",
            "avg_latency",
            "model",
            "Cost vs latency",
            "Estimated cost / 1,000 prompts (USD)",
            "Average latency (seconds)",
        )

        scatter_chart(
            combined,
            "estimated_cost_per_1000_prompts_usd",
            "failure_rate",
            "model",
            "Cost vs failure rate",
            "Estimated cost / 1,000 prompts (USD)",
            "Failure rate",
        )

        scatter_chart(
            combined,
            "avg_latency",
            "failure_rate",
            "model",
            "Latency vs failure rate",
            "Average latency (seconds)",
            "Failure rate",
        )

        st.info(
            "Pareto interpretation: better models are closer to the lower-left area, "
            "because they combine lower cost, lower latency, and lower failure rate."
        )

        st.dataframe(
            combined[
                [
                    "model",
                    "avg_latency",
                    "estimated_cost_per_1000_prompts_usd",
                    "failure_rate",
                    "possible_truncations",
                    "avg_total_tokens",
                ]
            ],
            use_container_width=True,
        )

    with tab_radar:
        st.header("Radar Chart")

        radar_df = prepare_radar_data(metrics_model, failures_model, cost_model)

        selected_radar_model = st.selectbox(
            "Select model for radar chart",
            radar_df["model"].tolist(),
        )

        row = radar_df[radar_df["model"] == selected_radar_model].iloc[0]

        radar_chart(row, f"Radar chart — {selected_radar_model}")

        st.caption(
            "Higher is better. Scores are normalized from latency, cost, failure rate, token usage, and truncation count."
        )

        st.dataframe(
            radar_df[
                [
                    "model",
                    "latency_score",
                    "cost_score",
                    "failure_score",
                    "token_score",
                    "truncation_score",
                ]
            ],
            use_container_width=True,
        )

    with tab_raw:
        st.header("Raw Data")

        st.subheader("Merged logs")
        st.dataframe(filtered_logs, use_container_width=True)

        st.subheader("Download data")

        st.download_button(
            "Download filtered logs as CSV",
            data=filtered_logs.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_logs.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
