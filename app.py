from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from database import connect, ensure_database_file, get_database_path, import_run_bundle
from reporting import (
    create_group_summary,
    create_model_summary,
    create_provider_summary,
    create_stratified_human_sample,
    flatten_judgments,
    recalculate_results,
)

st.set_page_config(
    page_title="LLM Evaluation — Qwen vs Gemma",
    page_icon="🧪",
    layout="wide",
)


def check_password() -> bool:
    expected = os.getenv("APP_PASSWORD", "").strip()
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("LLM Evaluation Platform")
    st.caption("Password-protected benchmark dashboard")
    password = st.text_input("Application password", type="password")
    if st.button("Sign in", type="primary"):
        if hashlib.sha256(password.encode()).digest() == hashlib.sha256(expected.encode()).digest():
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

DB_PATH = ensure_database_file(get_database_path())


@st.cache_data(show_spinner=False)
def query_df(query: str, params: tuple[Any, ...] = (), db_mtime: float = 0.0) -> pd.DataFrame:
    del db_mtime
    with connect(DB_PATH, readonly=True) as con:
        return pd.read_sql_query(query, con, params=params)


def db_mtime() -> float:
    return DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0


def load_summary(table: str, run_id: str) -> pd.DataFrame:
    allowed = {"model_summary", "category_summary", "difficulty_summary", "provider_summary"}
    if table not in allowed:
        raise ValueError(table)
    raw = query_df(f"SELECT summary_json FROM {table} WHERE run_id = ?", (run_id,), db_mtime())
    if raw.empty:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(value) for value in raw["summary_json"]])


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def fmt_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


runs = query_df(
    "SELECT run_id, created_at, dataset_name, dataset_version, judge_model, execution_mode, notes "
    "FROM runs ORDER BY created_at DESC",
    db_mtime=db_mtime(),
)
if runs.empty:
    st.error("The SQLite database contains no benchmark runs.")
    st.stop()

run_options = runs["run_id"].tolist()
selected_run = st.sidebar.selectbox("Benchmark run", run_options)
run_info = runs[runs["run_id"] == selected_run].iloc[0]

st.sidebar.caption(f"Database: `{DB_PATH}`")
st.sidebar.caption(f"Dataset v{run_info['dataset_version']} · {run_info['execution_mode']}")

st.title("LLM Evaluation Platform")
st.caption(
    "Research-inspired Greek benchmark comparing Qwen 3.6 27B and Gemma 4 31B Instruct. "
    "Results combine deterministic checks, an independent blind LLM judge, latency, tokens, cost and provider metadata."
)

model_summary = load_summary("model_summary", selected_run)
category_summary = load_summary("category_summary", selected_run)
difficulty_summary = load_summary("difficulty_summary", selected_run)
provider_summary = load_summary("provider_summary", selected_run)
outputs = query_df(
    "SELECT * FROM model_outputs WHERE run_id = ? ORDER BY prompt_id, model_name",
    (selected_run,),
    db_mtime(),
)
judge = query_df(
    "SELECT * FROM judge_scores WHERE run_id = ? ORDER BY prompt_id, model_name",
    (selected_run,),
    db_mtime(),
)
pairwise = query_df(
    "SELECT * FROM pairwise_judgments WHERE run_id = ? ORDER BY prompt_id",
    (selected_run,),
    db_mtime(),
)

models = sorted(outputs["model_name"].dropna().unique().tolist())
categories = sorted(outputs["category"].dropna().unique().tolist())
difficulties = [item for item in ["easy", "medium", "hard"] if item in outputs["difficulty"].unique()]
selected_models = st.sidebar.multiselect("Models", models, default=models)
selected_categories = st.sidebar.multiselect("Categories", categories, default=categories)
selected_difficulties = st.sidebar.multiselect("Difficulty", difficulties, default=difficulties)

filtered_outputs = outputs[
    outputs["model_name"].isin(selected_models)
    & outputs["category"].isin(selected_categories)
    & outputs["difficulty"].isin(selected_difficulties)
].copy()
filtered_judge = judge[
    judge["model_name"].isin(selected_models)
    & judge["category"].isin(selected_categories)
    & judge["difficulty"].isin(selected_difficulties)
].copy()

request_count = len(filtered_outputs)
success_rate = (filtered_outputs["status"] == "success").mean() if request_count else float("nan")
avg_latency = filtered_outputs.loc[filtered_outputs["status"] == "success", "latency_seconds"].mean()
total_cost = filtered_outputs["cost_usd"].sum(min_count=1)
quality = filtered_judge["overall_quality"].mean() if not filtered_judge.empty else float("nan")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests", request_count)
c2.metric("API success", fmt_pct(success_rate))
c3.metric("Mean latency", f"{avg_latency:.2f}s" if pd.notna(avg_latency) else "—")
c4.metric("Recorded model cost", f"${total_cost:.4f}" if pd.notna(total_cost) else "—")
c5.metric("Mean judge quality", f"{quality:.2f}/5" if pd.notna(quality) else "—")

st.info(
    "Latency and cost reflect this specific OpenRouter run and the providers selected for each request. "
    "They should not be interpreted as provider-independent properties of the models."
)

tabs = st.tabs([
    "Overview",
    "Quality",
    "Deterministic checks",
    "Performance & providers",
    "Prompt explorer",
    "Blind human review",
    "Methodology",
    "Admin import",
])

with tabs[0]:
    st.subheader("Corrected model summary")
    visible = [
        "model_name", "api_success_rate", "latency_mean_seconds", "latency_p50_seconds",
        "latency_p95_seconds", "total_cost_usd", "deterministic_pass_rate",
        "structured_syntax_validity_rate", "structured_schema_validity_rate",
        "structured_format_compliance_rate", "judge_overall_quality_mean",
        "judge_task_success_rate", "judge_hallucination_rate", "judge_win_count",
        "judge_tie_count", "judge_win_rate_decisive", "judge_win_rate_ci95_low",
        "judge_win_rate_ci95_high",
    ]
    st.dataframe(model_summary[[c for c in visible if c in model_summary]], hide_index=True, width="stretch")

    if not model_summary.empty:
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                model_summary,
                x="model_name",
                y="judge_overall_quality_mean",
                error_y=model_summary.get("judge_overall_quality_ci95_high") - model_summary.get("judge_overall_quality_mean"),
                error_y_minus=model_summary.get("judge_overall_quality_mean") - model_summary.get("judge_overall_quality_ci95_low"),
                text_auto=".3f",
                title="Mean judge quality with approximate 95% CI",
                labels={"model_name": "Model", "judge_overall_quality_mean": "Score (1–5)"},
            )
            fig.update_yaxes(range=[0, 5])
            st.plotly_chart(fig, width="stretch")
        with right:
            outcomes = model_summary[["model_name", "judge_win_count", "judge_tie_count"]].melt(
                id_vars="model_name", var_name="outcome", value_name="prompts"
            )
            fig = px.bar(outcomes, x="model_name", y="prompts", color="outcome", barmode="group", title="Effective pairwise outcomes")
            st.plotly_chart(fig, width="stretch")

    st.markdown(
        "**Interpretation:** Qwen has the higher decisive win rate, while Gemma has lower latency and cost, "
        "higher task success, and a lower hallucination rate in this run. The overall judge-quality means overlap strongly."
    )

with tabs[1]:
    st.subheader("Quality by category")
    quality_by_category = category_summary[category_summary["model_name"].isin(selected_models)]
    if not quality_by_category.empty:
        fig = px.bar(
            quality_by_category,
            x="category",
            y="overall_quality_mean",
            color="model_name",
            barmode="group",
            text_auto=".2f",
            labels={"overall_quality_mean": "Overall quality (1–5)"},
        )
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(quality_by_category, hide_index=True, width="stretch")

    st.subheader("Quality by difficulty")
    quality_by_difficulty = difficulty_summary[difficulty_summary["model_name"].isin(selected_models)]
    if not quality_by_difficulty.empty:
        fig = px.bar(
            quality_by_difficulty,
            x="difficulty",
            y="overall_quality_mean",
            color="model_name",
            barmode="group",
            text_auto=".2f",
            category_orders={"difficulty": ["easy", "medium", "hard"]},
        )
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, width="stretch")

    if not filtered_judge.empty:
        score_columns = ["correctness", "instruction_following", "factuality_grounding", "greek_quality", "overall_quality"]
        long = filtered_judge.groupby("model_name")[score_columns].mean().reset_index().melt(
            id_vars="model_name", var_name="metric", value_name="score"
        )
        fig = px.bar(long, x="metric", y="score", color="model_name", barmode="group", text_auto=".2f", title="Judge dimensions")
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, width="stretch")

with tabs[2]:
    st.subheader("Exact-answer and numeric tasks")
    exact = model_summary[[
        "model_name", "deterministic_evaluated", "deterministic_pass_rate",
        "strict_exact_rate", "normalized_exact_rate", "numeric_accuracy",
    ]].copy()
    st.dataframe(exact, hide_index=True, width="stretch")

    st.subheader("Structured output")
    structured = model_summary[[
        "model_name", "structured_evaluated", "structured_syntax_validity_rate",
        "structured_schema_validity_rate", "structured_format_compliance_rate",
    ]].copy()
    st.dataframe(structured, hide_index=True, width="stretch")
    st.caption(
        "Syntax validity checks whether JSON/YAML/CSV/Markdown is parseable. Schema validity checks the requested keys and types. "
        "Format compliance also enforces instructions such as ‘only JSON’ and no Markdown code fences."
    )

    structured_rows = filtered_outputs[filtered_outputs["structured_required"] == 1]
    if not structured_rows.empty:
        plot_data = structured_rows.groupby("model_name", as_index=False).agg(
            syntax_validity=("syntax_valid", "mean"),
            schema_validity=("schema_valid", "mean"),
            format_compliance=("format_compliance", "mean"),
        ).melt(id_vars="model_name", var_name="metric", value_name="rate")
        fig = px.bar(plot_data, x="metric", y="rate", color="model_name", barmode="group", text_auto=".1%")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

with tabs[3]:
    st.subheader("Latency distributions")
    successful = filtered_outputs[filtered_outputs["status"] == "success"]
    fig = px.box(
        successful,
        x="model_name",
        y="latency_seconds",
        color="model_name",
        points="all",
        hover_data=["prompt_id", "category", "difficulty", "provider"],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Provider routing")
    provider_filtered = provider_summary[provider_summary["model_name"].isin(selected_models)]
    st.dataframe(provider_filtered, hide_index=True, width="stretch")
    if not provider_filtered.empty:
        fig = px.bar(
            provider_filtered,
            x="provider",
            y="requests",
            color="model_name",
            barmode="group",
            hover_data=["latency_mean_seconds", "latency_p95_seconds", "total_cost_usd"],
        )
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        cost = model_summary[model_summary["model_name"].isin(selected_models)]
        fig = px.bar(cost, x="model_name", y="total_cost_usd", text_auto=".5f", title="Total recorded model cost")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(
            model_summary[model_summary["model_name"].isin(selected_models)],
            x="latency_mean_seconds",
            y="judge_overall_quality_mean",
            size="total_cost_usd",
            color="model_name",
            hover_data=["judge_task_success_rate", "judge_hallucination_rate"],
            title="Quality–latency–cost trade-off",
        )
        st.plotly_chart(fig, width="stretch")

with tabs[4]:
    prompt_ids = sorted(filtered_outputs["prompt_id"].unique().tolist())
    if not prompt_ids:
        st.info("No prompts match the selected filters.")
    else:
        prompt_id = st.selectbox("Prompt", prompt_ids)
        prompt_rows = filtered_outputs[filtered_outputs["prompt_id"] == prompt_id]
        first = prompt_rows.iloc[0]
        st.markdown(f"**Category:** `{first['category']}` · **Difficulty:** `{first['difficulty']}`")
        prompt_text = query_df(
            "SELECT prompt_text FROM prompts WHERE prompt_id = ? ORDER BY dataset_version DESC LIMIT 1",
            (prompt_id,),
            db_mtime(),
        )
        st.markdown("### Prompt")
        st.write(prompt_text.iloc[0, 0] if not prompt_text.empty else "")

        columns = st.columns(max(1, len(prompt_rows)))
        for column, (_, row) in zip(columns, prompt_rows.iterrows()):
            with column:
                st.markdown(f"### {row['model_name']}")
                st.caption(
                    f"{row['latency_seconds']:.3f}s · {int(row['total_tokens'] or 0)} tokens · "
                    f"${float(row['cost_usd'] or 0):.6f} · {row['provider']}"
                )
                st.write(row["response"])
                score = filtered_judge[(filtered_judge["prompt_id"] == prompt_id) & (filtered_judge["model_name"] == row["model_name"])]
                if not score.empty:
                    s = score.iloc[0]
                    st.caption(
                        f"Judge: quality {s['overall_quality']}/5 · task success {bool(s['task_success'])} · "
                        f"hallucination {bool(s['hallucination_detected'])}"
                    )
                    if s.get("critical_error"):
                        st.error(s["critical_error"])
                    with st.expander("Judge rationale"):
                        st.write(s["rationale"])

        pair = pairwise[pairwise["prompt_id"] == prompt_id]
        if not pair.empty:
            item = pair.iloc[0]
            st.markdown("### Pairwise judgment")
            st.write(
                f"Effective result: **{item['winner_model_effective']}** "
                f"(raw label: `{item['winner_label_raw']}`, confidence: {item['confidence']})"
            )
            st.write(item["pairwise_reason"])

with tabs[5]:
    st.subheader("Stratified blind human sample")
    st.caption("30 prompts: one easy, one medium and one hard from each category. Model identity remains hidden during review.")
    sample = query_df(
        "SELECT * FROM human_review_sample WHERE run_id = ? ORDER BY category, difficulty, prompt_id",
        (selected_run,),
        db_mtime(),
    )
    reviews = query_df(
        "SELECT * FROM human_reviews WHERE run_id = ?",
        (selected_run,),
        db_mtime(),
    )
    completed = reviews["prompt_id"].nunique() if not reviews.empty else 0
    st.progress(completed / len(sample) if len(sample) else 0, text=f"Completed: {completed}/{len(sample)}")

    if not sample.empty:
        review_prompt = st.selectbox("Review item", sample["prompt_id"].tolist(), key="human_prompt")
        row = sample[sample["prompt_id"] == review_prompt].iloc[0]
        st.markdown(f"**{row['category']} · {row['difficulty']}**")
        st.write(row["prompt_text"])
        a_col, b_col = st.columns(2)
        with a_col:
            st.markdown("### Answer A")
            st.write(row["answer_a"])
        with b_col:
            st.markdown("### Answer B")
            st.write(row["answer_b"])

        existing = reviews[reviews["prompt_id"] == review_prompt] if "prompt_id" in reviews.columns else pd.DataFrame()
        existing_row = existing.iloc[0] if not existing.empty else None
        winner_options = ["", "A", "B", "tie"]
        default_winner = str(existing_row["winner"]) if existing_row is not None and pd.notna(existing_row["winner"]) else ""
        winner = st.selectbox("Winner", winner_options, index=winner_options.index(default_winner) if default_winner in winner_options else 0)
        cols = st.columns(4)
        metrics: dict[str, tuple[int, int]] = {}
        for col, metric in zip(cols, ["correctness", "instruction_following", "factuality", "greek_quality"]):
            with col:
                old_a = int(existing_row[f"{metric}_a"]) if existing_row is not None and pd.notna(existing_row[f"{metric}_a"]) else 3
                old_b = int(existing_row[f"{metric}_b"]) if existing_row is not None and pd.notna(existing_row[f"{metric}_b"]) else 3
                a_score = st.slider(f"{metric} A", 1, 5, old_a, key=f"{review_prompt}-{metric}-a")
                b_score = st.slider(f"{metric} B", 1, 5, old_b, key=f"{review_prompt}-{metric}-b")
                metrics[metric] = (a_score, b_score)
        old_comments = str(existing_row["comments"]) if existing_row is not None and pd.notna(existing_row["comments"]) else ""
        comments = st.text_area("Comments", value=old_comments)
        if st.button("Save blinded review", type="primary", disabled=not winner):
            with connect(DB_PATH) as con:
                con.execute(
                    """
                    INSERT INTO human_reviews(
                        run_id,prompt_id,reviewer,winner,correctness_a,correctness_b,
                        instruction_following_a,instruction_following_b,factuality_a,factuality_b,
                        greek_quality_a,greek_quality_b,comments,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                    ON CONFLICT(run_id,prompt_id,reviewer) DO UPDATE SET
                        winner=excluded.winner,correctness_a=excluded.correctness_a,correctness_b=excluded.correctness_b,
                        instruction_following_a=excluded.instruction_following_a,
                        instruction_following_b=excluded.instruction_following_b,
                        factuality_a=excluded.factuality_a,factuality_b=excluded.factuality_b,
                        greek_quality_a=excluded.greek_quality_a,greek_quality_b=excluded.greek_quality_b,
                        comments=excluded.comments,updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        selected_run, review_prompt, "default", winner,
                        *metrics["correctness"], *metrics["instruction_following"],
                        *metrics["factuality"], *metrics["greek_quality"], comments,
                    ),
                )
                con.commit()
            st.cache_data.clear()
            st.success("Review saved.")
            st.rerun()

        reveal = st.checkbox("Reveal model identities and human–judge agreement", disabled=completed < len(sample))
        if reveal:
            merged = sample.merge(reviews, on=["run_id", "prompt_id"], how="inner")
            def human_model(item: pd.Series) -> str:
                if item["winner"] == "A": return item["answer_a_model"]
                if item["winner"] == "B": return item["answer_b_model"]
                return "tie"
            merged["human_winner_model"] = merged.apply(human_model, axis=1)
            pair_labels = pairwise[["prompt_id", "winner_model_effective"]]
            merged = merged.merge(pair_labels, on="prompt_id", how="left")
            merged["agreement"] = merged["human_winner_model"] == merged["winner_model_effective"]
            st.metric("Exact human–judge agreement", fmt_pct(merged["agreement"].mean()))
            st.dataframe(
                merged[["prompt_id", "category", "difficulty", "human_winner_model", "winner_model_effective", "agreement"]],
                hide_index=True,
                width="stretch",
            )

with tabs[6]:
    st.markdown(
        """
        ### Dataset
        - 120 original Greek prompts across 10 categories.
        - 40 easy, 40 medium and 40 hard prompts.
        - Task formats inspired by IFEval/M-IFEval, HELM, BIG-Bench Hard, GSM8K, TruthfulQA and MT-Bench.

        ### Evaluation layers
        1. **API telemetry:** success, latency, provider, tokens and recorded cost.
        2. **Deterministic evaluation:** strict/normalized/numeric matching and structured-output validation.
        3. **Independent blind judge:** GPT-4.1 mini, with randomized A/B order and explicit ties.
        4. **Blind human review:** stratified 30-prompt sample.

        ### Corrections in benchmark v1.1
        - JSON detection is based only on the declared required output format.
        - JSON, YAML, CSV and Markdown-table tasks use separate syntax and schema checks.
        - Exact-answer tasks use explicit strict, contains or numeric matching.
        - MR-H03 reference answer was corrected to approximately 20.7143%.
        - `cannot_assess` is normalized to a tie only when both answers have the same recorded task-success status; the raw judge label is preserved.

        ### Interpretation limits
        - One run per prompt–model pair does not estimate generation variance.
        - Provider routing confounds model-only latency and cost comparisons.
        - LLM judging can contain preference and verbosity bias.
        - The benchmark supports workload-specific selection; it does not establish a universally best model.
        """
    )
    st.download_button(
        "Download model summary CSV",
        model_summary.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{selected_run}_model_summary.csv",
        mime="text/csv",
    )


with tabs[7]:
    st.subheader("Import a completed local run into SQLite")
    st.caption(
        "Upload the raw JSON artifacts created by main.py. The app recalculates deterministic metrics, "
        "rebuilds summaries and imports the run without repeating any model or judge API calls."
    )
    metadata_file = st.file_uploader("Run metadata JSON", type=["json"], key="metadata_upload")
    results_file = st.file_uploader("Raw results JSON", type=["json"], key="results_upload")
    judgments_file = st.file_uploader("Pairwise judgments JSON", type=["json"], key="judgments_upload")
    if st.button("Import run into SQLite", type="primary", disabled=not all([metadata_file, results_file, judgments_file])):
        try:
            metadata = json.load(metadata_file)
            raw_results = json.load(results_file)
            judgments = json.load(judgments_file)
            dataset = json.loads((Path(__file__).resolve().parent / "benchmark_prompts.json").read_text(encoding="utf-8"))
            prompts = dataset["prompts"]
            run_id = str(metadata.get("run_id") or raw_results[0].get("run_id"))
            metadata.setdefault("run_id", run_id)
            metadata.setdefault("dataset_metadata", {k: v for k, v in dataset.items() if k != "prompts"})
            corrected = recalculate_results(raw_results, prompts)
            judge_rows = flatten_judgments(judgments)
            model_rows = create_model_summary(run_id, corrected, judge_rows)
            category_rows = create_group_summary(run_id, corrected, judge_rows, "category")
            difficulty_rows = create_group_summary(run_id, corrected, judge_rows, "difficulty")
            provider_rows = create_provider_summary(run_id, corrected)
            human_rows, human_key = create_stratified_human_sample(run_id, prompts, corrected)
            import_run_bundle(
                db_path=DB_PATH, metadata=metadata, prompts=prompts, results=corrected,
                judgments=judgments, judge_rows=judge_rows, model_summary=model_rows,
                category_summary=category_rows, difficulty_summary=difficulty_rows,
                provider_summary=provider_rows, human_rows=human_rows, human_key=human_key,
                notes="Imported through the hosted Streamlit admin interface.",
            )
            st.cache_data.clear()
            st.success(f"Imported {run_id}.")
            st.rerun()
        except Exception as exc:
            st.exception(exc)
