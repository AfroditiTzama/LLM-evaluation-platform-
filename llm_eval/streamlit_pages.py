from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from database import database_label

from .domain import GenerationSettings, RunSpec
from .evaluators import default_evaluator_registry
from .framework_views import (
    build_leaderboard,
    decode_json,
    enrich_results,
    load_metric_values,
    load_results,
    load_runs,
    mark_pareto_frontier,
    quality_metric_options,
)
from .run_service import (
    available_datasets,
    compatible_prompt_strategies,
    execute_run,
    load_default_catalogs,
    new_run_id,
    runnable_task_ids,
)
from .streamlit_ui import openrouter_key_is_configured, render_workflow, switch_page_button


def _strategy_label(strategy) -> str:
    return f"{strategy.name} · v{strategy.version} ({strategy.strategy_type})"


def _run_label(row: pd.Series) -> str:
    return f"{row['run_id']} · {row.get('task_name') or row['task_id']} · {row['status']}"


def _selected_run(runs: pd.DataFrame, *, key: str) -> str:
    run_ids = runs["run_id"].tolist()
    preferred = st.session_state.get("framework_run_id")
    index = run_ids.index(preferred) if preferred in run_ids else 0
    labels = {row["run_id"]: _run_label(row) for _, row in runs.iterrows()}
    return st.selectbox(
        "Evaluation run",
        run_ids,
        index=index,
        format_func=lambda value: labels[value],
        key=key,
    )


def _fmt_number(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_cost(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        number = float(value)
        if 0 < abs(number) < 0.0001:
            return "<$0.0001"
        return f"${number:.4f}"
    except (TypeError, ValueError):
        return "—"


def _uploaded_payload() -> Any | None:
    uploaded = st.file_uploader(
        "Canonical dataset JSON",
        type=["json"],
        help="Upload a dataset with id, name, task_id, version, provenance and records.",
    )
    if uploaded is None:
        return None
    try:
        return json.load(uploaded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"The uploaded dataset is not valid JSON: {exc}")
        return None


def render_run_builder(*, base_dir: Path, db_path: Path | None) -> None:
    st.title("Run Evaluation")
    st.caption(
        "Choose what to evaluate, review the exact number of paid requests, and then start the run."
    )
    render_workflow(1)

    catalogs = load_default_catalogs(base_dir)
    evaluators = default_evaluator_registry()
    with st.expander("Advanced · Upload a different dataset"):
        payload = _uploaded_payload()
    try:
        datasets = available_datasets(
            benchmark_path=base_dir / "benchmark_prompts.json",
            uploaded_payload=payload,
        )
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    runnable = runnable_task_ids(catalogs, evaluators, datasets)
    with st.expander("Advanced · Task and evaluator availability"):
        rows = []
        for task in catalogs.tasks.values():
            rows.append(
                {
                    "task": task.name,
                    "task_id": task.id,
                    "dataset_available": task.id in datasets,
                    "evaluator": f"{task.evaluator_type}@{task.evaluator_version}",
                    "evaluator_status": "runnable" if evaluators.contains(task.evaluator_type) else "planned",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if not runnable:
        st.warning("No task currently has both a compatible dataset and an implemented evaluator.")
        return

    st.subheader("1 · Evaluation scope")
    task_id = st.selectbox(
        "Task",
        runnable,
        format_func=lambda value: catalogs.tasks.get(value).name,
        help="The task determines which evaluator and quality metrics will be used.",
    )
    task = catalogs.tasks.get(task_id)
    dataset = datasets[task_id]

    with st.container(border=True):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Dataset", dataset.name)
        d2.metric("Version", dataset.version)
        d3.metric("Examples", len(dataset.records))
        d4.metric("Language", dataset.language or "mixed / unspecified")
        st.caption(
            f"Provenance: {dataset.provenance.record_origin} · "
            f"{dataset.provenance.source_title or 'No source title supplied'}"
        )

    st.subheader("2 · Models and prompt strategies")
    model_options = [model.id for model in catalogs.models]
    model_labels = {model.id: f"{model.name} · {model.provider}" for model in catalogs.models}
    selected_model_ids = st.multiselect(
        "Models",
        model_options,
        default=model_options[:1],
        format_func=lambda value: model_labels[value],
        help="Start with one model to keep the first run smaller. Add more models when you want a comparison.",
    )
    selected_models = tuple(model for model in catalogs.models if model.id in selected_model_ids)

    strategies = compatible_prompt_strategies(dataset, catalogs.prompt_strategies.values())
    strategy_by_key = {strategy.key: strategy for strategy in strategies}
    strategy_keys = list(strategy_by_key)
    default_strategies = [
        key
        for key in ("basic-zero-shot-el@1", "optimized-grounded-el@1")
        if key in strategy_by_key
    ]
    selected_strategy_keys = st.multiselect(
        "Prompt strategies",
        strategy_keys,
        default=default_strategies or strategy_keys[:1],
        format_func=lambda key: _strategy_label(strategy_by_key[key]),
        help="Choose at least two strategies if you want to use Prompt Comparison after the run.",
    )
    selected_strategies = tuple(strategy_by_key[key] for key in selected_strategy_keys)
    excluded = len(catalogs.prompt_strategies.values()) - len(strategies)
    if excluded:
        st.caption(
            f"{excluded} incompatible prompt strategy/strategies are hidden because this dataset does not provide "
            "all of their required variables."
        )

    st.subheader("3 · Generation settings")
    g1, g2, g3 = st.columns(3)
    temperature = g1.slider("Temperature", 0.0, 2.0, 0.2, 0.05)
    top_p = g2.slider("Top-p", 0.05, 1.0, 0.9, 0.05)
    max_tokens = g3.number_input(
        "Max output tokens",
        min_value=1,
        max_value=100000,
        value=500,
        step=100,
        help="500 is a safer starting limit for the short tasks in the included dataset.",
    )
    reasoning_enabled = st.checkbox("Enable model reasoning", value=False)
    reasoning_effort = st.selectbox(
        "Reasoning effort",
        ["low", "medium", "high"],
        index=1,
        disabled=not reasoning_enabled,
    )
    unsupported_reasoning = [model.name for model in selected_models if not model.reasoning_support]
    if reasoning_enabled and unsupported_reasoning:
        st.error("Reasoning is not declared as supported for: " + ", ".join(unsupported_reasoning))

    st.subheader("4 · Review and run")
    request_count = len(selected_models) * len(selected_strategies) * len(dataset.records)
    with st.container(border=True):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Paid requests", request_count)
        r2.metric("Models", len(selected_models))
        r3.metric("Prompt strategies", len(selected_strategies))
        r4.metric("Examples", len(dataset.records))
        st.caption(
            f"Calculation: {len(selected_models)} model(s) × {len(selected_strategies)} prompt strategy/strategies "
            f"× {len(dataset.records)} examples = {request_count} OpenRouter requests."
        )

    if request_count > 24:
        st.warning(
            "This is a larger paid run. For a first test, select one model and one or two prompt strategies."
        )
    st.info(
        "Exact cost is recorded from OpenRouter after each response. A pre-run dollar estimate is unavailable "
        "because the catalog does not contain verified fixed prices for these models."
    )

    api_available = openrouter_key_is_configured()
    if not api_available:
        st.warning(
            "A real OPENROUTER_API_KEY is not configured. Add it to .env and restart the app. "
            "Placeholder values such as 'your-openrouter-api-key' are not accepted."
        )
    confirmed = st.checkbox(
        f"I understand that clicking Run Evaluation will send {request_count} potentially paid requests to OpenRouter.",
        value=False,
        disabled=request_count == 0,
    )
    disabled = (
        not selected_models
        or not selected_strategies
        or request_count == 0
        or not confirmed
        or not api_available
        or (reasoning_enabled and bool(unsupported_reasoning))
    )
    if st.button(
        f"Run evaluation · {request_count} requests",
        type="primary",
        disabled=disabled,
        width="stretch",
    ):
        spec = RunSpec(
            run_id=new_run_id(),
            models=selected_models,
            task=task,
            dataset=dataset,
            prompt_strategies=selected_strategies,
            generation_settings=GenerationSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=int(max_tokens),
                reasoning_enabled=reasoning_enabled,
                reasoning_effort=reasoning_effort if reasoning_enabled else None,
            ),
            metric_configuration={"common_metrics_version": "1.0"},
            metadata={"entrypoint": "streamlit_run_builder", "database": database_label(db_path)},
        )
        try:
            progress = st.progress(0.0, text=f"Preparing {request_count} requests...")
            progress_status = st.empty()
            accumulated_cost = 0.0

            def update_progress(result, completed: int, total: int) -> None:
                nonlocal accumulated_cost
                accumulated_cost += float(result.usage.cost_usd or 0)
                progress.progress(
                    completed / total,
                    text=f"Completed {completed}/{total} results",
                )
                progress_status.caption(
                    f"{result.record_id} · {result.model_snapshot.get('name', result.model_key)} · "
                    f"{result.prompt_strategy_snapshot.get('name', result.prompt_strategy_id)} · "
                    f"{result.status} · recorded cost ${accumulated_cost:.6f}"
                )

            with st.spinner(f"Running {request_count} provider requests. Results are saved after every example..."):
                results = execute_run(spec, db_path=db_path, on_result=update_progress)
            progress.progress(1.0, text=f"Completed {len(results)}/{request_count} results")
            successes = sum(result.status == "success" for result in results)
            st.session_state["framework_run_id"] = spec.run_id
            st.success(
                f"Run {spec.run_id} completed: {successes}/{len(results)} successful results."
            )
            action_left, action_right = st.columns(2)
            with action_left:
                switch_page_button(
                    "View framework results",
                    "pages/2_Framework_Results.py",
                    primary=True,
                    key="view-completed-results",
                )
            with action_right:
                switch_page_button(
                    "Compare prompt strategies",
                    "pages/3_Prompt_Comparison.py",
                    key="compare-completed-prompts",
                )
        except Exception as exc:
            st.exception(exc)


def _framework_data(db_path: Path | None, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = load_results(db_path, run_id)
    metrics = load_metric_values(db_path, run_id)
    return results, metrics, enrich_results(results, metrics)


def _filter_results(enriched: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    if enriched.empty:
        return enriched
    c1, c2, c3 = st.columns(3)
    models = sorted(enriched["model_name"].dropna().unique().tolist())
    prompts = sorted(enriched["prompt_strategy_name"].dropna().unique().tolist())
    difficulties = sorted(enriched["difficulty"].dropna().unique().tolist())
    selected_models = c1.multiselect("Models", models, default=models, key=f"{prefix}-models")
    selected_prompts = c2.multiselect("Prompt strategies", prompts, default=prompts, key=f"{prefix}-prompts")
    selected_difficulties = c3.multiselect(
        "Difficulty",
        difficulties,
        default=difficulties,
        key=f"{prefix}-difficulty",
    )
    return enriched[
        enriched["model_name"].isin(selected_models)
        & enriched["prompt_strategy_name"].isin(selected_prompts)
        & enriched["difficulty"].isin(selected_difficulties)
    ].copy()


def render_framework_results(*, db_path: Path | None) -> None:
    st.title("Framework Results")
    st.caption("Analyze quality, reliability, latency, tokens and recorded cost for a completed evaluation run.")
    render_workflow(2)
    runs = load_runs(db_path)
    if runs.empty:
        st.info("No framework evaluation runs exist yet. Create the first run to unlock results and charts.")
        switch_page_button(
            "Create the first evaluation run",
            "pages/1_Run_Evaluation.py",
            primary=True,
            key="empty-results-create-run",
        )
        return
    run_id = _selected_run(runs, key="results-run")
    run = runs[runs["run_id"] == run_id].iloc[0]
    st.caption(
        f"Task: {run.get('task_name') or run['task_id']} · Dataset: {run.get('dataset_name') or run['dataset_id']} "
        f"v{run['dataset_version']} · Evaluator: {run['evaluator_type']}@{run['evaluator_version']}"
    )
    results, metrics, enriched = _framework_data(db_path, run_id)
    filtered = _filter_results(enriched, prefix="results")
    filtered_ids = set(filtered["result_id"].tolist()) if not filtered.empty else set()
    filtered_metrics = metrics[metrics["result_id"].isin(filtered_ids)] if filtered_ids else metrics.iloc[0:0]

    request_count = len(filtered)
    success_rate = (filtered["status"] == "success").mean() if request_count else None
    total_cost = filtered["cost_usd"].sum(min_count=1) if request_count else None
    mean_latency = filtered["end_to_end_seconds"].mean() if request_count else None
    total_tokens = filtered["total_tokens"].sum(min_count=1) if request_count else None
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Results", request_count)
        c2.metric("Success", f"{success_rate:.1%}" if success_rate is not None else "—")
        c3.metric("Mean latency", f"{_fmt_number(mean_latency, 3)}s")
        c4.metric("Total tokens", int(total_tokens) if pd.notna(total_tokens) else "—")
        c5.metric("Recorded cost", _fmt_cost(total_cost))

    quality_options = quality_metric_options(filtered_metrics)
    quality_metric = st.selectbox(
        "Quality metric",
        quality_options,
        index=0,
        help="Metrics remain task-specific; the framework does not create a universal quality score.",
    ) if quality_options else None
    if quality_metric is None:
        st.warning("This run contains no task-specific numeric quality metric.")

    leaderboard = mark_pareto_frontier(
        build_leaderboard(filtered, quality_metric=quality_metric)
    )
    st.subheader("Leaderboard")
    st.dataframe(leaderboard, hide_index=True, width="stretch")

    if not leaderboard.empty and quality_metric and "quality_mean" in leaderboard:
        left, right = st.columns(2)
        with left:
            chart = px.bar(
                leaderboard,
                x="model_name",
                y="quality_mean",
                color="prompt_strategy_name",
                barmode="group",
                error_y="quality_std",
                title=f"{quality_metric}: mean ± std",
                hover_data=["prompt_strategy_version", "quality_median", "results"],
            )
            st.plotly_chart(chart, width="stretch")
        with right:
            pareto = px.scatter(
                leaderboard,
                x="cost_total_usd",
                y="quality_mean",
                color="model_name",
                symbol="pareto_optimal",
                hover_data=["prompt_strategy_name", "prompt_strategy_version", "latency_mean_seconds"],
                title="Quality–cost trade-off (Pareto points marked)",
            )
            st.plotly_chart(pareto, width="stretch")

    if not filtered.empty and quality_metric and quality_metric in filtered:
        difficulty = filtered.groupby(
            ["model_name", "prompt_strategy_name", "difficulty"], as_index=False
        )[quality_metric].mean()
        if not difficulty.empty:
            st.subheader("Quality by difficulty")
            chart = px.bar(
                difficulty,
                x="difficulty",
                y=quality_metric,
                color="model_name",
                facet_col="prompt_strategy_name",
                barmode="group",
            )
            st.plotly_chart(chart, width="stretch")

    st.subheader("Individual example")
    if filtered.empty:
        st.info("No result matches the current filters.")
        return
    result_options = filtered["result_id"].tolist()
    labels = {
        row["result_id"]: (
            f"{row['record_id']} · {row['model_name']} · "
            f"{row['prompt_strategy_name']} v{row['prompt_strategy_version']} · {row['status']}"
        )
        for _, row in filtered.iterrows()
    }
    result_id = st.selectbox(
        "Result",
        result_options,
        format_func=lambda value: labels[value],
    )
    item = filtered[filtered["result_id"] == result_id].iloc[0]
    in_col, out_col = st.columns(2)
    with in_col:
        st.markdown("#### Input")
        st.write(decode_json(item["raw_input_json"]))
        st.markdown("#### Reference")
        st.write(decode_json(item["reference_json"]))
    with out_col:
        st.markdown("#### Output")
        st.write(item["raw_output"])
        if item["error"]:
            st.error(item["error"])
    with st.expander("Immutable rendered prompt"):
        st.markdown("**System prompt**")
        st.code(item["system_prompt_snapshot"] or "(empty)")
        st.markdown("**User prompt**")
        st.code(item["user_prompt_snapshot"])
    example_metrics = metrics[metrics["result_id"] == result_id].copy()
    if not example_metrics.empty:
        st.markdown("#### Metrics and scoring metadata")
        visible = example_metrics[["metric_name", "value", "unit", "metadata_json"]].copy()
        visible["scoring_metadata"] = visible["metadata_json"].map(decode_json)
        st.dataframe(
            visible[["metric_name", "value", "unit", "scoring_metadata"]],
            hide_index=True,
            width="stretch",
        )


def render_prompt_comparison(*, db_path: Path | None) -> None:
    st.title("Prompt Comparison")
    st.caption("Compare prompt strategy versions with the same model and the same dataset example.")
    render_workflow(3)
    runs = load_runs(db_path)
    if runs.empty:
        st.info("No framework evaluation runs exist yet. Run at least two prompt strategies to compare them here.")
        switch_page_button(
            "Create an evaluation run",
            "pages/1_Run_Evaluation.py",
            primary=True,
            key="empty-comparison-create-run",
        )
        return
    run_id = _selected_run(runs, key="comparison-run")
    results, metrics, enriched = _framework_data(db_path, run_id)
    if enriched.empty:
        st.info("This run has no persisted results.")
        return

    model_names = sorted(enriched["model_name"].dropna().unique().tolist())
    model_name = st.selectbox("Model", model_names)
    model_rows = enriched[enriched["model_name"] == model_name].copy()
    model_rows["strategy_key"] = (
        model_rows["prompt_strategy_id"] + "@" + model_rows["prompt_strategy_version"].astype(str)
    )
    strategy_labels = {
        row["strategy_key"]: f"{row['prompt_strategy_name']} · v{row['prompt_strategy_version']}"
        for _, row in model_rows.drop_duplicates("strategy_key").iterrows()
    }
    strategy_keys = sorted(strategy_labels)
    selected = st.multiselect(
        "Prompt strategies",
        strategy_keys,
        default=strategy_keys[: min(2, len(strategy_keys))],
        format_func=lambda value: strategy_labels[value],
    )
    if len(selected) < 2:
        st.warning("Select at least two prompt strategies.")
        return
    comparison_rows = model_rows[model_rows["strategy_key"].isin(selected)]
    counts = comparison_rows.groupby("record_id")["strategy_key"].nunique()
    record_ids = sorted(counts[counts == len(selected)].index.tolist())
    if not record_ids:
        st.warning("No dataset example has results for every selected prompt strategy.")
        return
    record_id = st.selectbox("Dataset example", record_ids)
    example = comparison_rows[comparison_rows["record_id"] == record_id].sort_values("strategy_key")
    first = example.iloc[0]
    st.markdown("#### Shared input")
    st.write(decode_json(first["raw_input_json"]))
    with st.expander("Reference"):
        st.write(decode_json(first["reference_json"]))

    columns = st.columns(len(example))
    for column, (_, row) in zip(columns, example.iterrows()):
        with column:
            st.markdown(f"### {row['prompt_strategy_name']}")
            st.caption(
                f"v{row['prompt_strategy_version']} · {row['end_to_end_seconds']:.3f}s · "
                f"{int(row['total_tokens'] or 0)} tokens · ${float(row['cost_usd'] or 0):.6f}"
            )
            st.write(row["raw_output"])
            with st.expander("Rendered prompt"):
                st.markdown("**System**")
                st.code(row["system_prompt_snapshot"] or "(empty)")
                st.markdown("**User**")
                st.code(row["user_prompt_snapshot"])

    quality_options = quality_metric_options(
        metrics[metrics["result_id"].isin(example["result_id"])]
    )
    selected_quality = st.selectbox("Quality metric for delta", quality_options) if quality_options else None
    comparison_metrics = [
        "cost_usd",
        "end_to_end_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "effective_output_tokens_per_second",
    ]
    if selected_quality:
        comparison_metrics.append(selected_quality)
    available = [name for name in comparison_metrics if name in example.columns]
    table = example[["strategy_key", "prompt_strategy_name", *available]].copy()
    baseline = table.iloc[0]
    for metric in available:
        if pd.api.types.is_numeric_dtype(table[metric]):
            table[f"Δ {metric}"] = table[metric] - baseline[metric]
    st.subheader("Metric and cost deltas")
    st.caption(f"Baseline: {baseline['prompt_strategy_name']} ({baseline['strategy_key']})")
    st.dataframe(table, hide_index=True, width="stretch")
