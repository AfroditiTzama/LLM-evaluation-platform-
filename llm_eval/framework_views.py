from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from database import connect


OPERATIONAL_METRICS = {
    "success",
    "end_to_end_latency",
    "provider_request_latency",
    "time_to_first_token",
    "generation_time",
    "inter_token_latency",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost",
    "effective_output_tokens_per_second",
    "generation_output_tokens_per_second",
}


def query_frame(
    db_path: Path | None,
    query: str,
    params: tuple[Any, ...] = (),
) -> pd.DataFrame:
    with connect(db_path, readonly=True) as con:
        return pd.read_sql_query(query, con, params=params)


def load_runs(db_path: Path | None) -> pd.DataFrame:
    return query_frame(
        db_path,
        """
        SELECT r.run_id,r.status,r.created_at,r.started_at,r.ended_at,r.task_id,
               t.name AS task_name,r.dataset_id,r.dataset_version,d.name AS dataset_name,
               r.evaluator_type,r.evaluator_version,r.generation_settings_json,
               r.metric_configuration_json,r.metadata_json
        FROM evaluation_runs r
        LEFT JOIN tasks t ON t.task_id=r.task_id
        LEFT JOIN datasets d ON d.dataset_id=r.dataset_id AND d.version=r.dataset_version
        ORDER BY r.created_at DESC
        """,
    )


def load_results(db_path: Path | None, run_id: str) -> pd.DataFrame:
    return query_frame(
        db_path,
        """
        SELECT er.*,m.name AS model_name,ps.name AS prompt_strategy_name,
               dr.difficulty,dr.domain,dr.language
        FROM evaluation_results er
        LEFT JOIN models m ON m.model_key=er.model_key
        LEFT JOIN prompt_strategies ps
          ON ps.strategy_id=er.prompt_strategy_id AND ps.version=er.prompt_strategy_version
        LEFT JOIN dataset_records dr
          ON dr.dataset_id=er.dataset_id AND dr.dataset_version=er.dataset_version
         AND dr.record_id=er.record_id
        WHERE er.run_id=?
        ORDER BY er.record_id,er.model_key,er.prompt_strategy_id,er.prompt_strategy_version
        """,
        (run_id,),
    )


def load_metric_values(db_path: Path | None, run_id: str) -> pd.DataFrame:
    frame = query_frame(
        db_path,
        """
        SELECT mv.result_id,mv.metric_name,mv.value_real,mv.value_text,
               mv.value_type,mv.unit,mv.metadata_json
        FROM metric_values mv
        JOIN evaluation_results er ON er.result_id=mv.result_id
        WHERE er.run_id=?
        ORDER BY mv.result_id,mv.metric_name
        """,
        (run_id,),
    )
    if not frame.empty:
        frame["value"] = frame.apply(
            lambda row: row["value_text"] if row["value_type"] == "text" else row["value_real"],
            axis=1,
        )
    return frame


def enrich_results(results: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if results.empty or metrics.empty:
        return results.copy()
    numeric = metrics[metrics["value_type"].isin(["number", "boolean"])].copy()
    if numeric.empty:
        return results.copy()
    pivot = numeric.pivot_table(
        index="result_id",
        columns="metric_name",
        values="value_real",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return results.merge(pivot, on="result_id", how="left", suffixes=("", "_metric"))


def quality_metric_options(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty:
        return []
    numeric = metrics[metrics["value_type"].isin(["number", "boolean"])]
    return sorted(set(numeric["metric_name"].tolist()) - OPERATIONAL_METRICS)


def build_leaderboard(
    enriched: pd.DataFrame,
    *,
    quality_metric: str | None,
) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()
    keys = ["model_key", "model_name", "prompt_strategy_id", "prompt_strategy_version", "prompt_strategy_name"]
    rows: list[dict[str, Any]] = []
    for key, group in enriched.groupby(keys, dropna=False):
        row: dict[str, Any] = dict(zip(keys, key))
        row["results"] = len(group)
        row["success_rate"] = (group["status"] == "success").mean()
        row["cost_total_usd"] = group["cost_usd"].sum(min_count=1)
        row["latency_mean_seconds"] = group["end_to_end_seconds"].mean()
        row["latency_median_seconds"] = group["end_to_end_seconds"].median()
        row["input_tokens_total"] = group["input_tokens"].sum(min_count=1)
        row["output_tokens_total"] = group["output_tokens"].sum(min_count=1)
        if "effective_output_tokens_per_second" in group:
            row["effective_tokens_per_second_mean"] = group[
                "effective_output_tokens_per_second"
            ].mean()
        if "generation_output_tokens_per_second" in group:
            row["generation_tokens_per_second_mean"] = group[
                "generation_output_tokens_per_second"
            ].mean()
        if quality_metric and quality_metric in group:
            values = group[quality_metric].dropna()
            row["quality_metric"] = quality_metric
            row["quality_mean"] = values.mean() if not values.empty else None
            row["quality_median"] = values.median() if not values.empty else None
            row["quality_std"] = values.std(ddof=1) if len(values) > 1 else (0.0 if len(values) == 1 else None)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["model_name", "prompt_strategy_name", "prompt_strategy_version"],
        kind="stable",
    )


def mark_pareto_frontier(leaderboard: pd.DataFrame) -> pd.DataFrame:
    output = leaderboard.copy()
    output["pareto_optimal"] = False
    if output.empty or "quality_mean" not in output:
        return output
    eligible = output.dropna(subset=["quality_mean", "cost_total_usd"])
    for index, candidate in eligible.iterrows():
        dominated = False
        for other_index, other in eligible.iterrows():
            if index == other_index:
                continue
            no_worse = (
                other["quality_mean"] >= candidate["quality_mean"]
                and other["cost_total_usd"] <= candidate["cost_total_usd"]
            )
            strictly_better = (
                other["quality_mean"] > candidate["quality_mean"]
                or other["cost_total_usd"] < candidate["cost_total_usd"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        output.loc[index, "pareto_optimal"] = not dominated
    return output


def decode_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
