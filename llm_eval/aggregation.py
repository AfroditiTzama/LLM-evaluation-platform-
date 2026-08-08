from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Sequence

from .domain import EvaluationResult


def _dimension(result: EvaluationResult, name: str) -> Any:
    if hasattr(result, name):
        return getattr(result, name)
    record = result.metadata.get("record", {})
    if name in {"difficulty", "domain", "language"}:
        return record.get(name, "")
    raise ValueError(f"Unsupported aggregation dimension: {name}")


def _numeric_metric_values(results: Iterable[EvaluationResult]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for metric in result.metrics:
            value = metric.value
            if isinstance(value, bool):
                values[metric.name].append(float(value))
            elif isinstance(value, (int, float)):
                values[metric.name].append(float(value))
    return values


def aggregate_results(
    results: Sequence[EvaluationResult],
    *,
    group_by: tuple[str, ...] = ("model_key", "task_id", "prompt_strategy_id", "prompt_strategy_version"),
) -> list[dict[str, Any]]:
    """Aggregate every numeric metric without inventing a universal score."""
    grouped: dict[tuple[Any, ...], list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[tuple(_dimension(result, name) for name in group_by)].append(result)

    rows: list[dict[str, Any]] = []
    for key, subset in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row: dict[str, Any] = dict(zip(group_by, key))
        row["result_count"] = len(subset)
        row["success_rate"] = sum(item.status == "success" for item in subset) / len(subset)
        for metric_name, values in sorted(_numeric_metric_values(subset).items()):
            row[f"{metric_name}_count"] = len(values)
            row[f"{metric_name}_mean"] = statistics.fmean(values)
            row[f"{metric_name}_median"] = statistics.median(values)
            row[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric_name}_sum"] = sum(values)
        rows.append(row)
    return rows


def pareto_frontier(
    rows: Sequence[dict[str, Any]],
    *,
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return non-dominated rows for explicitly selected metrics.

    Missing metrics are excluded instead of being treated as zero.
    """
    metrics = (*maximize, *minimize)
    eligible = [row for row in rows if all(isinstance(row.get(name), (int, float)) for name in metrics)]

    def dominates(first: dict[str, Any], second: dict[str, Any]) -> bool:
        no_worse = all(first[name] >= second[name] for name in maximize) and all(
            first[name] <= second[name] for name in minimize
        )
        strictly_better = any(first[name] > second[name] for name in maximize) or any(
            first[name] < second[name] for name in minimize
        )
        return no_worse and strictly_better

    return [
        row
        for row in eligible
        if not any(other is not row and dominates(other, row) for other in eligible)
    ]
