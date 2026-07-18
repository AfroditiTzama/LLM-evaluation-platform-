from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation import deterministic_evaluation

SCORE_FIELDS = [
    "correctness",
    "instruction_following",
    "factuality_grounding",
    "greek_quality",
    "overall_quality",
]


def number_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return round(ordered[lo], 4)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return round(value, 4)


def mean_ci95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        value = round(values[0], 4)
        return value, value
    center = statistics.fmean(values)
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return round(center - margin, 4), round(center + margin, 4)


def wilson_ci(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.96
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recalculate_results(
    results: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompts_by_id = {str(prompt["prompt_id"]): prompt for prompt in prompts}
    corrected: list[dict[str, Any]] = []
    for source_row in results:
        row = dict(source_row)
        prompt = prompts_by_id.get(str(row.get("prompt_id")))
        if prompt and row.get("status") == "success":
            row.update(deterministic_evaluation(prompt, str(row.get("response", ""))))
            row["evaluation_type"] = prompt.get("evaluation_type", row.get("evaluation_type", ""))
            row["expected_format"] = prompt.get("expected_format", row.get("expected_format", ""))
            row["benchmark_version"] = "1.1"
        corrected.append(row)
    return corrected


def effective_winner(judgment: dict[str, Any]) -> tuple[str, str]:
    raw = str(judgment.get("winner_label") or judgment.get("winner") or "")
    if raw != "cannot_assess":
        return raw, "raw"

    answer_a = judgment.get("answer_a") or {}
    answer_b = judgment.get("answer_b") or {}
    success_a = answer_a.get("task_success")
    success_b = answer_b.get("task_success")
    if success_a is not None and success_b is not None and bool(success_a) == bool(success_b):
        return "tie", "cannot_assess_normalized_to_tie_equal_task_success"
    return raw, "raw_cannot_assess"


def flatten_judgments(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for judgment in judgments:
        if judgment.get("status") != "success":
            rows.append({
                "run_id": judgment.get("run_id"),
                "prompt_id": judgment.get("prompt_id"),
                "status": "error",
                "error": judgment.get("error", ""),
                "judge_model": judgment.get("judge_model", ""),
            })
            continue

        raw_winner = str(judgment.get("winner_label") or judgment.get("winner") or "")
        effective, normalization_reason = effective_winner(judgment)
        model_a = str(judgment.get("answer_a_model_name", ""))
        model_b = str(judgment.get("answer_b_model_name", ""))
        effective_model = model_a if effective == "A" else model_b if effective == "B" else effective

        for label, model_name, model_id in (
            ("A", model_a, judgment.get("answer_a_model_id", "")),
            ("B", model_b, judgment.get("answer_b_model_id", "")),
        ):
            scores = judgment.get(f"answer_{label.lower()}") or {}
            row = {
                "run_id": judgment.get("run_id"),
                "prompt_id": judgment.get("prompt_id"),
                "category": judgment.get("category"),
                "difficulty": judgment.get("difficulty"),
                "status": "success",
                "model_name": model_name,
                "model_id": model_id,
                "blind_answer_label": label,
                "winner_label_raw": raw_winner,
                "winner_label_effective": effective,
                "winner_model_name_effective": effective_model,
                "winner_normalization_reason": normalization_reason,
                "is_winner": float(effective_model == model_name),
                "is_tie": float(effective == "tie"),
                "is_cannot_assess": float(effective == "cannot_assess"),
                "judge_confidence": judgment.get("confidence"),
                "hallucination_detected": scores.get("hallucination_detected"),
                "task_success": scores.get("task_success"),
                "critical_error": scores.get("critical_error", ""),
                "rationale": scores.get("rationale", ""),
                "pairwise_reason": judgment.get("pairwise_reason", ""),
                "judge_model": judgment.get("judge_model", ""),
                "judge_latency_seconds": judgment.get("judge_latency_seconds"),
                "judge_input_tokens": judgment.get("judge_input_tokens"),
                "judge_output_tokens": judgment.get("judge_output_tokens"),
                "judge_total_tokens": judgment.get("judge_total_tokens"),
                "judge_cost_usd": judgment.get("judge_cost_usd"),
                "judge_provider": judgment.get("judge_provider", ""),
                "timestamp": judgment.get("timestamp", ""),
            }
            for field in SCORE_FIELDS:
                row[field] = scores.get(field)
            rows.append(row)
    return rows


def _rates(rows: list[dict[str, Any]], key: str) -> tuple[int, float | None]:
    values = [number_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return len(values), mean(values)


def create_model_summary(
    run_id: str,
    results: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_names = sorted({str(row.get("model_name")) for row in results if row.get("model_name")})
    summaries: list[dict[str, Any]] = []
    for model_name in model_names:
        model_rows = [row for row in results if row.get("model_name") == model_name]
        successful = [row for row in model_rows if row.get("status") == "success"]
        judged = [row for row in judge_rows if row.get("model_name") == model_name and row.get("status") == "success"]
        latencies = [float(row["latency_seconds"]) for row in successful if number_or_none(row.get("latency_seconds")) is not None]
        costs = [float(row["cost_usd"]) for row in successful if number_or_none(row.get("cost_usd")) is not None]

        deterministic_n, deterministic_rate = _rates(model_rows, "deterministic_pass")
        strict_n, strict_rate = _rates(model_rows, "strict_exact_match")
        normalized_n, normalized_rate = _rates(model_rows, "normalized_exact_match")
        numeric_n, numeric_rate = _rates(model_rows, "numeric_match")
        structured_n, syntax_rate = _rates(model_rows, "syntax_valid")
        _, schema_rate = _rates(model_rows, "schema_valid")
        _, format_rate = _rates(model_rows, "format_compliance")

        task_success_values = [bool(row.get("task_success")) for row in judged if row.get("task_success") is not None]
        hallucination_values = [bool(row.get("hallucination_detected")) for row in judged if row.get("hallucination_detected") is not None]
        overall_values = [float(row["overall_quality"]) for row in judged if number_or_none(row.get("overall_quality")) is not None]
        wins = int(sum(float(row.get("is_winner") or 0) for row in judged))
        ties = int(sum(float(row.get("is_tie") or 0) for row in judged))
        cannot = int(sum(float(row.get("is_cannot_assess") or 0) for row in judged))
        # Each pair appears once for this model, so decisive comparisons are judged rows minus ties/cannot.
        decisive = max(0, len(judged) - ties - cannot)
        win_ci = wilson_ci(wins, decisive)
        task_ci = wilson_ci(sum(task_success_values), len(task_success_values))
        quality_ci = mean_ci95(overall_values)

        model_id = next((str(row.get("model_id")) for row in model_rows if row.get("model_id")), "")
        summary: dict[str, Any] = {
            "run_id": run_id,
            "model_name": model_name,
            "model_id": model_id,
            "request_count": len(model_rows),
            "success_count": len(successful),
            "api_success_rate": round(len(successful) / len(model_rows), 4) if model_rows else None,
            "latency_mean_seconds": mean(latencies),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "total_cost_usd": round(sum(costs), 8),
            "total_input_tokens": sum(int(float(row.get("input_tokens") or 0)) for row in model_rows),
            "total_output_tokens": sum(int(float(row.get("output_tokens") or 0)) for row in model_rows),
            "total_tokens": sum(int(float(row.get("total_tokens") or 0)) for row in model_rows),
            "total_retries": sum(int(float(row.get("retry_count") or 0)) for row in model_rows),
            "deterministic_evaluated": deterministic_n,
            "deterministic_pass_rate": deterministic_rate,
            "strict_exact_evaluated": strict_n,
            "strict_exact_rate": strict_rate,
            "normalized_exact_evaluated": normalized_n,
            "normalized_exact_rate": normalized_rate,
            "numeric_evaluated": numeric_n,
            "numeric_accuracy": numeric_rate,
            "structured_evaluated": structured_n,
            "structured_syntax_validity_rate": syntax_rate,
            "structured_schema_validity_rate": schema_rate,
            "structured_format_compliance_rate": format_rate,
            "judge_evaluated": len(judged),
            "judge_win_count": wins,
            "judge_tie_count": ties,
            "judge_cannot_assess_count": cannot,
            "judge_decisive_count": decisive,
            "judge_win_rate_decisive": round(wins / decisive, 4) if decisive else None,
            "judge_win_rate_ci95_low": win_ci[0],
            "judge_win_rate_ci95_high": win_ci[1],
            "judge_task_success_rate": round(sum(task_success_values) / len(task_success_values), 4) if task_success_values else None,
            "judge_task_success_ci95_low": task_ci[0],
            "judge_task_success_ci95_high": task_ci[1],
            "judge_hallucination_rate": round(sum(hallucination_values) / len(hallucination_values), 4) if hallucination_values else None,
            "judge_overall_quality_ci95_low": quality_ci[0],
            "judge_overall_quality_ci95_high": quality_ci[1],
        }
        for field in SCORE_FIELDS:
            values = [float(row[field]) for row in judged if number_or_none(row.get(field)) is not None]
            summary[f"judge_{field}_mean"] = mean(values)
        summaries.append(summary)
    return summaries


def create_group_summary(
    run_id: str,
    results: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    keys = sorted({(str(row.get("model_name")), str(row.get(group_key))) for row in results})
    output: list[dict[str, Any]] = []
    for model_name, group_value in keys:
        result_subset = [row for row in results if str(row.get("model_name")) == model_name and str(row.get(group_key)) == group_value]
        judge_subset = [row for row in judge_rows if str(row.get("model_name")) == model_name and str(row.get(group_key)) == group_value and row.get("status") == "success"]
        successful = [row for row in result_subset if row.get("status") == "success"]
        latencies = [float(row["latency_seconds"]) for row in successful if number_or_none(row.get("latency_seconds")) is not None]
        costs = [float(row["cost_usd"]) for row in successful if number_or_none(row.get("cost_usd")) is not None]
        deterministic_n, deterministic_rate = _rates(result_subset, "deterministic_pass")
        structured_n, format_rate = _rates(result_subset, "format_compliance")
        task_values = [bool(row.get("task_success")) for row in judge_subset if row.get("task_success") is not None]
        hallucination_values = [bool(row.get("hallucination_detected")) for row in judge_subset if row.get("hallucination_detected") is not None]
        row: dict[str, Any] = {
            "run_id": run_id,
            "model_name": model_name,
            group_key: group_value,
            "prompts": len(result_subset),
            "success_rate": round(len(successful) / len(result_subset), 4) if result_subset else None,
            "latency_mean_seconds": mean(latencies),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "total_cost_usd": round(sum(costs), 8),
            "deterministic_evaluated": deterministic_n,
            "deterministic_pass_rate": deterministic_rate,
            "structured_evaluated": structured_n,
            "structured_format_compliance_rate": format_rate,
            "judge_evaluated": len(judge_subset),
            "win_count": int(sum(float(item.get("is_winner") or 0) for item in judge_subset)),
            "tie_count": int(sum(float(item.get("is_tie") or 0) for item in judge_subset)),
            "task_success_rate": round(sum(task_values) / len(task_values), 4) if task_values else None,
            "hallucination_rate": round(sum(hallucination_values) / len(hallucination_values), 4) if hallucination_values else None,
        }
        for field in SCORE_FIELDS:
            values = [float(item[field]) for item in judge_subset if number_or_none(item.get(field)) is not None]
            row[f"{field}_mean"] = mean(values)
        output.append(row)
    return output


def create_provider_summary(run_id: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(str(row.get("model_name")), str(row.get("provider"))) for row in results})
    output: list[dict[str, Any]] = []
    for model_name, provider in keys:
        subset = [row for row in results if str(row.get("model_name")) == model_name and str(row.get("provider")) == provider]
        successful = [row for row in subset if row.get("status") == "success"]
        latencies = [float(row["latency_seconds"]) for row in successful if number_or_none(row.get("latency_seconds")) is not None]
        costs = [float(row["cost_usd"]) for row in successful if number_or_none(row.get("cost_usd")) is not None]
        output.append({
            "run_id": run_id,
            "model_name": model_name,
            "provider": provider,
            "requests": len(subset),
            "success_rate": round(len(successful) / len(subset), 4) if subset else None,
            "latency_mean_seconds": mean(latencies),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "total_cost_usd": round(sum(costs), 8),
        })
    return output


def create_stratified_human_sample(
    run_id: str,
    prompts: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts_by_id = {str(prompt["prompt_id"]): prompt for prompt in prompts}
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("status") == "success":
            by_prompt[str(row.get("prompt_id"))].append(row)

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        if len(by_prompt.get(prompt_id, [])) == 2:
            grouped[(str(prompt["category"]), str(prompt["difficulty"]))].append(prompt_id)

    rng = random.Random(f"{run_id}-stratified-v2")
    selected: list[str] = []
    for key in sorted(grouped):
        candidates = grouped[key][:]
        rng.shuffle(candidates)
        if candidates:
            selected.append(candidates[0])

    # Exactly 10 categories x 3 difficulties = 30 prompts for this dataset.
    rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    qwen_as_a_target = len(selected) // 2
    assignment_flags = [True] * qwen_as_a_target + [False] * (len(selected) - qwen_as_a_target)
    rng.shuffle(assignment_flags)

    for prompt_id, qwen_as_a in zip(selected, assignment_flags):
        pair = by_prompt[prompt_id]
        qwen = next(row for row in pair if str(row.get("model_name", "")).startswith("Qwen"))
        gemma = next(row for row in pair if str(row.get("model_name", "")).startswith("Gemma"))
        answer_a, answer_b = (qwen, gemma) if qwen_as_a else (gemma, qwen)
        prompt = prompts_by_id[prompt_id]
        rows.append({
            "prompt_id": prompt_id,
            "category": prompt["category"],
            "difficulty": prompt["difficulty"],
            "prompt": prompt["prompt"],
            "answer_a": answer_a["response"],
            "answer_b": answer_b["response"],
            "human_winner_A_B_tie": "",
            "correctness_a_1_to_5": "",
            "correctness_b_1_to_5": "",
            "instruction_following_a_1_to_5": "",
            "instruction_following_b_1_to_5": "",
            "factuality_a_1_to_5": "",
            "factuality_b_1_to_5": "",
            "greek_quality_a_1_to_5": "",
            "greek_quality_b_1_to_5": "",
            "comments": "",
        })
        key_rows.append({
            "prompt_id": prompt_id,
            "answer_a_model": answer_a["model_name"],
            "answer_b_model": answer_b["model_name"],
        })
    return rows, key_rows


def rebuild_artifacts(
    run_id: str,
    prompts: list[dict[str, Any]],
    results: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    corrected_results = recalculate_results(results, prompts)
    judge_rows = flatten_judgments(judgments)
    model_summary = create_model_summary(run_id, corrected_results, judge_rows)
    category_summary = create_group_summary(run_id, corrected_results, judge_rows, "category")
    difficulty_summary = create_group_summary(run_id, corrected_results, judge_rows, "difficulty")
    provider_summary = create_provider_summary(run_id, corrected_results)
    human_rows, human_key = create_stratified_human_sample(run_id, prompts, corrected_results)

    paths = {
        "results_json": output_dir / f"{run_id}_results_v2.json",
        "results_csv": output_dir / f"{run_id}_results_v2.csv",
        "judge_json": output_dir / f"{run_id}_judge_scores_v2.json",
        "judge_csv": output_dir / f"{run_id}_judge_scores_v2.csv",
        "model_json": output_dir / f"{run_id}_model_summary_v2.json",
        "model_csv": output_dir / f"{run_id}_model_summary_v2.csv",
        "category_json": output_dir / f"{run_id}_category_summary_v2.json",
        "category_csv": output_dir / f"{run_id}_category_summary_v2.csv",
        "difficulty_json": output_dir / f"{run_id}_difficulty_summary_v2.json",
        "difficulty_csv": output_dir / f"{run_id}_difficulty_summary_v2.csv",
        "provider_json": output_dir / f"{run_id}_provider_summary_v2.json",
        "provider_csv": output_dir / f"{run_id}_provider_summary_v2.csv",
        "human_csv": output_dir / f"{run_id}_blind_human_sample_v2.csv",
        "human_key": output_dir / f"{run_id}_blind_human_sample_key_v2.json",
    }
    write_json(paths["results_json"], corrected_results)
    write_csv(paths["results_csv"], corrected_results)
    write_json(paths["judge_json"], judge_rows)
    write_csv(paths["judge_csv"], judge_rows)
    write_json(paths["model_json"], model_summary)
    write_csv(paths["model_csv"], model_summary)
    write_json(paths["category_json"], category_summary)
    write_csv(paths["category_csv"], category_summary)
    write_json(paths["difficulty_json"], difficulty_summary)
    write_csv(paths["difficulty_csv"], difficulty_summary)
    write_json(paths["provider_json"], provider_summary)
    write_csv(paths["provider_csv"], provider_summary)
    write_csv(paths["human_csv"], human_rows)
    write_json(paths["human_key"], {
        "run_id": run_id,
        "warning": "Open only after completing the blinded human review.",
        "design": "30 prompts: one easy, one medium and one hard from each of 10 categories; balanced A/B assignment.",
        "answers": human_key,
    })
    return {
        "run_id": run_id,
        "corrected_results": corrected_results,
        "judge_rows": judge_rows,
        "model_summary": model_summary,
        "category_summary": category_summary,
        "difficulty_summary": difficulty_summary,
        "provider_summary": provider_summary,
        "human_rows": human_rows,
        "human_key": human_key,
        "paths": {key: str(value) for key, value in paths.items()},
    }
