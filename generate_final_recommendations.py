from pathlib import Path

import pandas as pd


RECOMMENDATIONS_OVERALL_FILE = "recommendations_overall.csv"
RECOMMENDATIONS_USECASE_FILE = "recommendations_by_usecase.csv"

JUDGE_MODEL_FILE = "judge_summary_by_model.csv"
JUDGE_USECASE_FILE = "judge_summary_by_model_usecase.csv"

OUT_OVERALL = "final_recommendations_overall.csv"
OUT_USECASE = "final_recommendations_by_usecase.csv"
OUT_SUMMARY = "final_recommendation_summary.txt"


def normalize_lower_is_better(series):
    series = pd.to_numeric(series, errors="coerce")

    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([0.0] * len(series), index=series.index)

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1 - ((series - min_val) / (max_val - min_val))


def normalize_higher_is_better(series):
    series = pd.to_numeric(series, errors="coerce")

    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([0.0] * len(series), index=series.index)

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


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


def require_files(files):
    for file in files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Cannot find {file}")


def prepare_overall():
    rec = pd.read_csv(RECOMMENDATIONS_OVERALL_FILE)
    judge = pd.read_csv(JUDGE_MODEL_FILE)

    df = rec.merge(
        judge[
            [
                "model",
                "rows",
                "avg_correctness",
                "avg_completeness",
                "avg_instruction_following",
                "avg_hallucination_safety",
                "avg_format_quality",
                "avg_overall_quality",
            ]
        ],
        on="model",
        how="left",
        suffixes=("", "_judge"),
    )

    # Quality is already 0-10, but we normalize it across models for final ranking.
    df["quality_score"] = normalize_higher_is_better(df["avg_overall_quality"])

    # Recalculate the core normalized scores from raw metrics.
    df["latency_score_final"] = normalize_lower_is_better(df["avg_latency"])
    df["cost_score_final"] = normalize_lower_is_better(df["estimated_cost_per_1000_prompts_usd"])
    df["failure_score_final"] = normalize_lower_is_better(df["failure_rate"])
    df["truncation_score_final"] = normalize_lower_is_better(df["possible_truncations"])

    # Final balanced score.
    # Quality matters most, but operational factors still matter.
    df["final_score"] = (
        0.40 * df["quality_score"]
        + 0.20 * df["failure_score_final"]
        + 0.15 * df["latency_score_final"]
        + 0.15 * df["cost_score_final"]
        + 0.10 * df["truncation_score_final"]
    )

    df["final_rank"] = df["final_score"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    df["model_label"] = df["model"].apply(short_model_name)

    df = df.sort_values("final_rank")

    return df


def prepare_usecase():
    rec = pd.read_csv(RECOMMENDATIONS_USECASE_FILE)
    judge = pd.read_csv(JUDGE_USECASE_FILE)

    df = rec.merge(
        judge[
            [
                "model",
                "use_case",
                "rows",
                "avg_correctness",
                "avg_completeness",
                "avg_instruction_following",
                "avg_hallucination_safety",
                "avg_format_quality",
                "avg_overall_quality",
            ]
        ],
        on=["model", "use_case"],
        how="left",
        suffixes=("", "_judge"),
    )

    scored_groups = []

    for use_case, group in df.groupby("use_case"):
        group = group.copy()

        group["quality_score"] = normalize_higher_is_better(group["avg_overall_quality"])
        group["latency_score_final"] = normalize_lower_is_better(group["avg_latency"])
        group["cost_score_final"] = normalize_lower_is_better(group["estimated_cost_per_1000_prompts_usd"])
        group["failure_score_final"] = normalize_lower_is_better(group["failure_rate"])
        group["truncation_score_final"] = normalize_lower_is_better(group["possible_truncations"])

        if use_case == "customer_support":
            # Customer support needs quality, reliability and speed.
            group["final_score"] = (
                0.35 * group["quality_score"]
                + 0.25 * group["failure_score_final"]
                + 0.20 * group["latency_score_final"]
                + 0.15 * group["cost_score_final"]
                + 0.05 * group["truncation_score_final"]
            )

        elif use_case == "document_understanding":
            # Document understanding needs correctness and low hallucination risk.
            group["final_score"] = (
                0.40 * group["quality_score"]
                + 0.25 * group["failure_score_final"]
                + 0.15 * group["latency_score_final"]
                + 0.10 * group["cost_score_final"]
                + 0.10 * group["truncation_score_final"]
            )

        elif use_case == "website_generation":
            # Website generation needs quality and complete outputs.
            group["final_score"] = (
                0.40 * group["quality_score"]
                + 0.20 * group["truncation_score_final"]
                + 0.15 * group["failure_score_final"]
                + 0.15 * group["cost_score_final"]
                + 0.10 * group["latency_score_final"]
            )

        else:
            group["final_score"] = (
                0.40 * group["quality_score"]
                + 0.20 * group["failure_score_final"]
                + 0.15 * group["latency_score_final"]
                + 0.15 * group["cost_score_final"]
                + 0.10 * group["truncation_score_final"]
            )

        group["final_rank"] = group["final_score"].rank(
            ascending=False,
            method="dense"
        ).astype(int)

        scored_groups.append(group)

    final = pd.concat(scored_groups, ignore_index=True)
    final["model_label"] = final["model"].apply(short_model_name)

    final = final.sort_values(["use_case", "final_rank"])

    return final


def create_summary(overall, usecase):
    lines = []

    lines.append("Final LLM Evaluation Recommendation Summary")
    lines.append("=" * 70)
    lines.append("")
    lines.append("This final recommendation combines:")
    lines.append("- LLM-as-a-judge quality score")
    lines.append("- latency")
    lines.append("- estimated cost")
    lines.append("- heuristic failure rate")
    lines.append("- possible truncations")
    lines.append("")
    lines.append("The quality scores are based on the balanced judged sample, not the full 1080 rows unless the judge was run on all rows.")
    lines.append("")

    best_overall = overall.sort_values("final_rank").iloc[0]
    best_quality = overall.sort_values("avg_overall_quality", ascending=False).iloc[0]
    cheapest = overall.sort_values("estimated_cost_per_1000_prompts_usd").iloc[0]
    fastest = overall.sort_values("avg_latency").iloc[0]
    lowest_failure = overall.sort_values("failure_rate").iloc[0]

    lines.append("Overall recommendation")
    lines.append("-" * 70)
    lines.append(f"Best balanced model: {best_overall['model_label']}")
    lines.append(f"Final score: {best_overall['final_score']:.4f}")
    lines.append(f"Judge quality: {best_overall['avg_overall_quality']:.3f}/10")
    lines.append(f"Average latency: {best_overall['avg_latency']:.3f}s")
    lines.append(f"Estimated cost / 1000 prompts: ${best_overall['estimated_cost_per_1000_prompts_usd']:.4f}")
    lines.append(f"Failure rate: {best_overall['failure_rate']:.3f}")
    lines.append("")

    lines.append("Specialized winners")
    lines.append("-" * 70)
    lines.append(f"Best quality: {best_quality['model_label']} ({best_quality['avg_overall_quality']:.3f}/10)")
    lines.append(f"Lowest cost: {cheapest['model_label']} (${cheapest['estimated_cost_per_1000_prompts_usd']:.4f} / 1000 prompts)")
    lines.append(f"Fastest: {fastest['model_label']} ({fastest['avg_latency']:.3f}s average latency)")
    lines.append(f"Lowest failure rate: {lowest_failure['model_label']} ({lowest_failure['failure_rate']:.3f})")
    lines.append("")

    lines.append("Recommended model by use case")
    lines.append("-" * 70)

    for use_case in sorted(usecase["use_case"].unique()):
        subset = usecase[usecase["use_case"] == use_case].sort_values("final_rank")
        best = subset.iloc[0]

        lines.append(f"{use_case}: {best['model_label']}")
        lines.append(f"  Final score: {best['final_score']:.4f}")
        lines.append(f"  Judge quality: {best['avg_overall_quality']:.3f}/10")
        lines.append(f"  Latency: {best['avg_latency']:.3f}s")
        lines.append(f"  Cost / 1000 prompts: ${best['estimated_cost_per_1000_prompts_usd']:.4f}")
        lines.append(f"  Failure rate: {best['failure_rate']:.3f}")
        lines.append("")

    lines.append("Interpretation")
    lines.append("-" * 70)
    lines.append("The best strategy is not necessarily one model for all tasks.")
    lines.append("A model-routing setup can select a stronger model for quality-sensitive tasks and a cheaper/faster model for simpler tasks.")
    lines.append("The final production choice should also consider provider stability, rate limits, and more human evaluation if available.")

    return "\n".join(lines)


def main():
    require_files([
        RECOMMENDATIONS_OVERALL_FILE,
        RECOMMENDATIONS_USECASE_FILE,
        JUDGE_MODEL_FILE,
        JUDGE_USECASE_FILE,
    ])

    overall = prepare_overall()
    usecase = prepare_usecase()

    overall.to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")
    usecase.to_csv(OUT_USECASE, index=False, encoding="utf-8-sig")

    summary = create_summary(overall, usecase)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary)

    print("Done.")
    print(f"Created: {OUT_OVERALL}")
    print(f"Created: {OUT_USECASE}")
    print(f"Created: {OUT_SUMMARY}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
