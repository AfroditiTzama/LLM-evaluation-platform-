import pandas as pd
from pathlib import Path

METRICS_USECASE_FILE = "metrics_by_model_usecase.csv"
FAILURES_USECASE_FILE = "failure_summary_by_model_usecase.csv"
COST_USECASE_FILE = "cost_analysis_by_model_usecase.csv"

METRICS_MODEL_FILE = "metrics_by_model.csv"
FAILURES_MODEL_FILE = "failure_summary_by_model.csv"
COST_MODEL_FILE = "cost_analysis_by_model.csv"

OUT_USECASE = "recommendations_by_usecase.csv"
OUT_OVERALL = "recommendations_overall.csv"
OUT_TXT = "recommendation_summary.txt"


def normalize_lower_is_better(series):
    """
    Converts a metric where lower is better into a 0-1 score where higher is better.
    Best value gets 1. Worst value gets 0.
    """
    series = pd.to_numeric(series, errors="coerce")

    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([0.0] * len(series), index=series.index)

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1 - ((series - min_val) / (max_val - min_val))

def load_and_merge_usecase():
    metrics = pd.read_csv(METRICS_USECASE_FILE)
    failures = pd.read_csv(FAILURES_USECASE_FILE)
    costs = pd.read_csv(COST_USECASE_FILE)

    # Keep failure columns with clear names to avoid duplicate column suffixes after merge.
    failures = failures.rename(columns={
        "possible_truncations": "failure_possible_truncations"
    })

    df = metrics.merge(
        failures[
            [
                "model",
                "use_case",
                "failure_rate",
                "avg_failure_count",
                "high_latency_count",
                "failure_possible_truncations",
                "empty_answers",
            ]
        ],
        on=["model", "use_case"],
        how="left",
    )

    df = df.merge(
        costs[
            [
                "model",
                "use_case",
                "total_cost_usd",
                "avg_cost_per_prompt_usd",
                "estimated_cost_per_1000_prompts_usd",
            ]
        ],
        on=["model", "use_case"],
        how="left",
    )

    # Use the failure-summary truncation count for scoring.
    df["possible_truncations"] = df["failure_possible_truncations"].fillna(0)

    return df


def load_and_merge_overall():
    metrics = pd.read_csv(METRICS_MODEL_FILE)
    failures = pd.read_csv(FAILURES_MODEL_FILE)
    costs = pd.read_csv(COST_MODEL_FILE)

    # Keep failure columns with clear names to avoid duplicate column suffixes after merge.
    failures = failures.rename(columns={
        "possible_truncations": "failure_possible_truncations"
    })

    df = metrics.merge(
        failures[
            [
                "model",
                "failure_rate",
                "avg_failure_count",
                "high_latency_count",
                "failure_possible_truncations",
                "empty_answers",
            ]
        ],
        on="model",
        how="left",
    )

    df = df.merge(
        costs[
            [
                "model",
                "total_cost_usd",
                "avg_cost_per_prompt_usd",
                "estimated_cost_per_1000_prompts_usd",
            ]
        ],
        on="model",
        how="left",
    )

    # Use the failure-summary truncation count for scoring.
    df["possible_truncations"] = df["failure_possible_truncations"].fillna(0)

    return df


def score_usecase_group(group):
    group = group.copy()

    group["latency_score"] = normalize_lower_is_better(group["avg_latency"])
    group["cost_score"] = normalize_lower_is_better(group["avg_cost_per_prompt_usd"])
    group["failure_score"] = normalize_lower_is_better(group["failure_rate"])
    group["truncation_score"] = normalize_lower_is_better(group["possible_truncations"])

    use_case = group["use_case"].iloc[0]

    if use_case == "customer_support":
        # Customer support needs reliability and speed.
        group["recommendation_score"] = (
            0.35 * group["failure_score"]
            + 0.30 * group["latency_score"]
            + 0.25 * group["cost_score"]
            + 0.10 * group["truncation_score"]
        )

    elif use_case == "website_generation":
        # Website generation needs complete outputs, so truncation matters more.
        group["recommendation_score"] = (
            0.30 * group["failure_score"]
            + 0.20 * group["latency_score"]
            + 0.20 * group["cost_score"]
            + 0.30 * group["truncation_score"]
        )

    elif use_case == "document_understanding":
        # Document understanding needs reliability and fewer unsupported answers.
        group["recommendation_score"] = (
            0.40 * group["failure_score"]
            + 0.20 * group["latency_score"]
            + 0.20 * group["cost_score"]
            + 0.20 * group["truncation_score"]
        )

    else:
        group["recommendation_score"] = (
            0.30 * group["failure_score"]
            + 0.25 * group["latency_score"]
            + 0.25 * group["cost_score"]
            + 0.20 * group["truncation_score"]
        )

    group["rank"] = group["recommendation_score"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    return group.sort_values("rank")


def score_overall(df):
    df = df.copy()

    df["latency_score"] = normalize_lower_is_better(df["avg_latency"])
    df["cost_score"] = normalize_lower_is_better(df["avg_cost_per_prompt_usd"])
    df["failure_score"] = normalize_lower_is_better(df["failure_rate"])
    df["truncation_score"] = normalize_lower_is_better(df["possible_truncations"])

    df["overall_score"] = (
        0.30 * df["failure_score"]
        + 0.25 * df["latency_score"]
        + 0.25 * df["cost_score"]
        + 0.20 * df["truncation_score"]
    )

    df["overall_rank"] = df["overall_score"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    return df.sort_values("overall_rank")


def create_summary_text(usecase_df, overall_df):
    lines = []

    lines.append("LLM Evaluation Preliminary Recommendation Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("This recommendation is based on latency, token cost, heuristic failure rate, and possible truncation rate.")
    lines.append("It does not yet include LLM-as-a-judge quality scoring or human evaluation.")
    lines.append("")

    for use_case in sorted(usecase_df["use_case"].unique()):
        subset = usecase_df[usecase_df["use_case"] == use_case].sort_values("rank")
        best = subset.iloc[0]

        lines.append(f"Use case: {use_case}")
        lines.append(f"Recommended model: {best['model']}")
        lines.append(f"Recommendation score: {best['recommendation_score']:.4f}")
        lines.append(f"Average latency: {best['avg_latency']:.3f}s")
        lines.append(f"Failure rate: {best['failure_rate']:.3f}")
        lines.append(f"Estimated cost / 1000 prompts: ${best['estimated_cost_per_1000_prompts_usd']:.4f}")
        lines.append("")

    best_overall = overall_df.sort_values("overall_rank").iloc[0]
    cheapest = overall_df.sort_values("estimated_cost_per_1000_prompts_usd").iloc[0]
    fastest = overall_df.sort_values("avg_latency").iloc[0]
    lowest_failure = overall_df.sort_values("failure_rate").iloc[0]

    lines.append("Overall")
    lines.append("-" * 60)
    lines.append(f"Best overall preliminary model: {best_overall['model']}")
    lines.append(f"Best overall score: {best_overall['overall_score']:.4f}")
    lines.append("")
    lines.append(f"Lowest cost model: {cheapest['model']} (${cheapest['estimated_cost_per_1000_prompts_usd']:.4f} / 1000 prompts)")
    lines.append(f"Fastest model: {fastest['model']} ({fastest['avg_latency']:.3f}s avg latency)")
    lines.append(f"Lowest heuristic failure rate: {lowest_failure['model']} ({lowest_failure['failure_rate']:.3f})")
    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 60)
    lines.append("A backend model should not be selected only from one global score.")
    lines.append("The safer strategy is model routing: choose the best model per use case.")
    lines.append("The final decision should be updated after LLM-as-a-judge and/or human evaluation.")

    return "\n".join(lines)


def main():
    required_files = [
        METRICS_USECASE_FILE,
        FAILURES_USECASE_FILE,
        COST_USECASE_FILE,
        METRICS_MODEL_FILE,
        FAILURES_MODEL_FILE,
        COST_MODEL_FILE,
    ]

    for file in required_files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Cannot find {file}")

    usecase_df = load_and_merge_usecase()
    overall_df = load_and_merge_overall()

    scored_usecase = (
        usecase_df
        .groupby("use_case", group_keys=False)
        .apply(score_usecase_group)
    )

    scored_overall = score_overall(overall_df)

    scored_usecase.to_csv(OUT_USECASE, index=False, encoding="utf-8-sig")
    scored_overall.to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")

    summary_text = create_summary_text(scored_usecase, scored_overall)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("Done.")
    print(f"Created: {OUT_USECASE}")
    print(f"Created: {OUT_OVERALL}")
    print(f"Created: {OUT_TXT}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
