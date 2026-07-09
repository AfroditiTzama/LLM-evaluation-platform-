import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "all_model_logs_merged_fixed.csv"

OUT_MODEL = "metrics_by_model.csv"
OUT_USECASE = "metrics_by_model_usecase.csv"
OUT_DIFFICULTY = "metrics_by_model_difficulty.csv"
OUT_PROMPT = "metrics_by_prompt.csv"

MAX_TOKENS_BY_USE_CASE = {
    "website_generation": 3000,
    "customer_support": 600,
    "document_understanding": 900,
}


def p95(series):
    return float(np.percentile(series.dropna(), 95)) if len(series.dropna()) > 0 else np.nan


def add_basic_flags(df):
    df["answer"] = df["answer"].fillna("").astype(str)
    df["error"] = df["error"].fillna("").astype(str)

    df["answer_chars"] = df["answer"].str.len()
    df["has_error"] = df["error"].str.strip() != ""
    df["empty_answer"] = df["answer"].str.strip() == ""

    df["max_tokens_expected"] = df["use_case"].map(MAX_TOKENS_BY_USE_CASE)
    df["possible_truncation"] = df["output_tokens"] >= df["max_tokens_expected"]

    df["cost_proxy_total_tokens"] = df["total_tokens"]

    return df


def summarize(group):
    return group.agg(
        rows=("prompt_id", "count"),
        unique_prompts=("prompt_id", "nunique"),
        avg_latency=("latency_seconds", "mean"),
        median_latency=("latency_seconds", "median"),
        p95_latency=("latency_seconds", p95),
        min_latency=("latency_seconds", "min"),
        max_latency=("latency_seconds", "max"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_total_tokens=("total_tokens", "mean"),
        total_tokens=("total_tokens", "sum"),
        avg_answer_chars=("answer_chars", "mean"),
        errors=("has_error", "sum"),
        empty_answers=("empty_answer", "sum"),
        possible_truncations=("possible_truncation", "sum"),
    ).reset_index()


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = add_basic_flags(df)

    metrics_model = summarize(df.groupby("model"))
    metrics_usecase = summarize(df.groupby(["model", "use_case"]))
    metrics_difficulty = summarize(df.groupby(["model", "difficulty"]))
    metrics_prompt = summarize(df.groupby(["prompt_id", "use_case", "difficulty"]))

    metrics_model.to_csv(OUT_MODEL, index=False, encoding="utf-8-sig")
    metrics_usecase.to_csv(OUT_USECASE, index=False, encoding="utf-8-sig")
    metrics_difficulty.to_csv(OUT_DIFFICULTY, index=False, encoding="utf-8-sig")
    metrics_prompt.to_csv(OUT_PROMPT, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Created: {OUT_MODEL}")
    print(f"Created: {OUT_USECASE}")
    print(f"Created: {OUT_DIFFICULTY}")
    print(f"Created: {OUT_PROMPT}")

    print("\nModel-level summary:")
    print(metrics_model.to_string(index=False))


if __name__ == "__main__":
    main()
