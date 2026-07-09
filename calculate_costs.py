import pandas as pd
from pathlib import Path

INPUT_FILE = "all_model_logs_merged_fixed.csv"

OUT_FULL = "cost_analysis_full.csv"
OUT_MODEL = "cost_analysis_by_model.csv"
OUT_MODEL_USECASE = "cost_analysis_by_model_usecase.csv"
OUT_MODEL_DIFFICULTY = "cost_analysis_by_model_difficulty.csv"
OUT_PROMPT = "cost_analysis_by_prompt.csv"


# Prices are in USD per 1M tokens.
# We use input/output prices only, because the CSV logs do not contain cached token columns.
PRICE_PER_1M_TOKENS = {
    "minimax/minimax-m2.5": {
        "input": 0.30,
        "output": 1.20,
    },
    "minimax/minimax-m2.7": {
        "input": 0.30,
        "output": 1.20,
    },
    "minimax/minimax-m3": {
        "input": 0.30,
        "output": 1.20,
    },
    "qwen/qwen3.6-plus": {
        "input": 0.50,
        "output": 3.00,
    },
    "qwen/qwen3.7-plus": {
        # Qwen 3.7 Plus <= 256K tokens
        "input": 0.40,
        "output": 1.60,
    },
    "qwen/qwen3.7-max": {
        "input": 2.50,
        "output": 7.50,
    },
}


def add_cost_columns(df):
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0)
    df["total_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0)

    df["input_price_per_1m"] = df["model"].map(
        lambda model: PRICE_PER_1M_TOKENS.get(model, {}).get("input")
    )

    df["output_price_per_1m"] = df["model"].map(
        lambda model: PRICE_PER_1M_TOKENS.get(model, {}).get("output")
    )

    missing_prices = df[
        df["input_price_per_1m"].isna() | df["output_price_per_1m"].isna()
    ]["model"].unique()

    if len(missing_prices) > 0:
        raise ValueError(
            "Missing prices for models: "
            + ", ".join(str(model) for model in missing_prices)
        )

    df["input_cost_usd"] = (df["input_tokens"] / 1_000_000) * df["input_price_per_1m"]
    df["output_cost_usd"] = (df["output_tokens"] / 1_000_000) * df["output_price_per_1m"]
    df["total_cost_usd"] = df["input_cost_usd"] + df["output_cost_usd"]

    return df


def summarize_costs(df, group_cols):
    summary = (
        df.groupby(group_cols)
        .agg(
            rows=("prompt_id", "count"),
            unique_prompts=("prompt_id", "nunique"),
            total_input_tokens=("input_tokens", "sum"),
            total_output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            total_input_cost_usd=("input_cost_usd", "sum"),
            total_output_cost_usd=("output_cost_usd", "sum"),
            total_cost_usd=("total_cost_usd", "sum"),
            avg_cost_per_prompt_usd=("total_cost_usd", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
        )
        .reset_index()
    )

    # Estimated cost if we run 1,000 prompts with the same average cost.
    summary["estimated_cost_per_1000_prompts_usd"] = (
        summary["avg_cost_per_prompt_usd"] * 1000
    )

    return summary


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = add_cost_columns(df)

    df.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")

    by_model = summarize_costs(df, ["model"])
    by_model_usecase = summarize_costs(df, ["model", "use_case"])
    by_model_difficulty = summarize_costs(df, ["model", "difficulty"])
    by_prompt = summarize_costs(df, ["prompt_id", "use_case", "difficulty"])

    by_model = by_model.sort_values("total_cost_usd")
    by_model_usecase = by_model_usecase.sort_values(["use_case", "total_cost_usd"])
    by_model_difficulty = by_model_difficulty.sort_values(["difficulty", "total_cost_usd"])

    by_model.to_csv(OUT_MODEL, index=False, encoding="utf-8-sig")
    by_model_usecase.to_csv(OUT_MODEL_USECASE, index=False, encoding="utf-8-sig")
    by_model_difficulty.to_csv(OUT_MODEL_DIFFICULTY, index=False, encoding="utf-8-sig")
    by_prompt.to_csv(OUT_PROMPT, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Created: {OUT_FULL}")
    print(f"Created: {OUT_MODEL}")
    print(f"Created: {OUT_MODEL_USECASE}")
    print(f"Created: {OUT_MODEL_DIFFICULTY}")
    print(f"Created: {OUT_PROMPT}")

    print("\nCost summary by model:")
    display_cols = [
        "model",
        "rows",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_cost_usd",
        "avg_cost_per_prompt_usd",
        "estimated_cost_per_1000_prompts_usd",
    ]
    print(by_model[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
