import pandas as pd
from pathlib import Path

CSV_PATTERN = "chat_logs_*.csv"

MERGED_OUTPUT = "all_model_logs_merged.csv"
ERROR_OUTPUT = "error_rows_to_rerun.csv"
VALIDATION_OUTPUT = "csv_validation_summary.csv"


def main():
    csv_files = sorted(Path(".").glob(CSV_PATTERN))

    # Avoid reading generated summary/rerun files by mistake
    csv_files = [
        f for f in csv_files
        if not f.name.startswith("all_model_logs")
        and not f.name.startswith("error_rows")
        and not f.name.startswith("rerun")
        and not f.name.startswith("csv_validation")
        and not f.name.startswith("initial_metrics")
    ]

    if not csv_files:
        raise FileNotFoundError("No chat_logs_*.csv files found in this folder.")

    all_dfs = []
    validation_rows = []

    for file in csv_files:
        df = pd.read_csv(file)

        if "error" not in df.columns:
            df["error"] = ""

        if "answer" not in df.columns:
            df["answer"] = ""

        df["source_file"] = file.name

        all_dfs.append(df)

        model_name = df["model"].iloc[0] if "model" in df.columns and len(df) > 0 else "UNKNOWN"

        error_count = df["error"].fillna("").astype(str).str.strip().ne("").sum()
        empty_answer_count = df["answer"].fillna("").astype(str).str.strip().eq("").sum()

        validation_rows.append({
            "source_file": file.name,
            "model": model_name,
            "rows": len(df),
            "unique_prompt_ids": df["prompt_id"].nunique() if "prompt_id" in df.columns else None,
            "errors": int(error_count),
            "empty_answers": int(empty_answer_count),
        })

    merged = pd.concat(all_dfs, ignore_index=True)

    # Put source_file first
    cols = ["source_file"] + [c for c in merged.columns if c != "source_file"]
    merged = merged[cols]

    merged.to_csv(MERGED_OUTPUT, index=False, encoding="utf-8-sig")

    error_mask = (
        merged["error"].fillna("").astype(str).str.strip().ne("")
        | merged["answer"].fillna("").astype(str).str.strip().eq("")
    )

    error_rows = merged[error_mask].copy()
    error_rows.to_csv(ERROR_OUTPUT, index=False, encoding="utf-8-sig")

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Created: {MERGED_OUTPUT} | rows: {len(merged)}")
    print(f"Created: {ERROR_OUTPUT} | rows: {len(error_rows)}")
    print(f"Created: {VALIDATION_OUTPUT}")

    print("\nValidation summary:")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
