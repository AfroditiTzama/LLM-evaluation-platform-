import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text


DEFAULT_DB_URL = "sqlite:///llm_eval.db"


TABLE_FILES = {
    "eval_prompts": "input_prompts.csv",
    "model_outputs": "custom_model_logs.csv",
    "model_summary": "custom_summary_by_model.csv",
    "model_usecase_summary": "custom_summary_by_model_usecase.csv",
    "preliminary_recommendations": "custom_preliminary_recommendations.csv",
    "judge_scores": "custom_judge_scores.csv",
    "judge_summary_model": "custom_judge_summary_by_model.csv",
    "final_recommendations": "custom_final_recommendations.csv",
}


def get_database_url():
    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

    # Some hosted providers return postgres://, while SQLAlchemy expects postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    return db_url


def read_csv_if_exists(path):
    if not path.exists():
        return None

    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def delete_existing_run_rows(engine, table_name, run_id):
    inspector = inspect(engine)

    if not inspector.has_table(table_name):
        return

    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE run_id = :run_id"),
            {"run_id": run_id},
        )


def write_run_table(engine, table_name, df, run_id):
    if df is None:
        return False

    df = df.copy()
    df["run_id"] = run_id
    df["synced_at"] = datetime.now().isoformat(timespec="seconds")

    delete_existing_run_rows(engine, table_name, run_id)

    if len(df) == 0:
        # Create empty table only if needed is not necessary.
        return True

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
    )

    return True


def load_run_config(run_dir):
    config_path = run_dir / "run_config.json"

    if not config_path.exists():
        return {}

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_run_metadata(run_dir, run_id):
    config = load_run_config(run_dir)

    logs = read_csv_if_exists(run_dir / "custom_model_logs.csv")
    prompts = read_csv_if_exists(run_dir / "input_prompts.csv")
    final = read_csv_if_exists(run_dir / "custom_final_recommendations.csv")
    judge = read_csv_if_exists(run_dir / "custom_judge_scores.csv")

    prompt_count = 0 if prompts is None else len(prompts)
    output_count = 0 if logs is None else len(logs)
    model_count = 0 if logs is None or "model" not in logs.columns else logs["model"].nunique()

    metadata = pd.DataFrame([
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "created_at": config.get("created_at", ""),
            "synced_at": datetime.now().isoformat(timespec="seconds"),
            "prompt_count": prompt_count,
            "model_count": model_count,
            "output_count": output_count,
            "has_judge_scores": judge is not None and len(judge) > 0,
            "has_final_recommendations": final is not None and len(final) > 0,
            "judge_model": config.get("judge_model", ""),
            "temperature": config.get("temperature", ""),
            "single_prompt_mode": config.get("single_prompt_mode", ""),
            "use_case": config.get("use_case", ""),
            "expected_format": config.get("expected_format", ""),
            "config_json": json.dumps(config, ensure_ascii=False),
        }
    ])

    return metadata


def sync_run(run_dir):
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_id = run_dir.name

    db_url = get_database_url()
    engine = create_engine(db_url)

    print(f"Database: {db_url}")
    print(f"Run directory: {run_dir}")
    print(f"Run ID: {run_id}")
    print("-" * 60)

    metadata = build_run_metadata(run_dir, run_id)

    write_run_table(
        engine=engine,
        table_name="runs",
        df=metadata,
        run_id=run_id,
    )

    synced_tables = ["runs"]

    for table_name, filename in TABLE_FILES.items():
        path = run_dir / filename
        df = read_csv_if_exists(path)

        if df is None:
            print(f"Skipping missing file: {filename}")
            continue

        ok = write_run_table(
            engine=engine,
            table_name=table_name,
            df=df,
            run_id=run_id,
        )

        if ok:
            synced_tables.append(table_name)
            print(f"Synced table: {table_name} from {filename} | rows={len(df)}")

    print()
    print("Done.")
    print("Synced tables:")
    for table in synced_tables:
        print(f"- {table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    sync_run(args.run_dir)


if __name__ == "__main__":
    main()
