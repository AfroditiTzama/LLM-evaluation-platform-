from __future__ import annotations

import argparse
import json
from pathlib import Path

from database import import_run_bundle
from reporting import rebuild_artifacts


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalculate deterministic metrics, rebuild summaries, create a stratified human sample and import a run into SQLite without new API calls."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--prompts", type=Path, default=Path("benchmark_prompts.json"))
    parser.add_argument("--database", type=Path, default=Path("data/llm_eval.db"))
    args = parser.parse_args()

    dataset = load_json(args.prompts)
    prompts = dataset["prompts"] if isinstance(dataset, dict) else dataset
    results = load_json(args.source_dir / f"{args.run_id}_results.json")
    judgments = load_json(args.source_dir / f"{args.run_id}_pairwise_judgments.json")
    metadata_path = args.source_dir / f"{args.run_id}_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {"run_id": args.run_id}
    metadata.setdefault("run_id", args.run_id)
    metadata.setdefault("dataset_metadata", {key: value for key, value in dataset.items() if key != "prompts"})
    metadata["dataset_metadata"]["version"] = dataset.get("version", "1.1") if isinstance(dataset, dict) else "1.1"

    bundle = rebuild_artifacts(
        args.run_id,
        prompts,
        results,
        judgments,
        args.output_dir,
    )

    import_run_bundle(
        db_path=args.database,
        metadata=metadata,
        prompts=prompts,
        results=bundle["corrected_results"],
        judgments=judgments,
        judge_rows=bundle["judge_rows"],
        model_summary=bundle["model_summary"],
        category_summary=bundle["category_summary"],
        difficulty_summary=bundle["difficulty_summary"],
        provider_summary=bundle["provider_summary"],
        human_rows=bundle["human_rows"],
        human_key=bundle["human_key"],
        notes="Existing run recalculated with benchmark v1.1; no model or judge API calls were repeated.",
    )

    print(f"Rebuilt run: {args.run_id}")
    print(f"Corrected artifacts: {args.output_dir.resolve()}")
    print(f"SQLite database: {args.database.resolve()}")


if __name__ == "__main__":
    main()
