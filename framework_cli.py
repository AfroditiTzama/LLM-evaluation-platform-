from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from llm_eval.catalog import load_catalogs, load_dataset_file
from llm_eval.domain import GenerationSettings, RunSpec
from llm_eval.evaluators import default_evaluator_registry
from llm_eval.persistence import SqliteRunRepository
from llm_eval.pipeline import EvaluationPipeline
from llm_eval.providers import OpenRouterProvider
from llm_eval.registry import ProviderRegistry


BASE_DIR = Path(__file__).resolve().parent


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_framework_%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extensible Model × Task × Prompt × Dataset evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="List models, tasks, evaluators and prompt strategy versions")

    run = subparsers.add_parser("run", help="Run a task slice from a versioned dataset")
    run.add_argument("--task", required=True)
    run.add_argument("--model", action="append", required=True, help="Catalog model id; repeat for multiple models")
    run.add_argument(
        "--prompt-strategy",
        action="append",
        required=True,
        help="Versioned key such as basic-zero-shot-el@1; repeat for multiple strategies",
    )
    run.add_argument("--dataset-file", type=Path, default=BASE_DIR / "benchmark_prompts.json")
    run.add_argument("--database", type=Path, default=BASE_DIR / "data" / "llm_eval.db")
    run.add_argument("--run-id", default=None)
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--top-p", type=float, default=0.9)
    run.add_argument("--max-tokens", type=int, default=3000)
    run.add_argument("--reasoning", action="store_true")
    run.add_argument("--reasoning-effort", default="medium")
    run.add_argument("--yes", action="store_true", help="Confirm paid provider requests")
    return parser


def list_catalog() -> None:
    catalogs = load_catalogs(BASE_DIR / "catalog")
    evaluators = default_evaluator_registry()
    print("Models")
    for model in catalogs.models:
        print(f"  {model.id} | {model.name} | provider={model.provider}")
    print("\nTasks")
    for task in catalogs.tasks.values():
        status = "runnable" if evaluators.contains(task.evaluator_type) else "evaluator planned"
        print(f"  {task.id} | evaluator={task.evaluator_type}@{task.evaluator_version} | {status}")
    print("\nPrompt strategies")
    for strategy in catalogs.prompt_strategies.values():
        print(f"  {strategy.key} | {strategy.name} | type={strategy.strategy_type}")


def run_evaluation(args: argparse.Namespace) -> None:
    catalogs = load_catalogs(BASE_DIR / "catalog")
    evaluators = default_evaluator_registry()
    task = catalogs.tasks.get(args.task)
    if not evaluators.contains(task.evaluator_type):
        raise ValueError(
            f"Task {task.id} uses evaluator {task.evaluator_type}@{task.evaluator_version}, "
            "which is cataloged but not implemented yet"
        )

    datasets = load_dataset_file(args.dataset_file)
    if task.id not in datasets:
        available = ", ".join(sorted(datasets))
        raise ValueError(f"Dataset file has no records for task {task.id}. Available: {available}")
    dataset = datasets[task.id]

    models_by_id = {model.id: model for model in catalogs.models}
    missing_models = sorted(set(args.model) - set(models_by_id))
    if missing_models:
        raise ValueError(f"Unknown model ids: {missing_models}")
    models = tuple(models_by_id[model_id] for model_id in args.model)
    prompts = tuple(catalogs.prompt_strategies.get(key) for key in args.prompt_strategy)
    request_count = len(models) * len(prompts) * len(dataset.records)
    if not args.yes:
        raise ValueError(
            f"This run would make {request_count} paid provider requests. "
            "Review the selection and repeat with --yes to confirm."
        )

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required in the environment or .env")
    extra_headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LLM Evaluation Framework"),
    }
    providers = ProviderRegistry()
    providers.register(
        "openrouter",
        OpenRouterProvider(
            api_key=api_key,
            timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180")),
            max_retries=int(os.getenv("MAX_API_RETRIES", "2")),
            retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0")),
            extra_headers=extra_headers,
        ),
    )
    spec = RunSpec(
        run_id=args.run_id or default_run_id(),
        models=models,
        task=task,
        dataset=dataset,
        prompt_strategies=prompts,
        generation_settings=GenerationSettings(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            reasoning_enabled=args.reasoning,
            reasoning_effort=args.reasoning_effort if args.reasoning else None,
        ),
        metric_configuration={"common_metrics_version": "1.0"},
        metadata={"entrypoint": "framework_cli.py", "dataset_file": str(args.dataset_file)},
    )
    pipeline = EvaluationPipeline(
        providers=providers,
        evaluators=evaluators,
        sink=SqliteRunRepository(args.database),
    )
    results = pipeline.run(spec)
    successes = sum(result.status == "success" for result in results)
    print(f"Run: {spec.run_id}")
    print(f"Results: {len(results)} | success: {successes} | failed: {len(results) - successes}")
    print(f"Database: {args.database.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "catalog":
            list_catalog()
        else:
            run_evaluation(args)
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
