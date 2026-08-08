from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .catalog import Catalogs, canonical_dataset_from_payload, load_catalogs, load_legacy_benchmark_datasets
from .domain import Dataset, EvaluationResult, PromptStrategy, RunSpec
from .evaluators import Evaluator, default_evaluator_registry
from .persistence import SqliteRunRepository
from .pipeline import EvaluationPipeline
from .providers import ModelProvider, OpenRouterProvider
from .registry import EvaluatorRegistry, ProviderRegistry


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_framework_%Y%m%d_%H%M%S")


def available_datasets(
    *,
    benchmark_path: Path,
    uploaded_payload: Any | None = None,
) -> dict[str, Dataset]:
    datasets = load_legacy_benchmark_datasets(benchmark_path)
    if uploaded_payload is not None:
        uploaded = canonical_dataset_from_payload(uploaded_payload)
        datasets[uploaded.task_id] = uploaded
    return datasets


def compatible_prompt_strategies(
    dataset: Dataset,
    strategies: tuple[PromptStrategy, ...],
) -> tuple[PromptStrategy, ...]:
    compatible: list[PromptStrategy] = []
    for strategy in strategies:
        valid = True
        for record in dataset.records:
            available = {"input", "reference", *record.variables.keys()}
            if any(variable not in available for variable in strategy.variables):
                valid = False
                break
        if valid:
            compatible.append(strategy)
    return tuple(compatible)


def runnable_task_ids(
    catalogs: Catalogs,
    evaluators: EvaluatorRegistry[Evaluator],
    datasets: dict[str, Dataset],
) -> tuple[str, ...]:
    return tuple(
        task_id
        for task_id in sorted(datasets)
        if catalogs.tasks.contains(task_id)
        and evaluators.contains(catalogs.tasks.get(task_id).evaluator_type)
    )


def build_openrouter_pipeline(
    db_path: Path | None,
    *,
    on_result: Callable[[EvaluationResult, int, int], None] | None = None,
) -> EvaluationPipeline:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required to start an evaluation run")
    providers: ProviderRegistry[ModelProvider] = ProviderRegistry()
    providers.register(
        "openrouter",
        OpenRouterProvider(
            api_key=api_key,
            timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180")),
            max_retries=int(os.getenv("MAX_API_RETRIES", "2")),
            retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0")),
            extra_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LLM Evaluation Framework"),
            },
        ),
    )
    return EvaluationPipeline(
        providers=providers,
        evaluators=default_evaluator_registry(),
        sink=SqliteRunRepository(db_path),
        on_result=on_result,
    )


def execute_run(
    spec: RunSpec,
    *,
    db_path: Path | None,
    on_result: Callable[[EvaluationResult, int, int], None] | None = None,
) -> list[EvaluationResult]:
    return build_openrouter_pipeline(db_path, on_result=on_result).run(spec)


def load_default_catalogs(base_dir: Path) -> Catalogs:
    return load_catalogs(base_dir / "catalog")
