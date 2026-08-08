from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import Dataset, DatasetRecord, Model, PromptStrategy, Provenance, Task
from .registry import PromptStrategyRegistry, TaskRegistry


@dataclass(frozen=True)
class Catalogs:
    models: tuple[Model, ...]
    tasks: TaskRegistry[Task]
    prompt_strategies: PromptStrategyRegistry[PromptStrategy]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalogs(path: Path) -> Catalogs:
    model_rows = _read_json(path / "models.json")
    task_rows = _read_json(path / "tasks.json")
    prompt_rows = _read_json(path / "prompt_strategies.json")

    models = tuple(
        Model(
            id=row["id"],
            name=row["name"],
            provider=row["provider"],
            context_window=row.get("context_window"),
            input_price_per_million=row.get("input_price_per_million"),
            output_price_per_million=row.get("output_price_per_million"),
            reasoning_support=bool(row.get("reasoning_support", False)),
            metadata=row.get("metadata", {}),
        )
        for row in model_rows
    )
    task_registry: TaskRegistry[Task] = TaskRegistry()
    for row in task_rows:
        task = Task(
            id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            task_type=row.get("task_type", row["id"]),
            evaluator_type=row["evaluator_type"],
            supported_metrics=tuple(row.get("supported_metrics", [])),
            evaluator_version=str(row.get("evaluator_version", "1.0")),
            metadata=row.get("metadata", {}),
        )
        task_registry.register(task.id, task)

    prompt_registry: PromptStrategyRegistry[PromptStrategy] = PromptStrategyRegistry()
    for row in prompt_rows:
        prompt = PromptStrategy(
            id=row["id"],
            name=row["name"],
            strategy_type=row["strategy_type"],
            version=str(row["version"]),
            system_prompt=row.get("system_prompt", ""),
            user_prompt_template=row["user_prompt_template"],
            language=row.get("language", ""),
            variables=tuple(row.get("variables", ["input"])),
            metadata=row.get("metadata", {}),
        )
        prompt_registry.register(prompt.key, prompt)
    return Catalogs(models, task_registry, prompt_registry)


def load_legacy_benchmark_datasets(path: Path) -> dict[str, Dataset]:
    """Expose the current benchmark as versioned datasets grouped by task.

    The source file contains newly authored prompts inspired by public
    benchmarks. This loader preserves that status explicitly and does not claim
    that the records were sourced from those benchmarks.
    """
    payload = _read_json(path)
    prompts = payload["prompts"] if isinstance(payload, dict) else payload
    version = str(payload.get("version", "legacy")) if isinstance(payload, dict) else "legacy"
    name = payload.get("dataset_name", path.stem) if isinstance(payload, dict) else path.stem
    language = payload.get("language", "") if isinstance(payload, dict) else ""
    groups: dict[str, list[DatasetRecord]] = {}
    for prompt in prompts:
        task_id = str(prompt["category"])
        provenance = Provenance(
            source_title=str(prompt.get("source_benchmark", "")),
            record_origin="newly_authored",
            notes="Benchmark inspiration only; this record is not copied source data.",
        )
        groups.setdefault(task_id, []).append(
            DatasetRecord(
                id=str(prompt["prompt_id"]),
                input=prompt["prompt"],
                reference=prompt.get("reference_answer"),
                language=language,
                difficulty=str(prompt.get("difficulty", "")),
                domain=str(prompt.get("domain", "")),
                provenance=provenance,
                metadata={
                    "legacy_prompt": prompt,
                    "evaluation": {
                        "evaluation_type": prompt.get("evaluation_type", ""),
                        "expected_format": prompt.get("expected_format", "text"),
                        "format_spec": prompt.get("format_spec", {}),
                        "accepted_answers": prompt.get("accepted_answers"),
                        "match_mode": prompt.get("match_mode", ""),
                        "numeric_tolerance": prompt.get("numeric_tolerance", 1e-9),
                    },
                },
            )
        )

    dataset_provenance = Provenance(
        source_title=str(name),
        record_origin="newly_authored",
        notes=str(payload.get("design_note", "")) if isinstance(payload, dict) else "",
    )
    return {
        task_id: Dataset(
            id=f"greek-research-inspired-{task_id.replace('_', '-')}",
            name=f"{name} — {task_id}",
            task_id=task_id,
            version=version,
            records=tuple(records),
            language=language,
            provenance=dataset_provenance,
            metadata={
                "design_sources": payload.get("design_sources", []) if isinstance(payload, dict) else [],
                "data_status": "newly_authored_benchmark_items",
            },
        )
        for task_id, records in groups.items()
    }


def _provenance(value: dict[str, Any] | None) -> Provenance:
    source = value or {}
    return Provenance(
        source_title=str(source.get("source_title", "")),
        organization=str(source.get("organization", "")),
        publication=str(source.get("publication", "")),
        url=str(source.get("url", "")),
        doi=str(source.get("doi", "")),
        record_origin=str(source.get("record_origin", "unknown")),
        notes=str(source.get("notes", "")),
    )


def canonical_dataset_from_payload(payload: Any) -> Dataset:
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("Canonical dataset JSON must be an object with a records list")
    required = {"id", "name", "task_id", "version"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Canonical dataset is missing fields: {sorted(missing)}")

    default_language = str(payload.get("language", ""))
    default_domain = str(payload.get("domain", ""))
    records = []
    for index, row in enumerate(payload["records"], start=1):
        if not isinstance(row, dict) or "id" not in row or "input" not in row:
            raise ValueError(f"Canonical dataset record #{index} requires id and input")
        records.append(
            DatasetRecord(
                id=str(row["id"]),
                input=row["input"],
                reference=row.get("reference"),
                language=str(row.get("language", default_language)),
                difficulty=str(row.get("difficulty", "")),
                domain=str(row.get("domain", default_domain)),
                variables=row.get("variables", {}),
                provenance=_provenance(row.get("provenance")),
                metadata=row.get("metadata", {}),
            )
        )
    return Dataset(
        id=str(payload["id"]),
        name=str(payload["name"]),
        task_id=str(payload["task_id"]),
        version=str(payload["version"]),
        records=tuple(records),
        language=default_language,
        domain=default_domain,
        provenance=_provenance(payload.get("provenance")),
        metadata=payload.get("metadata", {}),
    )


def load_canonical_dataset(path: Path) -> Dataset:
    return canonical_dataset_from_payload(_read_json(path))


def load_dataset_file(path: Path) -> dict[str, Dataset]:
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        dataset = load_canonical_dataset(path)
        return {dataset.task_id: dataset}
    return load_legacy_benchmark_datasets(path)
