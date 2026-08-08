from __future__ import annotations

import json
import string
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping


JsonMapping = Mapping[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _deep_freeze(dict(value or {}))


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value


def _snapshot(value: Any) -> dict[str, Any]:
    """Return a JSON-safe deep copy suitable for immutable run snapshots."""
    return json.loads(json.dumps(to_jsonable(value), ensure_ascii=False, default=str))


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    provider: str
    context_window: int | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    reasoning_support: bool = False
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip() or not self.provider.strip():
            raise ValueError("Model id, name and provider are required")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.id}"

    def snapshot(self) -> dict[str, Any]:
        return _snapshot(self)


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    description: str
    task_type: str
    evaluator_type: str
    supported_metrics: tuple[str, ...] = ()
    evaluator_version: str = "1.0"
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.evaluator_type.strip():
            raise ValueError("Task id and evaluator_type are required")
        object.__setattr__(self, "supported_metrics", tuple(self.supported_metrics))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def snapshot(self) -> dict[str, Any]:
        return _snapshot(self)


@dataclass(frozen=True)
class Provenance:
    source_title: str = ""
    organization: str = ""
    publication: str = ""
    url: str = ""
    doi: str = ""
    record_origin: str = "unknown"
    notes: str = ""


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    input: Any
    reference: Any | None = None
    language: str = ""
    difficulty: str = ""
    domain: str = ""
    variables: JsonMapping = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Dataset record id is required")
        object.__setattr__(self, "variables", _frozen_mapping(self.variables))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def snapshot(self) -> dict[str, Any]:
        return _snapshot(self)


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    task_id: str
    version: str
    records: tuple[DatasetRecord, ...]
    language: str = ""
    domain: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.version.strip() or not self.task_id.strip():
            raise ValueError("Dataset id, version and task_id are required")
        records = tuple(self.records)
        record_ids = [record.id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Dataset record ids must be unique within a version")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def snapshot(self, *, include_records: bool = False) -> dict[str, Any]:
        value = _snapshot(self)
        if not include_records:
            value.pop("records", None)
            value["record_count"] = len(self.records)
        return value


@dataclass(frozen=True)
class PromptStrategy:
    id: str
    name: str
    strategy_type: str
    version: str
    system_prompt: str
    user_prompt_template: str
    language: str = ""
    variables: tuple[str, ...] = ("input",)
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.version.strip():
            raise ValueError("Prompt strategy id and version are required")
        variables = tuple(self.variables)
        declared = set(variables)
        used = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.user_prompt_template)
            if field_name
        }
        missing = used - declared
        if missing:
            raise ValueError(f"Prompt template uses undeclared variables: {sorted(missing)}")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    def render(self, record: DatasetRecord) -> tuple[str, str]:
        values = {
            "input": record.input,
            "reference": "" if record.reference is None else record.reference,
            **dict(record.variables),
        }
        missing = [name for name in self.variables if name not in values]
        if missing:
            raise ValueError(f"Missing prompt variables for {self.key}: {missing}")
        try:
            user_prompt = self.user_prompt_template.format_map(values)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Could not render prompt strategy {self.key}: {exc}") from exc
        return self.system_prompt, user_prompt

    def snapshot(self) -> dict[str, Any]:
        return _snapshot(self)


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1000
    reasoning_enabled: bool = False
    reasoning_effort: str | None = None
    seed: int | None = None
    extra: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        object.__setattr__(self, "extra", _frozen_mapping(self.extra))

    def snapshot(self) -> dict[str, Any]:
        return _snapshot(self)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    models: tuple[Model, ...]
    task: Task
    dataset: Dataset
    prompt_strategies: tuple[PromptStrategy, ...]
    generation_settings: GenerationSettings = field(default_factory=GenerationSettings)
    metric_configuration: JsonMapping = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.dataset.task_id != self.task.id:
            raise ValueError(
                f"Dataset {self.dataset.id}@{self.dataset.version} belongs to task "
                f"{self.dataset.task_id}, not {self.task.id}"
            )
        if not self.models or not self.prompt_strategies:
            raise ValueError("At least one model and prompt strategy are required")
        models = tuple(self.models)
        prompts = tuple(self.prompt_strategies)
        if len({model.key for model in models}) != len(models):
            raise ValueError("Run models must be unique")
        if len({prompt.key for prompt in prompts}) != len(prompts):
            raise ValueError("Run prompt strategy versions must be unique")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "prompt_strategies", prompts)
        object.__setattr__(self, "metric_configuration", _frozen_mapping(self.metric_configuration))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass(frozen=True)
class Timings:
    end_to_end_seconds: float | None = None
    provider_request_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    generation_seconds: float | None = None
    inter_token_latency_seconds: float | None = None


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float | int | bool | str | None
    unit: str = ""
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Metric name is required")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class EvaluationResult:
    result_id: str
    run_id: str
    record_id: str
    model_key: str
    task_id: str
    dataset_id: str
    dataset_version: str
    prompt_strategy_id: str
    prompt_strategy_version: str
    status: str
    raw_input: Any
    raw_output: str
    reference: Any | None
    system_prompt_snapshot: str
    user_prompt_snapshot: str
    model_snapshot: JsonMapping
    task_snapshot: JsonMapping
    dataset_snapshot: JsonMapping
    prompt_strategy_snapshot: JsonMapping
    generation_settings_snapshot: JsonMapping
    evaluator_version: str
    metric_configuration_snapshot: JsonMapping
    provider: str
    resolved_model_id: str
    usage: Usage = field(default_factory=Usage)
    timings: Timings = field(default_factory=Timings)
    metrics: tuple[MetricValue, ...] = ()
    error: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "model_snapshot",
            "task_snapshot",
            "dataset_snapshot",
            "prompt_strategy_snapshot",
            "generation_settings_snapshot",
            "metric_configuration_snapshot",
            "metadata",
        ):
            object.__setattr__(self, name, _frozen_mapping(getattr(self, name)))
        object.__setattr__(self, "metrics", tuple(self.metrics))

    def metric(self, name: str) -> MetricValue | None:
        return next((item for item in self.metrics if item.name == name), None)
