from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evaluation import deterministic_evaluation, normalize_text

from .domain import DatasetRecord, MetricValue, PromptStrategy, Task
from .registry import EvaluatorRegistry


@dataclass(frozen=True)
class EvaluationContext:
    task: Task
    record: DatasetRecord
    prompt_strategy: PromptStrategy
    output: str


class Evaluator(Protocol):
    name: str
    version: str

    def evaluate(self, context: EvaluationContext) -> tuple[MetricValue, ...]: ...


class ExactMatchEvaluator:
    name = "exact_match"
    version = "1.0"

    def evaluate(self, context: EvaluationContext) -> tuple[MetricValue, ...]:
        if context.record.reference is None:
            return (MetricValue("exact_match", None, "ratio", {"reason": "missing_reference"}),)
        strict = context.output.strip() == str(context.record.reference).strip()
        normalized = normalize_text(context.output) == normalize_text(context.record.reference)
        return (
            MetricValue("strict_exact_match", strict, "ratio"),
            MetricValue("normalized_exact_match", normalized, "ratio"),
        )


class ClassificationEvaluator:
    name = "classification"
    version = "1.0"

    def evaluate(self, context: EvaluationContext) -> tuple[MetricValue, ...]:
        reference = context.record.reference
        predicted = normalize_text(context.output)
        expected = normalize_text(reference) if reference is not None else ""
        allowed = [normalize_text(item) for item in context.record.metadata.get("labels", [])]
        return (
            MetricValue("accuracy", None if reference is None else predicted == expected, "ratio"),
            MetricValue("label_valid", None if not allowed else predicted in allowed, "ratio"),
        )


class StructuredOutputEvaluator:
    name = "structured_output"
    version = "1.0"

    def evaluate(self, context: EvaluationContext) -> tuple[MetricValue, ...]:
        configuration = dict(context.record.metadata.get("evaluation", {}))
        prompt = {
            "evaluation_type": configuration.get("evaluation_type", "schema_validation"),
            "expected_format": configuration.get("expected_format", "json"),
            "format_spec": configuration.get("format_spec", {}),
            "reference_answer": context.record.reference,
            "accepted_answers": configuration.get("accepted_answers"),
            "match_mode": configuration.get("match_mode", ""),
            "numeric_tolerance": configuration.get("numeric_tolerance", 1e-9),
        }
        values = deterministic_evaluation(prompt, context.output)
        names = (
            "strict_exact_match",
            "normalized_exact_match",
            "numeric_match",
            "syntax_valid",
            "schema_valid",
            "format_compliance",
        )
        return tuple(MetricValue(name, values.get(name), "ratio") for name in names)


class LegacyDeterministicEvaluator:
    """Compatibility evaluator for records shaped like benchmark_prompts.json."""

    name = "legacy_deterministic"
    version = "1.1"

    def evaluate(self, context: EvaluationContext) -> tuple[MetricValue, ...]:
        prompt = dict(context.record.metadata.get("legacy_prompt", {}))
        if not prompt:
            raise ValueError("legacy_deterministic requires metadata.legacy_prompt")
        values = deterministic_evaluation(prompt, context.output)
        return tuple(
            MetricValue(name, value, "ratio" if isinstance(value, (bool, int, float)) else "")
            for name, value in values.items()
            if name
            in {
                "strict_exact_match",
                "normalized_exact_match",
                "contains_expected_answer",
                "numeric_match",
                "deterministic_pass",
                "syntax_valid",
                "schema_valid",
                "format_compliance",
            }
        )


def default_evaluator_registry() -> EvaluatorRegistry[Evaluator]:
    registry: EvaluatorRegistry[Evaluator] = EvaluatorRegistry()
    for evaluator in (
        ExactMatchEvaluator(),
        ClassificationEvaluator(),
        StructuredOutputEvaluator(),
        LegacyDeterministicEvaluator(),
    ):
        registry.register(evaluator.name, evaluator)
    return registry
