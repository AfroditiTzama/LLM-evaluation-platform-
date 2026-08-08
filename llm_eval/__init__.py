"""Extensible evaluation framework layered on top of the legacy benchmark app."""

from .domain import (
    Dataset,
    DatasetRecord,
    EvaluationResult,
    GenerationSettings,
    MetricValue,
    Model,
    PromptStrategy,
    RunSpec,
    Task,
)
from .pipeline import EvaluationPipeline

__all__ = [
    "Dataset",
    "DatasetRecord",
    "EvaluationPipeline",
    "EvaluationResult",
    "GenerationSettings",
    "MetricValue",
    "Model",
    "PromptStrategy",
    "RunSpec",
    "Task",
]
