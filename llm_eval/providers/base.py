from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..domain import GenerationSettings, Model, Timings, Usage


@dataclass(frozen=True)
class ProviderRequest:
    model: Model
    system_prompt: str
    user_prompt: str
    settings: GenerationSettings
    response_format: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResponse:
    status: str
    output: str = ""
    provider: str = ""
    resolved_model_id: str = ""
    usage: Usage = field(default_factory=Usage)
    timings: Timings = field(default_factory=Timings)
    finish_reason: str = ""
    error: str = ""
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    name: str

    def generate(self, request: ProviderRequest) -> ProviderResponse: ...
