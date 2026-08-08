from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar


class Keyed(Protocol):
    @property
    def key(self) -> str: ...


T = TypeVar("T")


class Registry(Generic[T]):
    """Small explicit registry with duplicate protection and useful errors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T, *, replace: bool = False) -> T:
        normalized = str(key).strip()
        if not normalized:
            raise ValueError(f"{self.name} registry keys cannot be empty")
        if normalized in self._items and not replace:
            raise ValueError(f"Duplicate {self.name} registry key: {normalized}")
        self._items[normalized] = item
        return item

    def register_many(self, items: Iterable[tuple[str, T]], *, replace: bool = False) -> None:
        for key, item in items:
            self.register(key, item, replace=replace)

    def get(self, key: str) -> T:
        try:
            return self._items[str(key)]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"Unknown {self.name} '{key}'. Available: {available}") from exc

    def contains(self, key: str) -> bool:
        return str(key) in self._items

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def values(self) -> tuple[T, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


class TaskRegistry(Registry[T]):
    def __init__(self) -> None:
        super().__init__("task")


class PromptStrategyRegistry(Registry[T]):
    def __init__(self) -> None:
        super().__init__("prompt strategy")


class ProviderRegistry(Registry[T]):
    def __init__(self) -> None:
        super().__init__("provider")


class EvaluatorRegistry(Registry[T]):
    def __init__(self) -> None:
        super().__init__("evaluator")


class MetricRegistry(Registry[T]):
    def __init__(self) -> None:
        super().__init__("metric")
