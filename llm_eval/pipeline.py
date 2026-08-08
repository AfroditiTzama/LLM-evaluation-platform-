from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Callable, Protocol

from .domain import EvaluationResult, RunSpec, Timings, Usage, to_jsonable
from .evaluators import EvaluationContext, Evaluator
from .metrics import common_metrics
from .providers import ModelProvider, ProviderRequest, ProviderResponse
from .registry import EvaluatorRegistry, ProviderRegistry


class ResultSink(Protocol):
    def begin_run(self, spec: RunSpec, *, evaluator_version: str) -> None: ...

    def save_result(self, result: EvaluationResult) -> None: ...

    def finish_run(self, run_id: str, *, status: str) -> None: ...


class EvaluationPipeline:
    def __init__(
        self,
        *,
        providers: ProviderRegistry[ModelProvider],
        evaluators: EvaluatorRegistry[Evaluator],
        sink: ResultSink | None = None,
        on_result: Callable[[EvaluationResult, int, int], None] | None = None,
    ) -> None:
        self.providers = providers
        self.evaluators = evaluators
        self.sink = sink
        self.on_result = on_result

    @staticmethod
    def _result_id(spec: RunSpec, model_key: str, prompt_key: str, record_id: str) -> str:
        identity = "|".join((spec.run_id, model_key, spec.task.id, spec.dataset.id, spec.dataset.version, prompt_key, record_id))
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    @staticmethod
    def _estimated_cost(spec: RunSpec, model_key: str, usage: Usage) -> Usage:
        if usage.cost_usd is not None:
            return usage
        model = next(item for item in spec.models if item.key == model_key)
        if model.input_price_per_million is None or model.output_price_per_million is None:
            return usage
        cost = (
            usage.input_tokens * model.input_price_per_million
            + usage.output_tokens * model.output_price_per_million
        ) / 1_000_000
        return replace(usage, cost_usd=cost)

    @staticmethod
    def _response_format(strategy, record):
        configured = strategy.metadata.get("response_format")
        if configured is not None:
            return configured
        if not strategy.metadata.get("requires_task_schema"):
            return None
        evaluation = record.metadata.get("evaluation", {})
        schema = evaluation.get("format_spec", {}).get("json_schema")
        if not schema:
            return None
        schema_name = "_".join(part for part in (strategy.id, record.id) if part).replace("-", "_")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name[:64],
                "strict": True,
                "schema": to_jsonable(schema),
            },
        }

    def run(self, spec: RunSpec) -> list[EvaluationResult]:
        evaluator = self.evaluators.get(spec.task.evaluator_type)
        if evaluator.version != spec.task.evaluator_version:
            raise ValueError(
                f"Task {spec.task.id} requires evaluator {spec.task.evaluator_type}@"
                f"{spec.task.evaluator_version}, but registry provides {evaluator.version}"
            )

        if self.sink:
            self.sink.begin_run(spec, evaluator_version=evaluator.version)

        results: list[EvaluationResult] = []
        failures = 0
        total_results = len(spec.models) * len(spec.prompt_strategies) * len(spec.dataset.records)
        try:
            for model in spec.models:
                provider = self.providers.get(model.provider)
                for strategy in spec.prompt_strategies:
                    for record in spec.dataset.records:
                        result = self._evaluate_one(spec, model.key, provider, evaluator, strategy, record)
                        results.append(result)
                        failures += result.status != "success"
                        if self.sink:
                            self.sink.save_result(result)
                        if self.on_result:
                            try:
                                self.on_result(result, len(results), total_results)
                            except Exception:
                                # Progress observers must never corrupt or abort a paid run.
                                pass
        except Exception:
            if self.sink:
                self.sink.finish_run(spec.run_id, status="failed")
            raise

        if self.sink:
            self.sink.finish_run(
                spec.run_id,
                status="completed_with_errors" if failures else "completed",
            )
        return results

    def _evaluate_one(self, spec, model_key, provider, evaluator, strategy, record) -> EvaluationResult:
        model = next(item for item in spec.models if item.key == model_key)
        system_prompt = ""
        user_prompt = ""
        response = ProviderResponse(
            status="error",
            provider=provider.name,
            resolved_model_id=model.id,
            error="Evaluation did not start",
        )
        try:
            system_prompt, user_prompt = strategy.render(record)
            request = ProviderRequest(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                settings=spec.generation_settings,
                response_format=self._response_format(strategy, record),
                metadata={"run_id": spec.run_id, "record_id": record.id},
            )
            started = time.perf_counter()
            response = provider.generate(request)
            effective_seconds = time.perf_counter() - started
            timings = replace(response.timings, end_to_end_seconds=effective_seconds)
            usage = self._estimated_cost(spec, model_key, response.usage)
            task_metrics = ()
            evaluation_seconds = None
            if response.status == "success":
                evaluation_started = time.perf_counter()
                task_metrics = evaluator.evaluate(
                    EvaluationContext(spec.task, record, strategy, response.output)
                )
                evaluation_seconds = time.perf_counter() - evaluation_started
            metrics = common_metrics(status=response.status, usage=usage, timings=timings) + task_metrics
            metric_names = [metric.name for metric in metrics]
            if len(metric_names) != len(set(metric_names)):
                raise ValueError("Evaluators must return unique metric names")
            metadata = {
                "finish_reason": response.finish_reason,
                "evaluation_latency_seconds": evaluation_seconds,
                "provider_metadata": dict(response.raw_metadata),
                "record": record.snapshot(),
            }
        except Exception as exc:
            timings = response.timings
            usage = response.usage
            response = replace(response, status="error", error=str(exc))
            metrics = common_metrics(status="error", usage=usage, timings=timings)
            metadata = {
                "pipeline_error_type": type(exc).__name__,
                "record": record.snapshot(),
            }

        return EvaluationResult(
            result_id=self._result_id(spec, model_key, strategy.key, record.id),
            run_id=spec.run_id,
            record_id=record.id,
            model_key=model.key,
            task_id=spec.task.id,
            dataset_id=spec.dataset.id,
            dataset_version=spec.dataset.version,
            prompt_strategy_id=strategy.id,
            prompt_strategy_version=strategy.version,
            status=response.status,
            raw_input=record.input,
            raw_output=response.output,
            reference=record.reference,
            system_prompt_snapshot=system_prompt,
            user_prompt_snapshot=user_prompt,
            model_snapshot=model.snapshot(),
            task_snapshot=spec.task.snapshot(),
            dataset_snapshot=spec.dataset.snapshot(),
            prompt_strategy_snapshot=strategy.snapshot(),
            generation_settings_snapshot=spec.generation_settings.snapshot(),
            evaluator_version=evaluator.version,
            metric_configuration_snapshot=dict(spec.metric_configuration),
            provider=response.provider or model.provider,
            resolved_model_id=response.resolved_model_id or model.id,
            usage=usage,
            timings=timings,
            metrics=metrics,
            error=response.error,
            metadata=metadata,
        )
