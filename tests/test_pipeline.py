from __future__ import annotations

import unittest

from llm_eval.domain import (
    Dataset,
    DatasetRecord,
    GenerationSettings,
    Model,
    PromptStrategy,
    RunSpec,
    Task,
    Timings,
    Usage,
)
from llm_eval.evaluators import default_evaluator_registry
from llm_eval.pipeline import EvaluationPipeline
from llm_eval.providers import ProviderResponse
from llm_eval.registry import ProviderRegistry, Registry


class FakeProvider:
    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if self.fail:
            return ProviderResponse(
                status="error",
                provider=self.name,
                resolved_model_id=request.model.id,
                error="planned provider failure",
            )
        return ProviderResponse(
            status="success",
            output="ΝΑΙ",
            provider=self.name,
            resolved_model_id=request.model.id,
            usage=Usage(input_tokens=8, output_tokens=1, total_tokens=9, cost_usd=0.001),
            timings=Timings(provider_request_seconds=0.01, generation_seconds=0.005),
        )


def make_spec(*, run_id: str = "run-pipeline", evaluator_version: str = "1.0") -> RunSpec:
    models = (
        Model("model-a", "Model A", "fake"),
        Model("model-b", "Model B", "fake"),
    )
    task = Task(
        "qa",
        "Question answering",
        "",
        "qa",
        "exact_match",
        ("strict_exact_match", "normalized_exact_match"),
        evaluator_version,
    )
    dataset = Dataset(
        "qa-demo",
        "QA demo",
        "qa",
        "1",
        (
            DatasetRecord("r1", "Είναι το 2 άρτιο;", "ΝΑΙ", difficulty="easy"),
            DatasetRecord("r2", "Είναι το 4 άρτιο;", "ΝΑΙ", difficulty="easy"),
        ),
    )
    strategies = (
        PromptStrategy("basic", "Basic", "zero_shot", "1", "", "{input}"),
        PromptStrategy("optimized", "Optimized", "optimized", "2", "Έλεγξε.", "Task: {input}"),
    )
    return RunSpec(run_id, models, task, dataset, strategies, GenerationSettings())


class RegistryTests(unittest.TestCase):
    def test_duplicate_and_unknown_entries_fail_clearly(self) -> None:
        registry = Registry[str]("example")
        registry.register("a", "first")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            registry.register("a", "second")
        with self.assertRaisesRegex(KeyError, "Available: a"):
            registry.get("missing")


class EvaluationPipelineTests(unittest.TestCase):
    def pipeline(self, provider: FakeProvider) -> EvaluationPipeline:
        providers = ProviderRegistry()
        providers.register("fake", provider)
        return EvaluationPipeline(providers=providers, evaluators=default_evaluator_registry())

    def test_executes_full_model_prompt_dataset_matrix_and_keeps_snapshots(self) -> None:
        provider = FakeProvider()
        spec = make_spec()

        results = self.pipeline(provider).run(spec)

        self.assertEqual(len(results), 2 * 2 * 2)
        self.assertEqual(len(provider.requests), 8)
        self.assertTrue(all(result.status == "success" for result in results))
        self.assertEqual(
            {result.prompt_strategy_version for result in results},
            {"1", "2"},
        )
        optimized = next(result for result in results if result.prompt_strategy_id == "optimized")
        self.assertEqual(optimized.system_prompt_snapshot, "Έλεγξε.")
        self.assertTrue(optimized.user_prompt_snapshot.startswith("Task: "))
        self.assertTrue(optimized.metric("normalized_exact_match").value)
        self.assertEqual(optimized.generation_settings_snapshot["temperature"], 0.0)

    def test_result_ids_and_prompt_snapshots_are_reproducible(self) -> None:
        spec = make_spec()

        first = self.pipeline(FakeProvider()).run(spec)
        second = self.pipeline(FakeProvider()).run(spec)

        self.assertEqual([item.result_id for item in first], [item.result_id for item in second])
        self.assertEqual(
            [item.prompt_strategy_snapshot for item in first],
            [item.prompt_strategy_snapshot for item in second],
        )

    def test_provider_failure_is_recorded_without_task_quality_score(self) -> None:
        results = self.pipeline(FakeProvider(fail=True)).run(make_spec())

        self.assertTrue(all(result.status == "error" for result in results))
        self.assertTrue(all(result.error == "planned provider failure" for result in results))
        self.assertTrue(all(result.metric("success").value is False for result in results))
        self.assertTrue(all(result.metric("normalized_exact_match") is None for result in results))

    def test_evaluator_version_mismatch_stops_before_requests(self) -> None:
        provider = FakeProvider()
        with self.assertRaisesRegex(ValueError, "requires evaluator"):
            self.pipeline(provider).run(make_spec(evaluator_version="2.0"))
        self.assertEqual(provider.requests, [])

    def test_structured_strategy_forwards_record_json_schema(self) -> None:
        model = Model("model-a", "Model A", "fake")
        task = Task("structured", "Structured", "", "extraction", "structured_output")
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        dataset = Dataset(
            "structured-demo",
            "Structured demo",
            "structured",
            "1",
            (
                DatasetRecord(
                    "one",
                    "Return an answer",
                    metadata={
                        "evaluation": {
                            "expected_format": "json",
                            "format_spec": {"json_schema": schema},
                        }
                    },
                ),
            ),
        )
        strategy = PromptStrategy(
            "json",
            "JSON",
            "structured_output",
            "1",
            "Only JSON",
            "{input}",
            metadata={"requires_task_schema": True},
        )
        provider = FakeProvider()
        self.pipeline(provider).run(RunSpec("run-json", (model,), task, dataset, (strategy,)))

        response_format = provider.requests[0].response_format
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["schema"], schema)

    def test_progress_callback_receives_completed_and_total_counts(self) -> None:
        provider = FakeProvider()
        providers = ProviderRegistry()
        providers.register("fake", provider)
        updates = []
        pipeline = EvaluationPipeline(
            providers=providers,
            evaluators=default_evaluator_registry(),
            on_result=lambda result, completed, total: updates.append(
                (result.result_id, completed, total)
            ),
        )

        results = pipeline.run(make_spec())

        self.assertEqual(len(updates), len(results))
        self.assertEqual([item[1] for item in updates], list(range(1, len(results) + 1)))
        self.assertTrue(all(item[2] == len(results) for item in updates))


if __name__ == "__main__":
    unittest.main()
