from __future__ import annotations

import unittest
from pathlib import Path

from llm_eval.evaluators import default_evaluator_registry
from llm_eval.run_service import (
    available_datasets,
    compatible_prompt_strategies,
    load_default_catalogs,
    runnable_task_ids,
)


ROOT = Path(__file__).resolve().parents[1]


class RunServiceTests(unittest.TestCase):
    def test_only_tasks_with_dataset_and_evaluator_are_runnable(self) -> None:
        catalogs = load_default_catalogs(ROOT)
        datasets = available_datasets(benchmark_path=ROOT / "benchmark_prompts.json")

        runnable = runnable_task_ids(catalogs, default_evaluator_registry(), datasets)

        self.assertIn("logical_reasoning", runnable)
        self.assertIn("structured_output", runnable)
        self.assertNotIn("summarization", runnable)

    def test_few_shot_is_hidden_without_examples_variable(self) -> None:
        catalogs = load_default_catalogs(ROOT)
        dataset = available_datasets(
            benchmark_path=ROOT / "benchmark_prompts.json"
        )["logical_reasoning"]

        strategies = compatible_prompt_strategies(dataset, catalogs.prompt_strategies.values())

        keys = {strategy.key for strategy in strategies}
        self.assertIn("basic-zero-shot-el@1", keys)
        self.assertNotIn("few-shot-el@1", keys)


if __name__ == "__main__":
    unittest.main()
