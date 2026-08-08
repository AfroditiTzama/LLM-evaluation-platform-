from __future__ import annotations

import unittest

from llm_eval.aggregation import aggregate_results, pareto_frontier
from tests.test_pipeline import FakeProvider, make_spec
from llm_eval.evaluators import default_evaluator_registry
from llm_eval.pipeline import EvaluationPipeline
from llm_eval.registry import ProviderRegistry


class AggregationTests(unittest.TestCase):
    def test_aggregates_by_prompt_and_difficulty_with_mean_median_std(self) -> None:
        providers = ProviderRegistry()
        providers.register("fake", FakeProvider())
        results = EvaluationPipeline(
            providers=providers,
            evaluators=default_evaluator_registry(),
        ).run(make_spec())

        rows = aggregate_results(
            results,
            group_by=("model_key", "prompt_strategy_id", "prompt_strategy_version", "difficulty"),
        )

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["difficulty"] == "easy" for row in rows))
        self.assertTrue(all(row["normalized_exact_match_mean"] == 1.0 for row in rows))
        self.assertTrue(all(row["normalized_exact_match_median"] == 1.0 for row in rows))
        self.assertTrue(all(row["normalized_exact_match_std"] == 0.0 for row in rows))

    def test_pareto_frontier_excludes_dominated_comparisons(self) -> None:
        rows = [
            {"name": "balanced", "quality": 0.9, "cost": 2.0},
            {"name": "cheap", "quality": 0.8, "cost": 1.0},
            {"name": "dominated", "quality": 0.7, "cost": 3.0},
        ]

        frontier = pareto_frontier(rows, maximize=("quality",), minimize=("cost",))

        self.assertEqual({row["name"] for row in frontier}, {"balanced", "cheap"})


if __name__ == "__main__":
    unittest.main()
