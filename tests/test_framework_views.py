from __future__ import annotations

import unittest

import pandas as pd

from llm_eval.framework_views import (
    build_leaderboard,
    enrich_results,
    mark_pareto_frontier,
    quality_metric_options,
)


class FrameworkViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.DataFrame(
            [
                {
                    "result_id": "a",
                    "model_key": "fake:a",
                    "model_name": "Model A",
                    "prompt_strategy_id": "basic",
                    "prompt_strategy_version": "1",
                    "prompt_strategy_name": "Basic",
                    "status": "success",
                    "cost_usd": 0.02,
                    "end_to_end_seconds": 2.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
                {
                    "result_id": "b",
                    "model_key": "fake:b",
                    "model_name": "Model B",
                    "prompt_strategy_id": "basic",
                    "prompt_strategy_version": "1",
                    "prompt_strategy_name": "Basic",
                    "status": "success",
                    "cost_usd": 0.01,
                    "end_to_end_seconds": 1.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            ]
        )
        self.metrics = pd.DataFrame(
            [
                {"result_id": "a", "metric_name": "accuracy", "value_real": 1.0, "value_type": "number"},
                {"result_id": "b", "metric_name": "accuracy", "value_real": 0.0, "value_type": "number"},
                {"result_id": "a", "metric_name": "cost", "value_real": 0.02, "value_type": "number"},
                {"result_id": "b", "metric_name": "cost", "value_real": 0.01, "value_type": "number"},
            ]
        )

    def test_quality_options_exclude_common_operational_metrics(self) -> None:
        self.assertEqual(quality_metric_options(self.metrics), ["accuracy"])

    def test_leaderboard_and_pareto_use_explicit_quality_metric(self) -> None:
        enriched = enrich_results(self.results, self.metrics)
        leaderboard = mark_pareto_frontier(
            build_leaderboard(enriched, quality_metric="accuracy")
        )

        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(set(leaderboard["quality_metric"]), {"accuracy"})
        self.assertTrue(leaderboard["pareto_optimal"].all())


if __name__ == "__main__":
    unittest.main()
