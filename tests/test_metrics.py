from __future__ import annotations

import unittest

from llm_eval.domain import Timings, Usage
from llm_eval.metrics import common_metrics


class CommonMetricTests(unittest.TestCase):
    def test_effective_and_generation_throughput_have_distinct_denominators(self) -> None:
        metrics = {
            item.name: item
            for item in common_metrics(
                status="success",
                usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.01),
                timings=Timings(end_to_end_seconds=4, generation_seconds=2),
            )
        }

        self.assertEqual(metrics["effective_output_tokens_per_second"].value, 5)
        self.assertEqual(metrics["generation_output_tokens_per_second"].value, 10)
        self.assertNotEqual(
            metrics["effective_output_tokens_per_second"].metadata["definition"],
            metrics["generation_output_tokens_per_second"].metadata["definition"],
        )

    def test_streaming_metrics_remain_missing_when_provider_does_not_report_them(self) -> None:
        metrics = {
            item.name: item
            for item in common_metrics(
                status="success",
                usage=Usage(output_tokens=20),
                timings=Timings(end_to_end_seconds=4),
            )
        }

        self.assertIsNone(metrics["time_to_first_token"].value)
        self.assertIsNone(metrics["generation_time"].value)
        self.assertIsNone(metrics["generation_output_tokens_per_second"].value)


if __name__ == "__main__":
    unittest.main()
