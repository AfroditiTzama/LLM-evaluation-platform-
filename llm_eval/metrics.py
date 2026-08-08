from __future__ import annotations

from .domain import MetricValue, Timings, Usage


def _safe_rate(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def common_metrics(*, status: str, usage: Usage, timings: Timings) -> tuple[MetricValue, ...]:
    """Return provider-neutral operational metrics with unambiguous names.

    ``effective_output_tokens_per_second`` uses end-to-end request latency.
    ``generation_output_tokens_per_second`` uses provider-reported generation
    time and therefore remains unavailable for non-streaming providers that do
    not expose it.
    """
    return (
        MetricValue("success", status == "success", "boolean"),
        MetricValue("end_to_end_latency", timings.end_to_end_seconds, "seconds"),
        MetricValue("provider_request_latency", timings.provider_request_seconds, "seconds"),
        MetricValue("time_to_first_token", timings.time_to_first_token_seconds, "seconds"),
        MetricValue("generation_time", timings.generation_seconds, "seconds"),
        MetricValue("inter_token_latency", timings.inter_token_latency_seconds, "seconds/token"),
        MetricValue("input_tokens", usage.input_tokens, "tokens"),
        MetricValue("output_tokens", usage.output_tokens, "tokens"),
        MetricValue("total_tokens", usage.total_tokens, "tokens"),
        MetricValue("cost", usage.cost_usd, "USD"),
        MetricValue(
            "effective_output_tokens_per_second",
            _safe_rate(usage.output_tokens, timings.end_to_end_seconds),
            "tokens/second",
            {"definition": "output_tokens / end_to_end_latency"},
        ),
        MetricValue(
            "generation_output_tokens_per_second",
            _safe_rate(usage.output_tokens, timings.generation_seconds),
            "tokens/second",
            {"definition": "output_tokens / provider_reported_generation_time"},
        ),
    )
