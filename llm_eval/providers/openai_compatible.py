from __future__ import annotations

import time
from typing import Any, Mapping

import requests

from ..domain import Timings, Usage, to_jsonable
from .base import ProviderRequest, ProviderResponse


class OpenAICompatibleProvider:
    """Provider adapter for non-streaming OpenAI-compatible chat APIs.

    TTFT and pure generation timings remain ``None`` unless the upstream API
    returns them. That prevents end-to-end throughput from being mislabeled as
    inference throughput.
    """

    def __init__(
        self,
        *,
        name: str,
        api_url: str,
        api_key: str,
        timeout_seconds: float = 180,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        extra_headers: Mapping[str, str] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not name.strip() or not api_url.strip():
            raise ValueError("Provider name and api_url are required")
        self.name = name
        self.api_url = api_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.extra_headers = dict(extra_headers or {})
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _extra_payload(self, request: ProviderRequest) -> dict[str, Any]:
        return to_jsonable(request.settings.extra)

    def _payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model.id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.settings.temperature,
            "top_p": request.settings.top_p,
            "max_tokens": request.settings.max_tokens,
            **self._extra_payload(request),
        }
        if request.settings.seed is not None:
            payload["seed"] = request.settings.seed
        if request.response_format is not None:
            payload["response_format"] = to_jsonable(request.response_format)
        return payload

    @staticmethod
    def _content(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if not self.api_key:
            return ProviderResponse(
                status="error",
                provider=self.name,
                resolved_model_id=request.model.id,
                error=f"Missing API key for provider {self.name}",
            )

        started = time.perf_counter()
        attempt_log: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            attempt_started = time.perf_counter()
            try:
                response = self.session.post(
                    self.api_url,
                    headers=self._headers(),
                    json=self._payload(request),
                    timeout=self.timeout_seconds,
                )
                request_seconds = time.perf_counter() - attempt_started
            except requests.RequestException as exc:
                request_seconds = time.perf_counter() - attempt_started
                attempt_log.append({"attempt": attempt + 1, "status": "network_error"})
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                return ProviderResponse(
                    status="error",
                    provider=self.name,
                    resolved_model_id=request.model.id,
                    timings=Timings(
                        end_to_end_seconds=time.perf_counter() - started,
                        provider_request_seconds=request_seconds,
                    ),
                    error=f"Network error: {exc}",
                    raw_metadata={"attempts": attempt_log},
                )

            try:
                data = response.json()
            except ValueError:
                data = {}

            attempt_log.append(
                {"attempt": attempt + 1, "status": response.status_code, "seconds": request_seconds}
            )
            if not response.ok:
                if response.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                return ProviderResponse(
                    status="error",
                    provider=self.name,
                    resolved_model_id=request.model.id,
                    timings=Timings(
                        end_to_end_seconds=time.perf_counter() - started,
                        provider_request_seconds=request_seconds,
                    ),
                    error=f"HTTP {response.status_code}: {str(data or response.text)[:1000]}",
                    raw_metadata={"attempts": attempt_log},
                )

            choices = data.get("choices") or []
            if not choices:
                return ProviderResponse(
                    status="error",
                    provider=self.name,
                    resolved_model_id=request.model.id,
                    timings=Timings(
                        end_to_end_seconds=time.perf_counter() - started,
                        provider_request_seconds=request_seconds,
                    ),
                    error="Provider response contained no choices",
                    raw_metadata={"attempts": attempt_log},
                )

            choice = choices[0]
            output = self._content((choice.get("message") or {}).get("content"))
            usage_data = data.get("usage") or {}
            generation_seconds = self._float(
                usage_data.get("generation_time") or data.get("generation_time")
            )
            ttft_seconds = self._float(usage_data.get("time_to_first_token") or data.get("time_to_first_token"))
            output_tokens = int(usage_data.get("completion_tokens") or 0)
            inter_token = (
                generation_seconds / max(1, output_tokens - 1)
                if generation_seconds is not None and output_tokens > 1
                else None
            )
            return ProviderResponse(
                status="success" if output else "error",
                output=output,
                provider=str(data.get("provider") or self.name),
                resolved_model_id=str(data.get("model") or request.model.id),
                usage=Usage(
                    input_tokens=int(usage_data.get("prompt_tokens") or 0),
                    output_tokens=output_tokens,
                    total_tokens=int(usage_data.get("total_tokens") or 0),
                    cost_usd=self._float(usage_data.get("cost")),
                ),
                timings=Timings(
                    end_to_end_seconds=time.perf_counter() - started,
                    provider_request_seconds=request_seconds,
                    time_to_first_token_seconds=ttft_seconds,
                    generation_seconds=generation_seconds,
                    inter_token_latency_seconds=inter_token,
                ),
                finish_reason=str(choice.get("finish_reason") or ""),
                error="" if output else "Provider returned empty content",
                raw_metadata={"generation_id": data.get("id", ""), "attempts": attempt_log},
            )

        raise RuntimeError("Unreachable provider retry state")


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(
            name="openrouter",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key=api_key,
            **kwargs,
        )

    def _extra_payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload = super()._extra_payload(request)
        if request.settings.reasoning_enabled:
            payload["reasoning"] = {
                "enabled": True,
                "effort": request.settings.reasoning_effort or "medium",
            }
        else:
            payload["reasoning"] = {"effort": "none", "exclude": True}
        return payload
