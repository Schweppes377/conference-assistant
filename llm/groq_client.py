"""Groq Chat Completions adapter behind the generic LLMClient interface."""

from __future__ import annotations

import os
from time import sleep
from typing import Any

import httpx

from llm.base import (
    APIUnavailableError,
    AccessDeniedError,
    AuthenticationError,
    EmptyResponseError,
    HTTPDiagnostic,
    LLMClient,
    LLMCompletion,
    MissingAPIKeyError,
    ModelUnavailableError,
    RateLimitError,
)


class GroqClient(LLMClient):
    """Minimal Groq API implementation using the installed httpx client."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        http_client: httpx.Client | None = None,
        max_retries: int = 0,
        retry_delay_seconds: float = 0.75,
    ) -> None:
        if not api_key.strip():
            raise MissingAPIKeyError(
                "GROQ_API_KEY is missing. Add it to the local .env file or process environment."
            )
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = http_client
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._model_validated = False
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        timeout_seconds: float = 60.0,
        http_client: httpx.Client | None = None,
        max_retries: int = 0,
        retry_delay_seconds: float = 0.75,
    ) -> "GroqClient":
        """Create a client using only GROQ_API_KEY from the environment."""

        api_key = os.getenv("GROQ_API_KEY", "")
        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

    @property
    def model(self) -> str:
        return self._model

    def list_available_models(self) -> list[str]:
        """Return model identifiers currently exposed to this API key."""

        response = self._request("GET", "/models")
        payload = self._response_json(response)
        data = payload.get("data")
        if not isinstance(data, list):
            raise APIUnavailableError("Groq returned an invalid models response")
        models = sorted(
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
        return models

    def validate_model(self) -> None:
        """Fail explicitly when the configured model is unavailable."""

        if self._model_validated:
            return
        models = self.list_available_models()
        if self._model not in models:
            raise ModelUnavailableError(
                f"Groq model {self._model!r} is not available for this API key. "
                "Set GROQ_MODEL or pass --model using an ID returned by --list-models."
            )
        self._model_validated = True

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        """Call Groq without automatic model substitution."""

        self.validate_model()
        response = self._request(
            "POST",
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
        payload = self._response_json(response)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        empty_completion = LLMCompletion(
            content="", model=str(payload.get("model") or self._model),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
        )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmptyResponseError(
                "Groq returned no answer content", completion=empty_completion,
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise EmptyResponseError("Groq returned an empty answer", completion=empty_completion)

        return LLMCompletion(
            content=content.strip(),
            model=str(payload.get("model") or self._model),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        endpoint = f"{self.BASE_URL}{path}"
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._get_client().request(
                    method,
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    **kwargs,
                )
                break
            except httpx.ProxyError as exc:
                raise APIUnavailableError(f"Cannot reach Groq API: {exc}") from exc
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt >= self._max_retries:
                    raise APIUnavailableError(f"Cannot reach Groq API: {exc}") from exc
                sleep(self._retry_delay_seconds * (2**attempt))
            except httpx.RequestError as exc:
                raise APIUnavailableError(f"Groq API request failed: {exc}") from exc

        if response is None:  # pragma: no cover - defensive invariant
            raise APIUnavailableError("Groq API request produced no response")

        diagnostic = _http_diagnostic(
            response,
            model=self._model,
            endpoint=endpoint,
        )
        if response.status_code == 401:
            raise AuthenticationError(
                "Groq rejected GROQ_API_KEY",
                diagnostic=diagnostic,
            )
        if response.status_code == 403:
            raise AccessDeniedError(
                "Groq denied access to the requested API endpoint (HTTP 403)",
                diagnostic=diagnostic,
            )
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            suffix = f" Retry after {retry_after} seconds." if retry_after else ""
            raise RateLimitError(
                f"Groq rate limit exceeded.{suffix}",
                retry_after_seconds=_optional_float(retry_after),
                diagnostic=diagnostic,
            )
        if response.status_code in {400, 404}:
            message = diagnostic.message or f"HTTP {response.status_code}"
            if "model" in message.casefold() or response.status_code == 404:
                raise ModelUnavailableError(
                    f"Groq model {self._model!r} is unavailable: {message}",
                    diagnostic=diagnostic,
                )
            raise APIUnavailableError(
                f"Groq rejected the request: {message}",
                diagnostic=diagnostic,
            )
        if response.status_code >= 500:
            raise APIUnavailableError(
                f"Groq API is temporarily unavailable (HTTP {response.status_code})",
                diagnostic=diagnostic,
            )
        if response.is_error:
            raise APIUnavailableError(
                f"Groq API returned HTTP {response.status_code}",
                diagnostic=diagnostic,
            )
        return response

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            try:
                self._client = httpx.Client(timeout=self._timeout_seconds)
            except (ImportError, RuntimeError) as exc:
                raise APIUnavailableError(
                    f"Cannot initialize Groq HTTP client for the current proxy: {exc}"
                ) from exc
        return self._client

    @staticmethod
    def _response_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise APIUnavailableError("Groq returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise APIUnavailableError("Groq returned an unexpected response")
        return payload


def _http_diagnostic(
    response: httpx.Response,
    *,
    model: str,
    endpoint: str,
) -> HTTPDiagnostic:
    message: str | None = None
    error_type: str | None = None
    error_code: str | int | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = _safe_text(error.get("message"))
            error_type = _safe_text(error.get("type"))
            error_code = _safe_code(error.get("code"))
        elif isinstance(error, str):
            message = _safe_text(error)

        message = message or _safe_text(payload.get("message"))
        error_type = error_type or _safe_text(payload.get("type"))
        error_code = error_code if error_code is not None else _safe_code(payload.get("code"))

    return HTTPDiagnostic(
        status=response.status_code,
        message=message,
        error_type=error_type,
        error_code=error_code,
        model=model,
        endpoint=endpoint,
    )


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split())
    return compact[:1000] if compact else None


def _safe_code(value: Any) -> str | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return _safe_text(value)


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
