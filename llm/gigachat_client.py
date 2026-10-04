"""Official GigaChat SDK adapter; orchestration remains provider-independent."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import httpx
from gigachat import GigaChat
from gigachat.exceptions import ResponseError
from gigachat.models.chat import Chat, Messages, MessagesRole
from gigachat.models.response_format import JsonSchemaResponseFormat

from llm.base import (
    APIUnavailableError, AccessDeniedError, AuthenticationError, EmptyResponseError,
    HTTPDiagnostic, LLMClient, LLMCompletion, MissingAPIKeyError,
    ModelUnavailableError, RateLimitError,
)


class GigaChatClient(LLMClient):
    """One generation per complete(); OAuth is managed by the official SDK."""

    BASE_URL = "https://api.giga.chat/v1"
    SCOPE = "GIGACHAT_API_PERS"

    def __init__(self, *, credentials: str, model: str, timeout_seconds: float = 60,
                 sdk_client: Any = None, verify_ssl_certs: bool = False) -> None:
        if not credentials.strip():
            raise MissingAPIKeyError("Отсутствует SBER_ULTRA в окружении или локальном .env.")
        self._credentials = credentials
        self._model = model
        try:
            self._sdk = sdk_client if sdk_client is not None else GigaChat(
                base_url=self.BASE_URL, credentials=credentials, scope=self.SCOPE,
                model=model, timeout=timeout_seconds, verify_ssl_certs=verify_ssl_certs,
                max_retries=0, access_token=None, user=None, password=None,
            )
        except Exception:
            # SDK exceptions may include settings/credentials; never render them.
            raise APIUnavailableError("Не удалось создать клиент GigaChat.") from None

    @classmethod
    def from_env(cls, *, model: str, timeout_seconds: float = 60,
                 sdk_client: Any = None) -> "GigaChatClient":
        return cls(credentials=os.getenv("SBER_ULTRA", ""), model=model,
                   timeout_seconds=timeout_seconds, sdk_client=sdk_client,
                   verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() == "true")

    @property
    def provider(self) -> str:
        return "gigachat"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float,
                 max_tokens: int) -> LLMCompletion:
        return self._complete(system_prompt=system_prompt, user_prompt=user_prompt,
                              temperature=temperature, max_tokens=max_tokens)

    def complete_detection(self, **kwargs: Any) -> LLMCompletion:
        return self._complete(**kwargs, response_format=JsonSchemaResponseFormat(
            type="json_schema", strict=True, schema=DETECTION_SCHEMA,
        ))

    def _complete(self, *, system_prompt: str, user_prompt: str, temperature: float,
                  max_tokens: int, response_format: Any = None) -> LLMCompletion:
        # The SDK's chat.create uses /v2 even with base_url=/v1.
        # Use its supported v1 chat interface for the requested endpoint.
        request = Chat(
            model=self.model,
            messages=[Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                      Messages(role=MessagesRole.USER, content=user_prompt)],
            temperature=temperature, max_tokens=max_tokens, response_format=response_format,
        )
        response = self._call(lambda: self._sdk.chat(request), "/chat/completions")
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", [])
        content = getattr(choices[0].message, "content", "") if choices else ""
        content = content.strip() if isinstance(content, str) else ""
        completion = LLMCompletion(
            content=content, model=getattr(response, "model", None) or self.model,
            prompt_tokens=_count(usage, "input_tokens", "prompt_tokens"),
            completion_tokens=_count(usage, "output_tokens", "completion_tokens"),
            total_tokens=_count(usage, "total_tokens"),
        )
        if not content:
            raise EmptyResponseError("GigaChat вернул пустой ответ.", completion=completion)
        return completion

    def list_available_models(self) -> list[str]:
        response = self._call(self._sdk.get_models, "/models")
        return sorted(item.id_ for item in response.data)

    def close(self) -> None:
        self._sdk.close()

    def _call(self, operation: Any, path: str) -> Any:
        try:
            return operation()
        except ResponseError as exc:
            status = exc.status_code
            # Never use str(exc): SDK includes response headers and raw body.
            try:
                payload = json.loads(exc.content or b"{}")
            except (ValueError, TypeError):
                payload = {}
            error = payload.get("error", payload) if isinstance(payload, dict) else {}
            if not isinstance(error, dict):
                error = {"message": error}
            diagnostic = HTTPDiagnostic(
                status=status, message=self._safe(error.get("message")),
                error_type=self._safe(error.get("type")), error_code=self._safe(error.get("code")),
                model=self.model,
                endpoint=("https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
                          if "/oauth" in str(exc.url) else self.BASE_URL + path),
            )
            if status == 429:
                raw = exc.headers.get("retry-after") if exc.headers else None
                try:
                    delay = float(raw)
                    if not math.isfinite(delay) or delay < 0:
                        delay = None
                except (ValueError, TypeError):
                    delay = None
                raise RateLimitError("Достигнуто ограничение LLM API.",
                                     retry_after_seconds=delay, diagnostic=diagnostic) from None
            cls = (AuthenticationError if status == 401 else AccessDeniedError if status == 403
                   else ModelUnavailableError if status == 404 or (
                       status == 400 and "model" in (diagnostic.message or "").lower())
                   else APIUnavailableError)
            raise cls(f"GigaChat API: HTTP {status}. Проверьте доступ и параметры модели.",
                      diagnostic=diagnostic) from None
        except (httpx.HTTPError, ValueError, RuntimeError):
            raise APIUnavailableError("Не удалось соединиться с GigaChat API; проверьте сеть и доступность сервиса.") from None
        except Exception:
            raise APIUnavailableError("Не удалось получить корректный ответ GigaChat SDK.") from None

    def _safe(self, value: Any) -> str | None:
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            return None
        text = str(value).replace(self._credentials, "[REDACTED]")
        text = re.sub(r"(?i)\b(?:Bearer|Basic)\s+\S+", "[REDACTED]", text)
        text = re.sub(r"[A-Za-z0-9_+/=-]{32,}", "[REDACTED]", text)
        return " ".join(text.split())[:500] or None


def _count(usage: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(usage, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


# Transport schema mirrors the existing detection JSON, not normative rules.
DETECTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["issues"],
    "properties": {"issues": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["segment_ids", "normalized_question", "search_terms", "category", "reason"],
        "properties": {
            "segment_ids": {"type": "array", "items": {"type": "string"}},
            "normalized_question": {"type": "string"},
            "search_terms": {"type": "array", "items": {"type": "string"}},
            "category": {"type": "string"}, "reason": {"type": "string"},
        },
    }}},
}
