"""Generation budget, immediate usage accounting and resumable response cache."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from typing import Any, Callable

from llm.base import (
    APIUnavailableError, EmptyResponseError, LLMClient, LLMCompletion,
    LLMError, RateLimitError,
)


class LLMBudgetExceeded(LLMError):
    """The next generation is blocked locally before contacting the provider."""


class GenerationLimitExceeded(LLMError):
    """An operation has already used its two generation responses."""


class RunLLMClient(LLMClient):
    """Track every response before validation; replay saved responses on resume.

    This adapter has no network retry loop. A provider failure propagates to
    the meeting runner, which saves progress and pauses the whole run.
    """

    def __init__(
        self, client: LLMClient, *, state: dict[str, Any] | None = None,
        max_calls: int = 1, on_change: Callable[[], None] | None = None,
    ) -> None:
        if max_calls < 1:
            raise ValueError("MAX_LLM_CALLS_PER_RUN must be positive")
        self.client = client
        self.state = state if state is not None else {}
        self.state.setdefault("events", [])
        self.state.setdefault("responses", {})
        self.max_calls = max_calls
        self.run_calls = 0
        self.on_change = on_change or (lambda: None)

    @property
    def model(self) -> str:
        return self.client.model

    def complete(self, **kwargs: Any) -> LLMCompletion:
        return self.generate(
            stage="answer", operation=f"standalone:{len(self.state['events'])}",
            attempt=0, **kwargs,
        )

    def generate(
        self, *, stage: str, operation: str, attempt: int, **kwargs: Any,
    ) -> LLMCompletion:
        if not 0 <= attempt < 2:
            raise GenerationLimitExceeded("Допускается не более двух генераций.")
        key = f"{stage}:{operation}:{attempt}"
        fingerprint = hashlib.sha256(json.dumps(
            {"model": self.model, **kwargs}, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        saved = self.state["responses"].get(key)
        if saved is not None:
            if saved["fingerprint"] != fingerprint:
                raise ValueError(
                    "Настройки или контекст изменились после checkpoint. "
                    "Для нового эксперимента используйте --transcript/--issues, "
                    "а не --resume."
                )
            return LLMCompletion(**saved["completion"])
        if self.run_calls >= self.max_calls:
            raise LLMBudgetExceeded(
                f"Лимит LLM-вызовов запуска исчерпан ({self.max_calls})."
            )

        self.run_calls += 1
        event: dict[str, Any] = {
            "stage": stage, "operation": operation, "corrective": attempt == 1,
            "successful": False, "prompt_tokens": None,
            "completion_tokens": None, "total_tokens": None,
        }
        self.state["events"].append(event)
        self.on_change()
        try:
            completion = self.client.complete(**kwargs)
        except EmptyResponseError as exc:
            # HTTP 200 with unusable text is still a paid, successful response.
            completion = getattr(exc, "completion", None)
            if completion is None:
                event["error_type"] = "unknown_empty_response"
                self.on_change()
                raise
        except LLMError as exc:
            event["error_type"] = type(exc).__name__
            self.on_change()
            raise
        event.update({
            "successful": True,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
        })
        self.state["responses"][key] = {
            "fingerprint": fingerprint, "completion": asdict(completion),
        }
        self.on_change()
        return completion

    def metrics(self) -> dict[str, Any]:
        events = self.state["events"]
        def tokens(items: list[dict[str, Any]]) -> dict[str, int | None]:
            result = {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                values = [e[name] for e in items if isinstance(e[name], int)]
                result[name] = sum(values) if values else None
            return result
        return {
            **tokens(events),
            "detection_tokens": tokens([e for e in events if e["stage"] == "detection"]),
            "answer_tokens": tokens([e for e in events if e["stage"] == "answer"]),
            # retry_tokens is a subset, not an extra addend to total_tokens.
            "retry_tokens": tokens([e for e in events if e["corrective"]]),
            "llm_calls": len(events),
            "successful_llm_calls": sum(e["successful"] for e in events),
            "corrective_retries": sum(e["corrective"] for e in events),
            "run_llm_calls": self.run_calls,
            "max_llm_calls_per_run": self.max_calls,
            "usage_incomplete": any(
                (e["successful"] and e["total_tokens"] is None)
                or e.get("error_type") in ("APIUnavailableError", "unknown_empty_response")
                for e in events
            ),
        }


def generate(
    client: LLMClient, *, stage: str, operation: str, attempt: int, **kwargs: Any,
) -> LLMCompletion:
    """Use a tracked session when present, otherwise preserve client contract."""
    if isinstance(client, RunLLMClient):
        return client.generate(stage=stage, operation=operation, attempt=attempt, **kwargs)
    return client.complete(**kwargs)


def pause_message(exc: Exception) -> str:
    """Render safe API diagnostics and delay without sleeping in the CLI."""
    if isinstance(exc, RateLimitError):
        delay = exc.retry_after_seconds
        diagnostic = exc.diagnostic
        message = diagnostic.message if diagnostic else None
        if delay is None and message:
            match = re.search(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
            if match:
                delay = int(match[1] or 0) * 60 + float(match[2])
        text = "Достигнут лимит Groq."
        if delay is not None:
            minutes, seconds = divmod(math.ceil(delay), 60)
            text += f" Повторный запрос возможен примерно через {minutes} мин {seconds} с."
        else:
            text += " Время ожидания API не указано; продолжите позже."
        if diagnostic:
            limit = next((kind for kind in ("TPD", "ITPM", "OTPM", "TPM", "RPD", "RPM")
                          if message and re.search(rf"\b{kind}\b", message)), None)
            text += f" HTTP {diagnostic.status}."
            if limit:
                text += f" Тип лимита: {limit}."
        return text
    if isinstance(exc, APIUnavailableError):
        return f"API временно недоступен. Прогресс сохранён: {exc}"
    return str(exc)
