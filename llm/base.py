"""Provider-independent contracts and expected LLM errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMCompletion:
    """Normalized text completion returned by any provider."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class HTTPDiagnostic:
    """Safe provider response details suitable for debug output."""

    status: int
    message: str | None
    error_type: str | None
    error_code: str | int | None
    model: str
    endpoint: str


class LLMError(RuntimeError):
    """Base class for expected provider failures."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: HTTPDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class MissingAPIKeyError(LLMError):
    """Raised when the provider API key is absent."""


class AuthenticationError(LLMError):
    """Raised when the provider rejects the API key."""


class AccessDeniedError(LLMError):
    """Raised when the provider denies an otherwise valid HTTP request."""


class APIUnavailableError(LLMError):
    """Raised for network, timeout, or provider availability failures."""


class RateLimitError(LLMError):
    """Raised when the provider rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        diagnostic: HTTPDiagnostic | None = None,
    ) -> None:
        super().__init__(message, diagnostic=diagnostic)
        self.retry_after_seconds = retry_after_seconds


class ModelUnavailableError(LLMError):
    """Raised when the configured model is absent or unavailable."""


class EmptyResponseError(LLMError):
    """Raised when the provider returns no usable answer text."""

    def __init__(
        self, message: str, *, completion: LLMCompletion | None = None,
    ) -> None:
        super().__init__(message)
        self.completion = completion


class LLMClient(ABC):
    """Interface that keeps RAG orchestration independent of the provider."""

    @property
    def provider(self) -> str:
        """Stable identity used to prevent mixing checkpoint experiments."""
        return "unknown"

    def complete_detection(self, **kwargs) -> LLMCompletion:
        """Optional native structured output, with a portable text fallback."""
        return self.complete(**kwargs)

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the configured provider model identifier."""

    @abstractmethod
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        """Generate one answer for the supplied prompts."""
