"""Configuration for grounded Question → RAG → LLM answers."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True, slots=True)
class QAConfig:
    """Configurable retrieval, context, and generation limits."""

    model: str = DEFAULT_GROQ_MODEL
    top_k: int = 5
    candidate_k: int = 20
    max_context_chars: int = 12_000
    min_relevance_score: float = 0.84
    temperature: float = 0.1
    max_output_tokens: int = 1_200
    request_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("LLM model cannot be empty")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.max_context_chars < 1_000:
            raise ValueError("max_context_chars must be at least 1000")
        if not -1.0 <= self.min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between -1 and 1")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be greater than 0 and at most 2")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    @classmethod
    def from_env(cls) -> "QAConfig":
        """Read optional non-secret overrides from environment variables."""

        return cls(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            top_k=_env_int("RAG_TOP_K", 5),
            candidate_k=_env_int("RAG_CANDIDATE_K", 20),
            max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 12_000),
            min_relevance_score=_env_float("RAG_MIN_SCORE", 0.84),
            temperature=_env_float("LLM_TEMPERATURE", 0.1),
            max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 1_200),
            request_timeout_seconds=_env_float("GROQ_TIMEOUT_SECONDS", 60.0),
        )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)
