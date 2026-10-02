"""Provider-independent LLM layer for grounded normative Q&A."""

from llm.base import LLMClient
from llm.models import QAResult

__all__ = ["LLMClient", "QAResult"]
