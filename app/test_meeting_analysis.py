from __future__ import annotations

import unittest

from app.meeting_analysis import MeetingAnalysisPipeline, _answer_status
from app.analysis_models import (
    STATUS_INSUFFICIENT_INFORMATION,
    STATUS_PARTIAL_INFORMATION,
    STATUS_POTENTIAL_NONCOMPLIANCE,
)
from llm.base import EmptyResponseError, RateLimitError


class RateLimitedQAService:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, *, search_terms: list[str]) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RateLimitError(
                "rate limited",
                retry_after_seconds=2.0,
            )
        return "grounded result"


class EmptyOnceQAService:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, *, search_terms: list[str]) -> str:
        self.calls += 1
        if self.calls == 1:
            raise EmptyResponseError("empty")
        return "grounded result"


class MeetingAnalysisPipelineTests(unittest.TestCase):
    def test_rate_limit_retries_one_issue_without_losing_it(self) -> None:
        service = RateLimitedQAService()
        delays: list[float] = []
        pipeline = MeetingAnalysisPipeline(
            transcription_pipeline=object(),  # type: ignore[arg-type]
            issue_detector=object(),  # type: ignore[arg-type]
            qa_service=service,  # type: ignore[arg-type]
            sleeper=delays.append,
        )

        result = pipeline._answer_with_rate_limit_retry("question", ["термин"])

        self.assertEqual(result, "grounded result")
        self.assertEqual(service.calls, 2)
        self.assertEqual(delays, [2.25])

    def test_empty_answer_retries_whole_qa_once(self) -> None:
        service = EmptyOnceQAService()
        pipeline = MeetingAnalysisPipeline(
            transcription_pipeline=object(),  # type: ignore[arg-type]
            issue_detector=object(),  # type: ignore[arg-type]
            qa_service=service,  # type: ignore[arg-type]
        )

        result = pipeline._answer_with_rate_limit_retry("question", ["термин"])

        self.assertEqual(result, "grounded result")
        self.assertEqual(service.calls, 2)

    def test_partial_normative_answer_is_marked_insufficient(self) -> None:
        status = _answer_status(
            insufficient=False,
            partial=False,
            has_sources=True,
            answer="В предоставленных фрагментах их значения не указаны.",
        )

        self.assertEqual(status, STATUS_INSUFFICIENT_INFORMATION)

    def test_missing_requested_limits_are_marked_insufficient(self) -> None:
        status = _answer_status(
            insufficient=False,
            partial=False,
            has_sources=True,
            answer=(
                "Общее требование найдено. Никаких дополнительных ограничений "
                "в приведённых нормативных фрагментах не указано."
            ),
        )

        self.assertEqual(status, STATUS_INSUFFICIENT_INFORMATION)

    def test_explicit_noncompliance_wording_is_marked_potential(self) -> None:
        status = _answer_status(
            insufficient=False,
            partial=False,
            has_sources=True,
            answer="При ширине коридора 0,8 м требования не соблюдаются.",
        )

        self.assertEqual(status, STATUS_POTENTIAL_NONCOMPLIANCE)

    def test_partial_context_has_distinct_status(self) -> None:
        status = _answer_status(
            insufficient=False,
            partial=True,
            has_sources=True,
            answer="Найдена только часть требований [SOURCE_1].",
        )

        self.assertEqual(status, STATUS_PARTIAL_INFORMATION)

if __name__ == "__main__":
    unittest.main()
