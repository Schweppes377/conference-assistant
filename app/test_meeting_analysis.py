from __future__ import annotations

import unittest

from app.meeting_analysis import _answer_status
from app.analysis_models import (
    STATUS_INSUFFICIENT_INFORMATION,
    STATUS_PARTIAL_INFORMATION,
    STATUS_POTENTIAL_NONCOMPLIANCE,
)


class MeetingAnalysisPipelineTests(unittest.TestCase):
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
