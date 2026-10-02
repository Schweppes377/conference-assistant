from __future__ import annotations

import unittest

from app.issue_detection import (
    IssueDetectionConfig,
    NormativeIssueDetector,
    StructuredDetectionError,
)
from app.transcript import TranscriptSegment
from llm.base import LLMClient, LLMCompletion


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    @property
    def model(self) -> str:
        return "fake"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_prompt)
        return LLMCompletion(next(self.responses), self.model, 10, 5, 15)


def _transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(0.0, 2.0, "SPEAKER_00", "Площадь квартир 520 квадратных метров."),
        TranscriptSegment(2.1, 4.0, "SPEAKER_01", "Можно оставить один выход?"),
        TranscriptSegment(4.1, 6.0, "SPEAKER_00", "Высота верхнего этажа 25 метров."),
        TranscriptSegment(6.1, 7.0, "SPEAKER_01", "Хорошо, идём дальше."),
    ]


class NormativeIssueDetectorTests(unittest.TestCase):
    def test_neighbor_context_is_traced_to_real_segments(self) -> None:
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0001","SEGMENT_0002",'
                '"SEGMENT_0003"],"normalized_question":"Допускается ли один '
                'эвакуационный выход при площади квартир 520 м² и высоте 25 м?",'
                '"category":"evacuation","reason":"Обсуждается число выходов с '
                'конкретными параметрами здания."}]}'
            ]
        )
        result = NormativeIssueDetector(llm_client=llm).detect(_transcript())

        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(issue.speakers, ["SPEAKER_00", "SPEAKER_01"])
        self.assertIn("520", issue.normalized_question)
        self.assertIn("25", issue.normalized_question)
        self.assertNotIn("идём дальше", issue.original_text)
        self.assertEqual(issue.start, 0.0)
        self.assertEqual(issue.end, 6.0)

    def test_measurement_omitted_by_llm_is_restored_from_transcript(self) -> None:
        transcript = [
            TranscriptSegment(
                0.0,
                2.0,
                "SPEAKER_00",
                "Высота эвакуационного выхода в свету метр восемьдесят пять.",
            )
        ]
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0001"],'
                '"normalized_question":"Какова минимальная высота дверного проема?",'
                '"category":"evacuation","reason":"Указана измеряемая высота."}]}'
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(transcript)

        self.assertIn("метр восемьдесят пять", result.issues[0].normalized_question)
        self.assertIn("в свету", result.issues[0].normalized_question)
        self.assertIn("Соответствует ли", result.issues[0].normalized_question)

    def test_specific_transcript_phrase_is_preserved_without_domain_mapping(self) -> None:
        transcript = [
            TranscriptSegment(
                0.0,
                2.0,
                "SPEAKER_00",
                "На этаже будут вне квартирные кладовые для жильцов.",
            )
        ]
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0001"],'
                '"normalized_question":"Какова ширина пути эвакуации из кладовых?",'
                '"category":"evacuation","reason":"Проверяется ширина пути."}]}'
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(transcript)

        self.assertIn(
            "На этаже будут вне квартирные кладовые для жильцов",
            result.issues[0].search_terms,
        )

    def test_search_terms_are_preserved_for_query_expansion(self) -> None:
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0001","SEGMENT_0002"],'
                '"normalized_question":"Допускается ли один выход при площади 520 м²?",'
                '"search_terms":["площадь квартирного этажа",'
                '"эвакуационные выходы многоквартирного дома"],'
                '"category":"evacuation","reason":"Проверяется число выходов."}]}'
            ]
        )

        issue = NormativeIssueDetector(llm_client=llm).detect(_transcript()).issues[0]

        self.assertEqual(
            issue.search_terms[:2],
            [
                "площадь квартирного этажа",
                "эвакуационные выходы многоквартирного дома",
            ],
        )
        self.assertLessEqual(len(issue.search_terms), 6)
        self.assertIn("search_terms", llm.system_prompts[0])

    def test_unknown_segment_id_is_rejected(self) -> None:
        invalid_response = (
            '{"issues":[{"segment_ids":["SEGMENT_9999"],'
            '"normalized_question":"Нужно ли проверить этот параметр по нормативам?",'
            '"category":"other","reason":"Есть нормативный вопрос."}]}'
        )
        llm = FakeLLM([invalid_response, invalid_response])
        with self.assertRaises(StructuredDetectionError):
            NormativeIssueDetector(llm_client=llm).detect(_transcript())
        self.assertEqual(len(llm.prompts), 2)

    def test_unknown_segment_id_is_removed_when_real_trace_remains(self) -> None:
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0001","SEGMENT_9999"],'
                '"normalized_question":"Какие требования относятся к площади 520 м2?",'
                '"search_terms":["площадь квартир"],'
                '"category":"evacuation","reason":"Проверка площади."}]}'
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(_transcript())

        self.assertEqual(result.issues[0].segment_ids, ["SEGMENT_0001"])
        self.assertEqual(len(llm.prompts), 1)

    def test_numeric_question_with_wrong_segment_ids_is_retried_once(self) -> None:
        transcript = [
            TranscriptSegment(0.0, 2.0, "SPEAKER_00", "Общие требования."),
            TranscriptSegment(
                2.1,
                4.0,
                "SPEAKER_00",
                "Площадь этажа 520 квадратных метров, высота 25 метров.",
            ),
        ]
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0001"],'
                '"normalized_question":"Допустим ли один выход при площади 520 м2 '
                'и высоте 25 м?","search_terms":["эвакуационный выход"],'
                '"category":"evacuation","reason":"Проверка выхода."}]}',
                '{"issues":[{"segment_ids":["SEGMENT_0002"],'
                '"normalized_question":"Допустим ли один выход при площади 520 м2 '
                'и высоте 25 м?","search_terms":["эвакуационный выход"],'
                '"category":"evacuation","reason":"Проверка выхода."}]}',
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(transcript)

        self.assertEqual(result.issues[0].segment_ids, ["SEGMENT_0002"])
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("ТРАССИРУЕМОСТИ", llm.prompts[1])

    def test_overlapping_window_duplicates_are_merged(self) -> None:
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0001","SEGMENT_0002",'
                '"SEGMENT_0003"],'
                '"normalized_question":"Допускается ли один эвакуационный выход при '
                'площади 520 м²?","category":"evacuation","reason":"Вопрос о выходе."}]}',
                '{"issues":[{"segment_ids":["SEGMENT_0003"],'
                '"normalized_question":"Допускается ли один эвакуационный выход?",'
                '"category":"evacuation",'
                '"reason":"Повтор того же вопроса."}]}',
            ]
        )
        detector = NormativeIssueDetector(
            llm_client=llm,
            config=IssueDetectionConfig(window_segments=3, overlap_segments=1),
        )
        result = detector.detect(_transcript())

        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.metrics.windows, 2)
        self.assertEqual(result.metrics.total_tokens, 30)

    def test_adjacent_parts_of_one_engineering_topic_are_merged(self) -> None:
        transcript = [
            TranscriptSegment(0.0, 2.0, "SPEAKER_00", "Какие требования к вентиляции?"),
            TranscriptSegment(2.1, 4.0, "SPEAKER_00", "Вентиляция естественная."),
            TranscriptSegment(4.1, 6.0, "SPEAKER_00", "Можно объединить вентканалы квартир?"),
        ]
        llm = FakeLLM(
            [
                '{"issues":['
                '{"segment_ids":["SEGMENT_0001","SEGMENT_0002"],'
                '"normalized_question":"Какие требования к естественной вентиляции?",'
                '"search_terms":["естественная вентиляция"],'
                '"category":"ventilation","reason":"Проверка вентиляции."},'
                '{"segment_ids":["SEGMENT_0003"],'
                '"normalized_question":"Можно ли объединять вентиляционные каналы?",'
                '"search_terms":["вентиляционные каналы"],'
                '"category":"ventilation","reason":"Проверка каналов."}'
                ']}'
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(transcript)

        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].segment_ids, [
            "SEGMENT_0001",
            "SEGMENT_0002",
            "SEGMENT_0003",
        ])

    def test_adjacent_different_objects_are_not_merged(self) -> None:
        transcript = [
            TranscriptSegment(0.0, 2.0, "SPEAKER_00", "Ширина лестничного марша метр."),
            TranscriptSegment(2.1, 4.0, "SPEAKER_00", "Коридор кладовых 80 сантиметров."),
        ]
        llm = FakeLLM(
            [
                '{"issues":['
                '{"segment_ids":["SEGMENT_0001"],'
                '"normalized_question":"Какова ширина лестничного марша?",'
                '"search_terms":["лестничный марш"],'
                '"category":"evacuation","reason":"Проверка лестницы."},'
                '{"segment_ids":["SEGMENT_0002"],'
                '"normalized_question":"Какова ширина коридора кладовых?",'
                '"search_terms":["коридор кладовых"],'
                '"category":"evacuation","reason":"Проверка кладовых."}'
                ']}'
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(transcript)

        self.assertEqual(len(result.issues), 2)

    def test_late_recap_of_multiple_topics_is_not_a_new_issue(self) -> None:
        transcript = [
            TranscriptSegment(0.0, 2.0, "SPEAKER_00", "Проверим ширину лестницы."),
            TranscriptSegment(3.0, 5.0, "SPEAKER_00", "Проверим вентиляцию квартир."),
            TranscriptSegment(
                20.0,
                24.0,
                "SPEAKER_00",
                "Подведем итог: нам нужно проверить ширину лестницы и вентиляцию.",
            ),
        ]
        llm = FakeLLM(
            [
                '{"issues":['
                '{"segment_ids":["SEGMENT_0001"],'
                '"normalized_question":"Какова ширина лестничного марша?",'
                '"search_terms":["лестничный марш"],'
                '"category":"evacuation","reason":"Проверка лестницы."},'
                '{"segment_ids":["SEGMENT_0002"],'
                '"normalized_question":"Какие требования к вентиляции квартир?",'
                '"search_terms":["вентиляция квартир"],'
                '"category":"ventilation","reason":"Проверка вентиляции."},'
                '{"segment_ids":["SEGMENT_0003"],'
                '"normalized_question":"Какие требования к ширине лестницы и вентиляции?",'
                '"search_terms":["ширина лестницы","вентиляция"],'
                '"category":"other","reason":"Итоговое перечисление."}'
                ']}'
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(transcript)

        self.assertEqual(len(result.issues), 2)
        self.assertNotIn("Итоговое перечисление", [item.reason for item in result.issues])

    def test_english_question_is_retried_once_in_russian(self) -> None:
        llm = FakeLLM(
            [
                '{"issues":[{"segment_ids":["SEGMENT_0002"],'
                '"normalized_question":"What evacuation exits are required?",'
                '"category":"evacuation","reason":"This requires a code check."}]}',
                '{"issues":[{"segment_ids":["SEGMENT_0002"],'
                '"normalized_question":"Какие эвакуационные выходы требуются?",'
                '"category":"evacuation","reason":"Требуется проверка по нормативам."}]}',
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(_transcript())

        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("Верни результат на русском языке", llm.prompts[1])
        self.assertIn("Какие эвакуационные", result.issues[0].normalized_question)
        self.assertEqual(result.metrics.total_tokens, 30)

    def test_second_english_response_is_rejected_without_more_retries(self) -> None:
        response = (
            '{"issues":[{"segment_ids":["SEGMENT_0002"],'
            '"normalized_question":"What evacuation exits are required?",'
            '"category":"evacuation","reason":"This requires a code check."}]}'
        )
        llm = FakeLLM([response, response])

        with self.assertRaisesRegex(StructuredDetectionError, "not in Russian"):
            NormativeIssueDetector(llm_client=llm).detect(_transcript())

        self.assertEqual(len(llm.prompts), 2)

    def test_invalid_json_is_retried_once(self) -> None:
        llm = FakeLLM(
            [
                '{"issues":[',
                '{"issues":[]}',
            ]
        )

        result = NormativeIssueDetector(llm_client=llm).detect(_transcript())

        self.assertEqual(result.issues, [])
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("валидный JSON", llm.prompts[1])


if __name__ == "__main__":
    unittest.main()
