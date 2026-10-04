from __future__ import annotations

import unittest

from llm.base import LLMClient, LLMCompletion
from llm.config import QAConfig
from llm.models import CitationValidationError, ContextValidationError, SourceMetadataError
from llm.prompts import INSUFFICIENT_CONTEXT_ANSWER, PARTIAL_CONTEXT_MARKER
from llm.service import QuestionAnsweringService
from rag.models import SearchResult


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        return self.results[:top_k]


class QueryAwareFakeRetriever:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        self.queries.append(query)
        return self.results_by_query.get(query, [])[:top_k]


class FakeLLMClient(LLMClient):
    def __init__(self, answer: str | list[str]) -> None:
        answers = [answer] if isinstance(answer, str) else answer
        self.answers = iter(answers)
        self.called = False
        self.calls = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        self.called = True
        self.calls += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return LLMCompletion(next(self.answers), self.model, 100, 20, 120)


def _result(
    *,
    title: str,
    designation: str,
    page: int,
    chunk_index: int,
    text: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        distance=1.0 - score,
        text=text,
        metadata={
            "source_file": f"doc_{chunk_index}.pdf",
            "title": title,
            "designation": designation,
            "page": page,
            "chunk_index": chunk_index,
        },
    )


class QuestionAnsweringServiceTests(unittest.TestCase):
    def test_grounded_answer_uses_real_metadata(self) -> None:
        result = _result(
            title="Эвакуационные пути и выходы",
            designation="СП 1.13130.2020",
            page=17,
            chunk_index=42,
            text="Эвакуационный выход должен соответствовать требованиям раздела.",
            score=0.92,
        )
        llm = FakeLLMClient("Требование приведено в источнике [SOURCE_1].")
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=llm,
            config=QAConfig(),
        )

        answer = service.answer("Какие требования к эвакуационному выходу?")

        self.assertTrue(llm.called)
        self.assertEqual(answer.sources[0].page, 17)
        self.assertEqual(answer.sources[0].designation, "СП 1.13130.2020")
        self.assertIn("только на основании", llm.last_system_prompt)
        self.assertIn("[SOURCE_1]", llm.last_user_prompt)

    def test_irrelevant_results_return_insufficient_without_llm(self) -> None:
        result = _result(
            title="Доступность зданий",
            designation="СП 59.13330.2020",
            page=4,
            chunk_index=1,
            text="Требования к доступности входа.",
            score=0.70,
        )
        llm = FakeLLMClient("Не должен вызываться")
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=llm,
            config=QAConfig(min_relevance_score=0.84),
        )

        answer = service.answer("Как приготовить борщ?")

        self.assertFalse(llm.called)
        self.assertTrue(answer.insufficient_context)
        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.sources, [])
        self.assertEqual(len(answer.retrieval_results), 1)

    def test_ambiguous_retrieval_maps_only_cited_source(self) -> None:
        first = _result(
            title="Эвакуационные пути",
            designation="СП 1.13130.2020",
            page=8,
            chunk_index=2,
            text="Общие требования к путям эвакуации.",
            score=0.91,
        )
        second = _result(
            title="Здания жилые многоквартирные",
            designation="СП 54.13330.2022",
            page=30,
            chunk_index=7,
            text="Требования к жилым многоквартирным зданиям.",
            score=0.89,
        )
        llm = FakeLLMClient("Для жилого здания применим второй фрагмент [SOURCE_2].")
        service = QuestionAnsweringService(
            retriever=FakeRetriever([first, second]),
            llm_client=llm,
            config=QAConfig(),
        )

        answer = service.answer("Какие требования относятся именно к жилому дому?")

        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].designation, "СП 54.13330.2022")
        self.assertEqual(len(answer.retrieval_results), 2)
        self.assertIn("[SOURCE_1]", llm.last_user_prompt)
        self.assertIn("[SOURCE_2]", llm.last_user_prompt)

    def test_citation_with_internal_spaces_is_resolved(self) -> None:
        result = _result(
            title="Здания жилые многоквартирные",
            designation="СП 54.13330.2022",
            page=30,
            chunk_index=8,
            text="Прямое нормативное требование.",
            score=0.91,
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=FakeLLMClient("Требование найдено [ SOURCE_1 ]."),
            config=QAConfig(),
        )

        answer = service.answer("Какое требование применяется?")

        self.assertEqual(len(answer.cited_sources), 1)

    def test_invented_source_id_is_rejected(self) -> None:
        result = _result(
            title="Эвакуационные пути",
            designation="СП 1.13130.2020",
            page=8,
            chunk_index=2,
            text="Общие требования к путям эвакуации.",
            score=0.91,
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=FakeLLMClient(
                [
                    "Недопустимая ссылка [SOURCE_99].",
                    "Недопустимая ссылка [SOURCE_99].",
                ]
            ),
            config=QAConfig(),
        )

        with self.assertRaises(CitationValidationError):
            service.answer("Вопрос")

    def test_corrupted_metadata_is_rejected(self) -> None:
        broken = SearchResult(
            distance=0.05,
            text="Фрагмент без номера страницы.",
            metadata={
                "source_file": "broken.pdf",
                "title": "Повреждённый источник",
                "designation": "СП TEST",
                "chunk_index": 1,
            },
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([broken]),
            llm_client=FakeLLMClient("Не должен вызываться"),
            config=QAConfig(),
        )

        with self.assertRaises(SourceMetadataError):
            service.answer("Вопрос")

    def test_query_expansion_and_reranking_prioritize_specific_object(self) -> None:
        question = "Какова ширина пути эвакуации из кладовых?"
        term = "вне квартирные кладовые для жильцов"
        general = _result(
            title="Общие требования",
            designation="СП 1.13130.2020",
            page=11,
            chunk_index=1,
            text="Ширина общих горизонтальных путей эвакуации должна быть 1,0 м.",
            score=0.93,
        )
        specific = _result(
            title="Здания жилые многоквартирные",
            designation="СП 54.13330.2022",
            page=30,
            chunk_index=2,
            text=(
                "Ширину путей эвакуации из внеквартирных хозяйственных "
                "кладовых жильцов следует предусматривать не менее 0,9 м."
            ),
            score=0.86,
        )
        retriever = QueryAwareFakeRetriever(
            {
                question: [general],
                term: [specific],
            }
        )
        service = QuestionAnsweringService(
            retriever=retriever,
            llm_client=FakeLLMClient("Требуется не менее 0,9 м [SOURCE_1]."),
            config=QAConfig(top_k=1, candidate_k=20),
        )

        answer = service.answer(question, search_terms=[term])

        self.assertEqual(retriever.queries, [question, term])
        self.assertEqual(answer.retrieved_candidates[0].designation, "СП 54.13330.2022")
        self.assertEqual(answer.cited_sources[0].designation, "СП 54.13330.2022")
        self.assertIn(term, answer.retrieved_candidates[0].matched_terms)

    def test_partial_answer_is_exposed_separately(self) -> None:
        result = _result(
            title="Защита от шума",
            designation="СП 54.13330.2022",
            page=42,
            chunk_index=3,
            text="Защиту от шума предусматривают по СП 51.13330.",
            score=0.91,
        )
        llm = FakeLLMClient(
            "Общее требование найдено [SOURCE_1]. "
            + PARTIAL_CONTEXT_MARKER
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=llm,
            config=QAConfig(),
        )

        answer = service.answer("Какие уровни шума допустимы днем и ночью?")

        self.assertTrue(answer.partial_context)
        self.assertFalse(answer.insufficient_context)

    def test_missing_numeric_subpart_is_classified_as_partial(self) -> None:
        result = _result(
            title="Защита от шума",
            designation="СП 54.13330.2022",
            page=42,
            chunk_index=30,
            text="Общее требование по защите от шума.",
            score=0.91,
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=FakeLLMClient(
                "Общее требование найдено [SOURCE_1]. Конкретные числовые "
                "дневные и ночные значения в фрагменте не указаны."
            ),
            config=QAConfig(),
        )

        answer = service.answer("Какие уровни действуют днем и ночью?")

        self.assertTrue(answer.partial_context)
        self.assertFalse(answer.insufficient_context)

    def test_day_and_night_question_without_two_limits_is_partial(self) -> None:
        result = _result(
            title="Защита от шума",
            designation="СП TEST",
            page=10,
            chunk_index=33,
            text=(
                "Уровень шума не должен более чем на 2 дБА превышать фоновые "
                "значения в дневное и ночное время."
            ),
            score=0.91,
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=FakeLLMClient("Общее ограничение действует [SOURCE_1]."),
            config=QAConfig(),
        )

        answer = service.answer("Какие дневные и ночные значения уровня шума?")

        self.assertTrue(answer.partial_context)
        self.assertIn(PARTIAL_CONTEXT_MARKER, answer.answer)

    def test_measured_value_absent_from_cited_chunk_is_rejected(self) -> None:
        result = _result(
            title="Эвакуационные пути",
            designation="СП 1.13130.2020",
            page=11,
            chunk_index=4,
            text="Ширина пути определяется расчетом.",
            score=0.91,
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=FakeLLMClient(
                [
                    "Требуется не менее 0,9 м [SOURCE_1].",
                    "Требуется не менее 0,9 м [SOURCE_1].",
                ]
            ),
            config=QAConfig(),
        )

        with self.assertRaises(CitationValidationError):
            service.answer("Какова ширина пути эвакуации?")

    def test_equivalent_length_units_pass_numeric_validation(self) -> None:
        result = _result(
            title="Здания жилые многоквартирные",
            designation="СП 54.13330.2022",
            page=30,
            chunk_index=5,
            text="Ширину пути следует принимать не менее 0,9 м.",
            score=0.91,
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=FakeLLMClient(
                "Проектные 0,80 м меньше требуемых 0,9 м [SOURCE_1]."
            ),
            config=QAConfig(),
        )

        answer = service.answer("Соответствует ли ширина 80 см требованию?")

        self.assertEqual(len(answer.cited_sources), 1)

    def test_value_absent_from_all_context_does_not_retry(self) -> None:
        result = _result(
            title="Защита от шума",
            designation="СП 54.13330.2022",
            page=42,
            chunk_index=6,
            text="Уровни шума не должны превышать допустимых значений.",
            score=0.91,
        )
        llm = FakeLLMClient(
            [
                "Допустимо 25 дБ [SOURCE_1].",
                "Общее требование найдено [SOURCE_1]. " + PARTIAL_CONTEXT_MARKER,
            ]
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([result]),
            llm_client=llm,
            config=QAConfig(),
        )

        with self.assertRaises(ContextValidationError):
            service.answer("Каковы дневные и ночные уровни шума?")
        self.assertEqual(llm.calls, 1)

    def test_retry_can_correct_citation_to_chunk_containing_number(self) -> None:
        general = _result(
            title="Общие требования",
            designation="СП TEST",
            page=1,
            chunk_index=31,
            text="Ширина определяется нормативными требованиями.",
            score=0.99,
        )
        numeric = _result(
            title="Числовое требование",
            designation="СП TEST",
            page=2,
            chunk_index=32,
            text="Минимальная ширина составляет 0,9 м.",
            score=0.85,
        )
        llm = FakeLLMClient(
            [
                "Минимальная ширина 0,9 м [SOURCE_1].",
                "Минимальная ширина 0,9 м [SOURCE_2].",
            ]
        )
        service = QuestionAnsweringService(
            retriever=FakeRetriever([general, numeric]),
            llm_client=llm,
            config=QAConfig(),
        )

        answer = service.answer("Какова минимальная ширина?")

        self.assertEqual(llm.calls, 2)
        self.assertEqual(answer.cited_sources[0].page, 2)


if __name__ == "__main__":
    unittest.main()
